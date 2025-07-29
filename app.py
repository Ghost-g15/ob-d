import os
import sys
import time
import logging
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
import zipfile
from pathlib import Path

# Set up logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Set YOLO environment variables BEFORE importing ultralytics
os.environ['YOLO_CONFIG_DIR'] = '/tmp'
os.environ['ULTRALYTICS_CONFIG_DIR'] = '/tmp'

app = Flask(__name__)

# Configuration
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png', 'gif', 'webm', 'mkv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model variable
model = None
model_loading_error = None

def add_cors_headers(response):
    """Add CORS headers for cross-origin requests"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.after_request
def after_request(response):
    return add_cors_headers(response)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'message': 'OK'})
        return add_cors_headers(response)

def load_yolo_model():
    """Load your custom trained YOLO model"""
    global model, model_loading_error
    
    try:
        logger.info("🔥 Loading custom YOLO fire detection model...")
        
        # Import ultralytics after setting environment variables
        from ultralytics import YOLO
        
        # Try to load your custom model first
        model_paths = [
            './best.pt',  # Your trained model
            'best.pt',
            os.path.join(os.getcwd(), 'best.pt'),
            'yolov8n.pt'  # Fallback to default
        ]
        
        model_loaded = False
        for model_path in model_paths:
            try:
                if os.path.exists(model_path):
                    logger.info(f"Trying to load model from: {model_path}")
                    model = YOLO(model_path)
                    logger.info(f"✅ Model loaded successfully from {model_path}!")
                    logger.info(f"Model classes: {model.names}")
                    model_loaded = True
                    break
                else:
                    logger.info(f"Model file not found: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load {model_path}: {str(e)}")
                continue
        
        if not model_loaded:
            raise Exception("No YOLO model could be loaded")
            
        return True
        
    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        logger.error(error_msg)
        model_loading_error = error_msg
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_fire_with_yolo(file_path, is_video=False):
    """
    Detect fire using your trained YOLO model - same logic as your local setup
    """
    if model is None:
        raise Exception("YOLO model not loaded")
    
    start_time = time.time()
    
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, 'detected')
            
            # Run detection - same as your yolo command
            logger.info(f"Running detection on: {file_path}")
            results = model(file_path, save=True, project=temp_dir, name='detected')
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Parse results - same format as your local setup
            detections = []
            fire_detected = False
            max_confidence = 0
            total_detections = 0
            
            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            fire_detected = True
                            total_detections += 1
                            
                            # Get bounding box coordinates
                            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            
                            # Get class name from your trained model
                            class_name = model.names[class_id] if hasattr(model, 'names') else f'fire_{class_id}'
                            
                            detections.append({
                                'class': class_name,
                                'confidence': round(confidence, 2),
                                'bbox': [round(coord, 2) for coord in bbox],
                                'bbox_x1': round(bbox[0], 2),
                                'bbox_y1': round(bbox[1], 2),
                                'bbox_x2': round(bbox[2], 2),
                                'bbox_y2': round(bbox[3], 2)
                            })
                            
                            if confidence > max_confidence:
                                max_confidence = confidence
            
            # Get output files (images/videos with detections drawn)
            output_files = []
            if os.path.exists(output_dir):
                for file in os.listdir(output_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi')):
                        output_files.append(os.path.join(output_dir, file))
            
            # Create zip file with results if there are output files
            zip_path = None
            if output_files:
                zip_path = os.path.join(tempfile.gettempdir(), f'detection_results_{int(time.time())}.zip')
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for file_path in output_files:
                        zipf.write(file_path, os.path.basename(file_path))
                logger.info(f"Created results zip: {zip_path}")
            
            return {
                'fire_detected': fire_detected,
                'confidence': round(max_confidence, 2),
                'total_detections': total_detections,
                'detections': detections,
                'processing_time': processing_time,
                'output_zip_path': zip_path,
                'has_output_files': len(output_files) > 0,
                'model_info': {
                    'model_name': 'Custom Fire Detection Model (best.pt)',
                    'classes': model.names if model and hasattr(model, 'names') else {}
                }
            }
            
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'Fire Detection API - Your Custom Trained Model',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            'health': '/health',
            'detect': '/detect-fire',
            'download': '/download-results/<filename>'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not_loaded"
    
    response = {
        'status': 'healthy',
        'model_status': model_status,
        'api_version': '2.0',
        'model_info': {
            'type': 'Custom Fire Detection Model',
            'classes': model.names if model and hasattr(model, 'names') else {}
        }
    }
    
    if model_loading_error:
        response['model_error'] = model_loading_error
    
    return jsonify(response)

@app.route('/detect-fire', methods=['POST'])
def detect_fire_endpoint():
    """
    Main fire detection endpoint - replicates your local yolo command functionality
    """
    try:
        if model is None:
            return jsonify({
                'error': 'YOLO model not loaded',
                'model_error': model_loading_error,
                'fire_detected': False
            }), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided. Please upload an image or video file.',
                'fire_detected': False
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'fire_detected': False
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Supported: {", ".join(ALLOWED_EXTENSIONS)}',
                'fire_detected': False
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        unique_filename = f"{timestamp}_{filename}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'_{filename}') as temp_file:
            file.save(temp_file.name)
            temp_filepath = temp_file.name
        
        logger.info(f"📁 File saved temporarily: {temp_filepath}")
        
        # Determine if it's a video
        is_video = filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mkv'))
        
        # Run fire detection with your trained model
        logger.info("🔥 Running fire detection with your custom model...")
        detection_result = detect_fire_with_yolo(temp_filepath, is_video)
        
        # Add file info to result
        detection_result.update({
            'filename': filename,
            'file_size': os.path.getsize(temp_filepath),
            'timestamp': timestamp,
            'file_type': 'video' if is_video else 'image'
        })
        
        # Clean up temporary file
        try:
            os.unlink(temp_filepath)
        except:
            pass
        
        logger.info(f"✅ Detection completed: Fire={detection_result['fire_detected']}, Confidence={detection_result['confidence']}, Detections={detection_result['total_detections']}")
        
        return jsonify(detection_result)
        
    except Exception as e:
        error_msg = f'Detection failed: {str(e)}'
        logger.error(f"❌ Error: {error_msg}")
        
        return jsonify({
            'error': error_msg,
            'fire_detected': False,
            'confidence': 0,
            'detections': []
        }), 500

@app.route('/download-results/<filename>', methods=['GET'])
def download_results(filename):
    """Download detection results zip file"""
    try:
        zip_path = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(zip_path):
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f'fire_detection_results_{int(time.time())}.zip',
                mimetype='application/zip'
            )
        else:
            return jsonify({'error': 'Results file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        'message': 'Fire Detection API is working!',
        'model_loaded': model is not None,
        'model_classes': model.names if model and hasattr(model, 'names') else {},
        'endpoints': ['/', '/health', '/detect-fire', '/test']
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/health', '/detect-fire', '/download-results/<filename>', '/test']
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Fire Detection API with your custom trained model...")
    
    # Load your custom model
    model_loaded = load_yolo_model()
    
    if model_loaded:
        logger.info("✅ Your custom fire detection model loaded successfully!")
        logger.info(f"🔥 Model classes: {model.names if model and hasattr(model, 'names') else 'Unknown'}")
    else:
        logger.error("❌ Failed to load your custom model, but API will still start")
    
    logger.info("📡 API accessible from external networks")
    logger.info("🌐 Health check: https://ob-d-ren.onrender.com/health")
    logger.info("🔥 Detection endpoint: https://ob-d-ren.onrender.com/detect-fire")
    
    # Get port from environment (Render sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Never use debug=True in production
        use_reloader=False,
        threaded=True
    )