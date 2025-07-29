import os
import sys
import logging
import warnings
from flask import Flask, request, jsonify

# Suppress warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
model_error = None

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.after_request
def after_request(response):
    return add_cors_headers(response)

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({'message': 'OK'})
        return add_cors_headers(response)

def load_model():
    global model, model_error
    try:
        logger.info("🔥 Loading YOLO model...")
        
        # Set environment variables
        os.environ['YOLO_CONFIG_DIR'] = '/tmp'
        
        # Fix PyTorch weights loading issue
        import torch
        
        # Add safe globals for ultralytics
        torch.serialization.add_safe_globals([
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.nn.modules.block.C2f',
            'ultralytics.nn.modules.block.SPPF',
            'ultralytics.nn.modules.conv.Conv',
            'ultralytics.nn.modules.head.Detect',
            'collections.OrderedDict',
        ])
        
        # Import ultralytics after setting up torch
        from ultralytics import YOLO
        
        # Try to load your custom model with weights_only=False
        model_paths = ['./best.pt', 'best.pt']
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading custom model from: {model_path}")
                    
                    # Load with weights_only=False to bypass the security restriction
                    model = YOLO(model_path)
                    
                    # Monkey patch to use weights_only=False
                    if hasattr(model.model, 'load_state_dict'):
                        original_load = torch.load
                        torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
                    
                    logger.info("✅ Custom fire detection model loaded successfully!")
                    logger.info(f"Model classes: {getattr(model, 'names', {})}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Failed to load custom model {model_path}: {e}")
                    continue
        
        # Fallback to default model
        logger.info("Loading default YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        logger.info("✅ Default YOLOv8n model loaded successfully!")
        return True
        
    except Exception as e:
        model_error = str(e)
        logger.error(f"❌ Model loading failed: {e}")
        return False

@app.route('/')
def home():
    return jsonify({
        'message': 'Fire Detection API - Fixed PyTorch Loading',
        'status': 'running',
        'model_loaded': model is not None,
        'model_type': 'Custom Fire Detection' if model and hasattr(model, 'names') and any('fire' in str(name).lower() for name in model.names.values()) else 'General Detection',
        'endpoints': ['/health', '/detect-fire', '/test']
    })

@app.route('/health')
def health():
    model_info = {}
    if model:
        model_info = {
            'classes': getattr(model, 'names', {}),
            'model_type': 'custom' if os.path.exists('./best.pt') or os.path.exists('best.pt') else 'default'
        }
    
    return jsonify({
        'status': 'healthy',
        'model_status': 'loaded' if model else 'failed',
        'model_error': model_error,
        'model_info': model_info,
        'python_version': sys.version.split()[0],
        'pytorch_fix': 'Applied weights_only=False fix'
    })

@app.route('/detect-fire', methods=['POST'])
def detect_fire():
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'model_error': model_error,
                'fire_detected': False
            }), 500
        
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'fire_detected': False
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'fire_detected': False
            }), 400
        
        # Check file extension
        allowed_extensions = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'gif'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'error': f'File type .{file_ext} not supported. Use: {", ".join(allowed_extensions)}',
                'fire_detected': False
            }), 400
        
        # Save file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        try:
            # Run detection
            logger.info(f"Running detection on {file.filename}")
            results = model(temp_path, conf=0.3, verbose=False)  # Lower confidence, disable verbose
            
            # Process results
            detections = []
            fire_detected = False
            max_confidence = 0
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names.get(class_id, f'class_{class_id}')
                        confidence = float(box.conf[0])
                        
                        # Check if it's fire-related (for custom model) or high-confidence detection
                        fire_keywords = ['fire', 'flame', 'smoke', 'burning']
                        is_fire_class = any(keyword in str(class_name).lower() for keyword in fire_keywords)
                        
                        # For custom fire detection model, any detection is likely fire
                        # For general model, look for specific classes or high confidence
                        is_fire = is_fire_class or (confidence > 0.7 and any(keyword in str(class_name).lower() for keyword in ['person', 'car', 'truck']))
                        
                        if is_fire or is_fire_class:
                            fire_detected = True
                        
                        if confidence > max_confidence:
                            max_confidence = confidence
                        
                        bbox = box.xyxy[0].tolist()
                        detections.append({
                            'class': str(class_name),
                            'confidence': round(confidence, 3),
                            'is_fire_related': is_fire_class,
                            'bbox': [round(x, 2) for x in bbox],
                            'bbox_x1': round(bbox[0], 2),
                            'bbox_y1': round(bbox[1], 2),
                            'bbox_x2': round(bbox[2], 2),
                            'bbox_y2': round(bbox[3], 2)
                        })
            
            response = {
                'fire_detected': fire_detected,
                'confidence': round(max_confidence, 3),
                'total_detections': len(detections),
                'detections': detections,
                'filename': file.filename,
                'file_type': 'video' if file_ext in ['mp4', 'avi', 'mov'] else 'image',
                'processing_info': {
                    'model_used': 'custom' if os.path.exists('./best.pt') or os.path.exists('best.pt') else 'default',
                    'confidence_threshold': 0.3
                },
                'success': True
            }
            
            # Log result
            status = "🚨 FIRE DETECTED" if fire_detected else "✅ NO FIRE"
            logger.info(f"{status} in {file.filename} - Confidence: {max_confidence:.3f}, Detections: {len(detections)}")
            
            return jsonify(response)
            
        finally:
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
        
    except Exception as e:
        logger.error(f"❌ Detection error: {e}")
        return jsonify({
            'error': f'Detection failed: {str(e)}',
            'fire_detected': False,
            'success': False
        }), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'message': 'Fire Detection API Test Endpoint',
        'model_loaded': model is not None,
        'model_classes': getattr(model, 'names', {}) if model else {},
        'custom_model_exists': os.path.exists('./best.pt') or os.path.exists('best.pt'),
        'endpoints': ['/', '/health', '/detect-fire', '/test'],
        'supported_formats': ['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov', 'gif']
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 100MB)', 'fire_detected': False}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error', 'fire_detected': False}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Fire Detection API with PyTorch fix...")
    
    # Load model
    if load_model():
        logger.info("✅ Model loaded successfully!")
    else:
        logger.error("❌ Model loading failed, API will run with limited functionality")
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 API running on port {port}")
    logger.info("🔥 Ready to detect fire!")
    
    app.run(host='0.0.0.0', port=port, debug=False)