import os
import sys
import logging
from flask import Flask, request, jsonify

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
        
        # Import and load model
        from ultralytics import YOLO
        
        # Try to load your custom model
        if os.path.exists('./best.pt'):
            logger.info("Found best.pt, loading custom model...")
            model = YOLO('./best.pt')
            logger.info("✅ Custom model loaded successfully!")
        elif os.path.exists('best.pt'):
            logger.info("Found best.pt, loading custom model...")
            model = YOLO('best.pt')
            logger.info("✅ Custom model loaded successfully!")
        else:
            logger.info("Custom model not found, loading default YOLOv8n...")
            model = YOLO('yolov8n.pt')
            logger.info("✅ Default model loaded successfully!")
        
        return True
        
    except Exception as e:
        model_error = str(e)
        logger.error(f"❌ Model loading failed: {e}")
        return False

@app.route('/')
def home():
    return jsonify({
        'message': 'Fire Detection API',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': ['/health', '/detect-fire']
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_status': 'loaded' if model else 'failed',
        'model_error': model_error,
        'python_version': sys.version
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
        
        # Save file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        try:
            # Run detection
            logger.info(f"Running detection on {temp_path}")
            results = model(temp_path, conf=0.5)
            
            # Process results
            detections = []
            fire_detected = False
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Check if it's fire-related
                        fire_keywords = ['fire', 'flame', 'smoke']
                        is_fire = any(keyword in class_name.lower() for keyword in fire_keywords)
                        
                        if is_fire:
                            fire_detected = True
                        
                        bbox = box.xyxy[0].tolist()
                        detections.append({
                            'class': class_name,
                            'confidence': round(confidence, 2),
                            'is_fire': is_fire,
                            'bbox': [round(x, 2) for x in bbox]
                        })
            
            response = {
                'fire_detected': fire_detected,
                'total_detections': len(detections),
                'detections': detections,
                'filename': file.filename,
                'success': True
            }
            
            logger.info(f"✅ Detection complete: Fire={fire_detected}, Detections={len(detections)}")
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
            'error': str(e),
            'fire_detected': False
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Fire Detection API...")
    
    # Load model
    if load_model():
        logger.info("✅ Model loaded successfully!")
    else:
        logger.error("❌ Model loading failed")
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)