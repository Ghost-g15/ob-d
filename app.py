from flask import Flask, request, jsonify, after_this_request
import os
import time
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import json

app = Flask(__name__)

# Manual CORS setup
def add_cors_headers(response):
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
        response = add_cors_headers(response)
        return response

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png', 'gif'}

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Load YOLO model globally
print("🔥 Loading YOLO fire detection model...")
try:
    model = YOLO(os.getenv("MODEL_PATH", "./best.pt"))
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_fire_with_yolo(file_path):
    """
    Detect fire using your trained YOLO model
    """
    if model is None:
        raise Exception("YOLO model not loaded")
    
    start_time = time.time()
    
    # Run detection
    results = model(file_path, save=True, project=app.config['OUTPUT_FOLDER'], name='detected')
    
    processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
    
    # Parse results
    detections = []
    fire_detected = False
    max_confidence = 0
    
    if results and len(results) > 0:
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            fire_detected = True
            
            for box in result.boxes:
                # Get bounding box coordinates
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get class name (assuming your model has class names)
                class_name = model.names[class_id] if hasattr(model, 'names') else f'class_{class_id}'
                
                detections.append({
                    'class': class_name,
                    'confidence': round(confidence, 2),
                    'bbox': [round(coord, 2) for coord in bbox],  # [x1, y1, x2, y2]
                    'bbox_x1': round(bbox[0], 2),
                    'bbox_y1': round(bbox[1], 2),
                    'bbox_x2': round(bbox[2], 2),
                    'bbox_y2': round(bbox[3], 2)
                })
                
                if confidence > max_confidence:
                    max_confidence = confidence
    
    # Get output path
    output_path = None
    if results and hasattr(results[0], 'save_dir'):
        output_path = str(results[0].save_dir)
    
    return {
        'fire_detected': fire_detected,
        'confidence': round(max_confidence, 2),
        'total_detections': len(detections),
        'detections': detections,
        'processing_time': processing_time,
        'output_path': output_path,
        'model_info': {
            'model_path': "best.pt",
            'classes': model.names if model and hasattr(model, 'names') else {}
        }
    }

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        'status': 'ok',
        'message': 'Flask API is running',
        'model_status': model_status,
        'endpoint': '/detect-fire'
    })

# Main fire detection endpoint
@app.route('/detect-fire', methods=['POST'])
def detect_fire_endpoint():
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided', 'fire_detected': False}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'fire_detected': False}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed', 'fire_detected': False}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        print(f"📁 File saved: {filepath}")
        
        # Run fire detection with your YOLO model
        print("🔥 Running fire detection...")
        detection_result = detect_fire_with_yolo(filepath)
        
        # Add file info to result
        detection_result.update({
            'filename': filename,
            'file_size': os.path.getsize(filepath),
            'file_path': filepath,
            'timestamp': timestamp
        })
        
        print(f"✅ Detection completed: {detection_result['fire_detected']} (confidence: {detection_result['confidence']})")
        
        # Optional: Clean up uploaded file
        # os.remove(filepath)
        
        return jsonify(detection_result)
        
    except Exception as e:
        error_msg = f'Detection failed: {str(e)}'
        print(f"❌ Error: {error_msg}")
        
        return jsonify({
            'error': error_msg,
            'fire_detected': False,
            'confidence': 0,
            'detections': []
        }), 500

# Test endpoint
@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        'message': 'Flask API is working!',
        'model_loaded': model is not None,
        'endpoints': ['/health', '/detect-fire', '/test']
    })

if __name__ == '__main__':
    print("🔥 Starting Fire Detection Flask API...")
    print("📡 API will be accessible from external networks")
    print(f"🤖 YOLO Model Status: {'Loaded' if model else 'Not Loaded'}")
    print("🌐 Health check: http://YOUR_IP:5000/health")
    print("🔥 Detection endpoint: http://YOUR_IP:5000/detect-fire")
    
    # Make accessible from outside localhost
    app.run(host='0.0.0.0', port=5000, debug=True)