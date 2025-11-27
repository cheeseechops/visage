# Initialize ONNX Runtime FIRST before ANYTHING else
# Add CUDA 12.2 DLL directory so ONNX Runtime can find the DLLs
import os
import shutil
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
cuda_bin = os.path.join(cuda_path, "bin")
required_dll = "cublasLt64_12.dll"

if os.path.exists(cuda_bin):
    dll_path = os.path.join(cuda_bin, required_dll)
    
    # Method 1: Use os.add_dll_directory() for Windows DLL loading (Python 3.8+)
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(cuda_bin)
    
    # Method 2: Also add to PATH for compatibility
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    
    # Method 3: Copy DLL to ONNX Runtime directory as fallback
    # This ensures Windows can find it when loading onnxruntime_providers_cuda.dll
    try:
        import site
        site_packages = site.getsitepackages()
        if site_packages:
            onnx_capi_dir = os.path.join(site_packages[0], "onnxruntime", "capi")
            if os.path.exists(onnx_capi_dir) and os.path.exists(dll_path):
                target_dll = os.path.join(onnx_capi_dir, required_dll)
                if not os.path.exists(target_dll):
                    shutil.copy2(dll_path, target_dll)
                    print(f"✓ Copied {required_dll} to ONNX Runtime directory")
    except Exception:
        pass  # If copy fails, rely on PATH/add_dll_directory
    
    os.environ["CUDA_PATH"] = cuda_path
    print(f"✓ Configured CUDA 12.2 DLL loading: {cuda_bin}")

import onnxruntime as ort
ort.set_default_logger_severity(3)  # Suppress warnings

from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session, send_from_directory, Response
import random
import uuid
import json
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
import io
import base64
from models import db, Person, Image, image_people
from face_recognition_module import register_face_recognition_commands, FaceRecognitionDB
from yolo_detector import get_yolo_detector
from functools import wraps
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import platform
import psutil
import shutil
import imagehash
import tempfile
import cv2
import threading
from sqlalchemy import or_, and_, not_, text, nullslast, nullsfirst, func
from sqlalchemy.orm import joinedload
import time
import multiprocessing

# Import performance logging
from utils.performance_logger import time_operation, time_function, get_performance_summary, print_performance_summary, save_performance_summary

from utils.phash_db import build_phash_db, get_all_phashes, find_similar_phash, compute_phash

# Import Ollama chat functionality (disabled - not needed)
OLLAMA_AVAILABLE = False
OllamaChat = None

# Import PyChromecast for Chromecast support
try:
    import pychromecast
    from pychromecast.controllers.media import MediaController
    CHROMECAST_AVAILABLE = True
except ImportError:
    CHROMECAST_AVAILABLE = False
    print("Warning: PyChromecast not available. Chromecast functionality will be disabled.")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# --- Config ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///visage.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'zip'}
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 16MB max file size

import os
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

db.init_app(app)

# --- Face recognition model warmup ---
from face_recognition_module import get_face_db
face_db = get_face_db()

with app.app_context():
    try:
        from models import Image  # Import models only after app context is set up
        real_image = None
        img = Image.query.filter(Image.original_path.isnot(None)).order_by(Image.id.desc()).first()
        print(f"[Startup] img from original_path query: {img} (type: {type(img)})")
        if img is not None and hasattr(img, 'original_path') and img.original_path and os.path.exists(img.original_path.replace('/static/', 'static/')):
            real_image = img.original_path.replace('/static/', 'static/')
        if real_image:
            print(f"[Startup] Warming up face recognition model with real image: {real_image}")
            face_db.detect_faces(real_image)
            print("[Startup] FaceRecognitionDB ran detection on real image to warm up model.")
        else:
            print("[Startup] No real image found for model warmup. Skipping warmup.")
    except Exception as e:
        print(f"[Startup] Error during real image face detection warmup: {e}")

# Global singleton instance of FaceRecognitionDB
_face_db_instance = None

def get_face_db():
    """Get or create the singleton FaceRecognitionDB instance"""
    global _face_db_instance
    if _face_db_instance is None:
        print("[Startup] Initializing FaceRecognitionDB (this may take a while)...")
        _face_db_instance = FaceRecognitionDB(app)
        try:
            _face_db_instance.initialize_insightface()
            print("[Startup] FaceRecognitionDB initialized and model loaded.")
        except Exception as e:
            print(f"[Startup] Error initializing FaceRecognitionDB: {e}")
    return _face_db_instance

def reload_face_db():
    """Reload the global face database instance to get latest changes"""
    global _face_db_instance
    if _face_db_instance is not None:
        return _face_db_instance.reload_database()
    return False

def autorotate_image_pil(pil_img):
    import numpy as np
    # Use the local get_face_db function instead of importing from face_recognition_module
    face_db = get_face_db()
    # Convert PIL to OpenCV
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    faces = face_db.app_insightface.get(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    if not faces:
        return pil_img  # No face detected
    # Use the largest face
    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    if hasattr(largest_face, 'pose') and len(largest_face.pose) > 0:
        roll = largest_face.pose[2]  # pose = [yaw, pitch, roll]
        if abs(roll) > 2:  # Only rotate if significant
            return pil_img.rotate(-roll, expand=True)
    return pil_img

# Configuration for file uploads
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

ADMIN_PASSWORD = 'admin'  # Set your admin password here

# Bug report file path
BUG_REPORT_FILE = 'bug_reports.txt'

# Face matching state persistence
FACE_MATCHING_STATE_FILE = 'face_matching_state.json'

def get_system_info():
    """Get system information for bug reports"""
    try:
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            'disk_usage': f"{psutil.disk_usage('/').percent:.1f}%",
            'app_version': '1.0.0'  # You can update this as needed
        }
    except Exception as e:
        return {'error': f'Could not get system info: {str(e)}'}

def save_bug_report(report_data):
    """Save bug report to text file with automatic context"""
    try:
        # Get system information
        system_info = get_system_info()
        
        # Get current database stats
        with app.app_context():
            total_people = Person.query.count()
            total_images = Image.query.count()
            confirmed_people = Person.query.filter_by(is_confirmed=True).count()
        
        # Format the report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_text = f"""
{'='*80}
BUG REPORT - {timestamp}
{'='*80}

DESCRIPTION:
{report_data.get('description', 'No description provided')}

SEVERITY: {report_data.get('severity', 'Unknown').upper()}

USER CONTEXT:
- Page: {report_data.get('page', 'Unknown')}
- URL: {report_data.get('url', 'Unknown')}
- User Agent: {report_data.get('user_agent', 'Unknown')}
- Timestamp: {report_data.get('timestamp', 'Unknown')}

SYSTEM INFORMATION:
- Platform: {system_info.get('platform', 'Unknown')}
- Python Version: {system_info.get('python_version', 'Unknown')}
- CPU Count: {system_info.get('cpu_count', 'Unknown')}
- Total Memory: {system_info.get('memory_total', 'Unknown')}
- Disk Usage: {system_info.get('disk_usage', 'Unknown')}
- App Version: {system_info.get('app_version', 'Unknown')}

APPLICATION STATE:
- Total People: {total_people}
- Total Images: {total_images}
- Confirmed People: {confirmed_people}
- Unidentified People: {total_people - confirmed_people}

{'='*80}

"""
        
        # Append to bug report file
        with open(BUG_REPORT_FILE, 'a', encoding='utf-8') as f:
            f.write(report_text)
        
        return True
    except Exception as e:
        print(f"Error saving bug report: {e}")
        return False

def save_face_matching_state(state):
    """Save face matching state for resuming"""
    try:
        with open(FACE_MATCHING_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving face matching state: {e}")
        return False

def load_face_matching_state():
    """Load face matching state for resuming"""
    try:
        if os.path.exists(FACE_MATCHING_STATE_FILE):
            with open(FACE_MATCHING_STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading face matching state: {e}")
    return {
        'last_processed_group': 0,
        'total_groups': 0,
        'processing_complete': False,
        'last_update': None
    }

def clear_face_matching_data():
    """Clear all face matching data"""
    try:
        # Remove state file
        if os.path.exists(FACE_MATCHING_STATE_FILE):
            os.remove(FACE_MATCHING_STATE_FILE)
        
        # Remove face crops database
        face_db_path = 'face_crops/face_crops.db'
        if os.path.exists(face_db_path):
            os.remove(face_db_path)
        
        # Remove dHash cache
        dhash_cache_path = 'face_crops/dhash_cache.pkl'
        if os.path.exists(dhash_cache_path):
            os.remove(dhash_cache_path)
        
        # Remove face crops directories (optional - keep the structure)
        # import shutil
        # if os.path.exists('face_crops/fitpexport'):
        #     shutil.rmtree('face_crops/fitpexport')
        # if os.path.exists('face_crops/visage'):
        #     shutil.rmtree('face_crops/visage')
        
        return True
    except Exception as e:
        print(f"Error clearing face matching data: {e}")
        return False

# Decorator to require login
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_zip_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'zip'

def process_uploaded_image(file):
    """Process uploaded image and create cropped person thumbnails"""
    try:
        # Read the image
        img = PILImage.open(file.stream)
        # Convert .webp to .jpg if needed
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == '.webp':
            img = img.convert('RGB')
            filename = secure_filename(os.path.splitext(file.filename)[0] + '.jpg')
        else:
            filename = secure_filename(file.filename)
        base_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], base_name)
        # Save original image (always as jpg if converted)
        if ext == '.webp':
            img.save(file_path, 'JPEG')
        else:
            img.save(file_path)
        # For demo purposes, create a "cropped person" by resizing the image
        cropped_img = img.copy()
        cropped_img.thumbnail((300, 400))  # Resize to thumbnail size
        # Save cropped version
        cropped_filename = f"cropped_{base_name}"
        cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
        cropped_img.save(cropped_path)
        return {
            'original_path': f'/static/uploads/{base_name}',
            'cropped_path': f'/static/uploads/{cropped_filename}',
            'filename': filename
        }
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Static file serving routes
@app.route('/static/face_crops/<path:filename>')
def serve_face_crops(filename):
    """Serve face crop images"""
    return send_from_directory('face_crops', filename)

@app.route('/static/fitpexport/<path:filename>')
def serve_fitpexport(filename):
    """Serve fitpexport images"""
    return send_from_directory('fitpexport', filename)

@app.route('/static/uploads/<path:filename>')
def serve_uploads(filename):
    """Serve uploaded images"""
    return send_from_directory('static/uploads', filename)

@app.route('/api/image/<int:image_id>/diagnostic')
def diagnose_image(image_id):
    """Diagnostic endpoint to check image issues"""
    try:
        image = Image.query.get(image_id)
        if not image:
            return jsonify({'error': 'Image not found in database', 'image_id': image_id}), 404
        
        # Determine source file path
        if image.original_path and image.original_path.startswith('/static/'):
            file_path = image.original_path[1:]  # Remove leading /
        else:
            file_path = f'static/uploads/{image.filename}'
        
        file_exists = os.path.exists(file_path)
        
        diagnostic = {
            'image_id': image_id,
            'database_record': {
                'id': image.id,
                'filename': image.filename,
                'original_path': image.original_path,
                'width': image.width,
                'height': image.height,
                'is_cropped': image.is_cropped,
                'created_at': str(image.created_at) if image.created_at else None,
            },
            'file_path': file_path,
            'file_exists': file_exists,
            'file_size': os.path.getsize(file_path) if file_exists else None,
        }
        
        if file_exists:
            try:
                # Try to open the image
                with PILImage.open(file_path) as img:
                    diagnostic['image_info'] = {
                        'format': img.format,
                        'mode': img.mode,
                        'size': img.size,
                        'can_open': True
                    }
            except Exception as e:
                diagnostic['image_info'] = {
                    'can_open': False,
                    'error': str(e)
                }
        else:
            # Check alternative paths
            alt_paths = [
                f'static/uploads/{image.filename}',
                image.original_path[1:] if image.original_path and image.original_path.startswith('/static/') else None,
                image.original_path if image.original_path else None,
            ]
            diagnostic['alternative_paths_checked'] = []
            for alt_path in alt_paths:
                if alt_path:
                    exists = os.path.exists(alt_path)
                    diagnostic['alternative_paths_checked'].append({
                        'path': alt_path,
                        'exists': exists
                    })
        
        return jsonify(diagnostic)
    except Exception as e:
        return jsonify({'error': str(e), 'image_id': image_id}), 500

@app.route('/api/image/<int:image_id>/optimized')
def serve_optimized_image(image_id):
    """Serve optimized image for slideshow - resized and compressed for faster network transfer"""
    from io import BytesIO
    from flask import send_file
    
    try:
        image = Image.query.get(image_id)
        if not image:
            return jsonify({'error': 'Image not found'}), 404
        
        # Determine source file path - try multiple possibilities
        file_path = None
        possible_paths = []
        
        if image.original_path:
            if image.original_path.startswith('/static/'):
                possible_paths.append(image.original_path[1:])  # Remove leading /
            elif image.original_path.startswith('static/'):
                possible_paths.append(image.original_path)
            else:
                possible_paths.append(image.original_path)
        
        # Also try default uploads path
        possible_paths.append(f'static/uploads/{image.filename}')
        
        # Try each path until we find one that exists
        for path in possible_paths:
            if path and os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            # Log diagnostic info for debugging
            print(f"[ERROR] Image {image_id} file not found. Tried paths: {possible_paths}")
            print(f"[ERROR] Image record: filename={image.filename}, original_path={image.original_path}")
            return jsonify({
                'error': 'File not found',
                'image_id': image_id,
                'tried_paths': possible_paths,
                'database_record': {
                    'filename': image.filename,
                    'original_path': image.original_path
                }
            }), 404
        
        # Get max dimensions based on screen size
        # Optimized for iPad (2019: 2160x1620) and mobile devices - smaller defaults for faster loading
        max_width = int(request.args.get('max_width', 1620))  # Default to iPad height (portrait)
        max_height = int(request.args.get('max_height', 1620))
        quality = int(request.args.get('quality', 80))  # Lower default quality for mobile/tablet (80 vs 85)
        
        # Open and resize image
        try:
            img = PILImage.open(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to open image {image_id} from {file_path}: {e}")
            return jsonify({
                'error': f'Failed to open image file: {str(e)}',
                'image_id': image_id,
                'file_path': file_path
            }), 500
        
        try:
            with img:
                # Convert to RGB if necessary (for JPEG)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparency
                    background = PILImage.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate new size maintaining aspect ratio
                original_width, original_height = img.size
                ratio = min(max_width / original_width, max_height / original_height)
                
                # Only resize if image is larger than target
                if ratio < 1:
                    new_width = int(original_width * ratio)
                    new_height = int(original_height * ratio)
                    img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                
                # Save to memory buffer
                output = BytesIO()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                output.seek(0)
                
                return send_file(
                    output,
                    mimetype='image/jpeg',
                    as_attachment=False,
                    download_name=f'image_{image_id}_optimized.jpg',
                    max_age=31536000  # Cache for 1 year
                )
        except Exception as e:
            print(f"[ERROR] Failed to process image {image_id}: {e}")
            return jsonify({
                'error': f'Failed to process image: {str(e)}',
                'image_id': image_id,
                'file_path': file_path
            }), 500
    except Exception as e:
        print(f"[ERROR] Error serving optimized image {image_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'image_id': image_id}), 500

@app.route('/')
def home():
    """Home page - All People grid"""
    sort_by = request.args.get('sort', 'id')  # Default to id (newest first)
    if sort_by == 'name':
        people = Person.query.filter(Person.images.any()).order_by(Person.name).all()
    elif sort_by == 'image_count':
        people = Person.query.filter(Person.images.any()).all()
        people = sorted(people, key=lambda p: len(p.images), reverse=True)
    elif sort_by == 'id':
        people = Person.query.filter(Person.images.any()).all()
        people = sorted(people, key=lambda p: max([img.id for img in p.images]) if p.images else 0, reverse=True)
    else:
        # Default to id-desc
        people = Person.query.filter(Person.images.any()).all()
        people = sorted(people, key=lambda p: max([img.id for img in p.images]) if p.images else 0, reverse=True)
        sort_by = 'id'
    return render_template('home.html', people=[p.to_dict() for p in people], current_sort=sort_by)

@app.route('/import', methods=['GET', 'POST'])
def import_workflow():
    """Combined Import, Crop, and Name workflow page with resume and stepper."""
    import glob
    import json
    import zipfile
    import tempfile
    import shutil
    from models import Image, Person
    from flask import request, flash, redirect, url_for
    import os
    
    if request.method == 'POST':
        imported_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'imported')
        os.makedirs(imported_folder, exist_ok=True)
        
        uploaded_count = 0
        errors = []
        
        # Handle individual images
        if 'images' in request.files:
            files = request.files.getlist('images')
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    try:
                        file.seek(0, 2)
                        file_size = file.tell()
                        file.seek(0)
                        if file_size > MAX_FILE_SIZE:
                            errors.append(f"{file.filename} is too large (max 16MB)")
                            continue
                        
                        ext = os.path.splitext(file.filename)[1].lower()
                        if ext == '.webp':
                            img = PILImage.open(file.stream).convert('RGB')
                            filename = secure_filename(os.path.splitext(file.filename)[0] + '.jpg')
                            base_name = f"{uuid.uuid4().hex}_{filename}"
                            file_path = os.path.join(imported_folder, base_name)
                            img.save(file_path, 'JPEG')
                        else:
                            filename = secure_filename(file.filename)
                            base_name = f"{uuid.uuid4().hex}_{filename}"
                            file_path = os.path.join(imported_folder, base_name)
                            file.save(file_path)
                        
                        # Create database entry
                        tags = ['uncropped']
                        new_image = Image(
                            original_path=f'/static/uploads/imported/{base_name}',
                            filename=filename,
                            is_cropped=False,
                            is_favorite=False,
                            tags=json.dumps(tags),
                            recognition_suggestions=json.dumps([])
                        )
                        
                        # Set width and height
                        try:
                            with PILImage.open(file_path) as im:
                                new_image.width = im.width
                                new_image.height = im.height
                        except Exception as e:
                            print(f"Error setting image dimensions: {e}")
                        
                        db.session.add(new_image)
                        db.session.commit()
                        uploaded_count += 1
                        
                    except Exception as e:
                        errors.append(f"Error processing {file.filename}: {str(e)}")
                elif file and file.filename:
                    errors.append(f"{file.filename} is not a valid image file")
        
        # Handle folder uploads
        if 'folder' in request.files:
            files = request.files.getlist('folder')
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    try:
                        file.seek(0, 2)
                        file_size = file.tell()
                        file.seek(0)
                        if file_size > MAX_FILE_SIZE:
                            errors.append(f"{file.filename} is too large (max 16MB)")
                            continue
                        
                        ext = os.path.splitext(file.filename)[1].lower()
                        if ext == '.webp':
                            img = PILImage.open(file.stream).convert('RGB')
                            filename = secure_filename(os.path.splitext(file.filename)[0] + '.jpg')
                            base_name = f"{uuid.uuid4().hex}_{filename}"
                            file_path = os.path.join(imported_folder, base_name)
                            img.save(file_path, 'JPEG')
                        else:
                            filename = secure_filename(file.filename)
                            base_name = f"{uuid.uuid4().hex}_{filename}"
                            file_path = os.path.join(imported_folder, base_name)
                            file.save(file_path)
                        
                        # Create database entry
                        tags = ['uncropped']
                        new_image = Image(
                            original_path=f'/static/uploads/imported/{base_name}',
                            filename=filename,
                            is_cropped=False,
                            is_favorite=False,
                            tags=json.dumps(tags),
                            recognition_suggestions=json.dumps([])
                        )
                        
                        # Set width and height
                        try:
                            with PILImage.open(file_path) as im:
                                new_image.width = im.width
                                new_image.height = im.height
                        except Exception as e:
                            print(f"Error setting image dimensions: {e}")
                        
                        db.session.add(new_image)
                        db.session.commit()
                        uploaded_count += 1
                        
                    except Exception as e:
                        errors.append(f"Error processing {file.filename}: {str(e)}")
        
        # Handle zip files
        if 'zip_file' in request.files:
            zip_files = request.files.getlist('zip_file')
            for zip_file in zip_files:
                if zip_file and zip_file.filename and is_zip_file(zip_file.filename):
                    try:
                        # Check file size
                        zip_file.seek(0, 2)
                        file_size = zip_file.tell()
                        zip_file.seek(0)
                        
                        if file_size > MAX_FILE_SIZE:
                            errors.append(f"{zip_file.filename} is too large (max 16MB)")
                            continue
                        
                        # Create a temporary directory to extract files
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Save zip file temporarily
                            temp_zip_path = os.path.join(temp_dir, 'upload.zip')
                            zip_file.save(temp_zip_path)
                            
                            # Extract zip file
                            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                                # Check for zip bomb (too many files)
                                file_list = zip_ref.namelist()
                                if len(file_list) > 1000:  # Limit to 1000 files
                                    errors.append(f"{zip_file.filename} contains too many files (max 1000)")
                                    continue
                                
                                # Extract files
                                zip_ref.extractall(temp_dir)
                            
                            # Process extracted files
                            for root, dirs, files in os.walk(temp_dir):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    file_ext = os.path.splitext(file)[1].lower()
                                    
                                    # Check if it's an image file
                                    if file_ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}:
                                        try:
                                            # Generate unique filename
                                            filename = secure_filename(file)
                                            base_name = f"{uuid.uuid4().hex}_{filename}"
                                            target_path = os.path.join(imported_folder, base_name)
                                            
                                            # Process the image
                                            if file_ext == '.webp':
                                                # Convert webp to jpg
                                                img = PILImage.open(file_path).convert('RGB')
                                                filename = os.path.splitext(filename)[0] + '.jpg'
                                                base_name = f"{uuid.uuid4().hex}_{filename}"
                                                target_path = os.path.join(imported_folder, base_name)
                                                img.save(target_path, 'JPEG')
                                            else:
                                                # Copy file as is
                                                shutil.copy2(file_path, target_path)
                                            
                                            # Create database entry
                                            tags = ['uncropped']
                                            new_image = Image(
                                                original_path=f'/static/uploads/imported/{base_name}',
                                                filename=filename,
                                                is_cropped=False,
                                                is_favorite=False,
                                                tags=json.dumps(tags),
                                                recognition_suggestions=json.dumps([])
                                            )
                                            
                                            # Set width and height
                                            try:
                                                with PILImage.open(target_path) as im:
                                                    new_image.width = im.width
                                                    new_image.height = im.height
                                            except Exception as e:
                                                print(f"Error setting image dimensions: {e}")
                                            
                                            db.session.add(new_image)
                                            db.session.commit()
                                            uploaded_count += 1
                                            
                                        except Exception as e:
                                            errors.append(f"Error processing {file} from {zip_file.filename}: {str(e)}")
                    
                    except zipfile.BadZipFile:
                        errors.append(f"{zip_file.filename} is not a valid zip file")
                    except Exception as e:
                        errors.append(f"Error processing zip file {zip_file.filename}: {str(e)}")
        
        # Handle mixed files (when files are uploaded through the unified interface)
        if 'files' in request.files:
            files = request.files.getlist('files')
            for file in files:
                if file and file.filename:
                    # Check if it's a zip file
                    if is_zip_file(file.filename):
                        try:
                            # Check file size
                            file.seek(0, 2)
                            file_size = file.tell()
                            file.seek(0)
                            
                            if file_size > MAX_FILE_SIZE:
                                errors.append(f"{file.filename} is too large (max 16MB)")
                                continue
                            
                            # Create a temporary directory to extract files
                            with tempfile.TemporaryDirectory() as temp_dir:
                                # Save zip file temporarily
                                temp_zip_path = os.path.join(temp_dir, 'upload.zip')
                                file.save(temp_zip_path)
                                
                                # Extract zip file
                                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                                    # Check for zip bomb (too many files)
                                    file_list = zip_ref.namelist()
                                    if len(file_list) > 1000:  # Limit to 1000 files
                                        errors.append(f"{file.filename} contains too many files (max 1000)")
                                        continue
                                    
                                    # Extract files
                                    zip_ref.extractall(temp_dir)
                                
                                # Process extracted files
                                for root, dirs, files_in_zip in os.walk(temp_dir):
                                    for file_in_zip in files_in_zip:
                                        file_path = os.path.join(root, file_in_zip)
                                        file_ext = os.path.splitext(file_in_zip)[1].lower()
                                        
                                        # Check if it's an image file
                                        if file_ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}:
                                            try:
                                                # Generate unique filename
                                                filename = secure_filename(file_in_zip)
                                                base_name = f"{uuid.uuid4().hex}_{filename}"
                                                target_path = os.path.join(imported_folder, base_name)
                                                
                                                # Process the image
                                                if file_ext == '.webp':
                                                    # Convert webp to jpg
                                                    img = PILImage.open(file_path).convert('RGB')
                                                    filename = os.path.splitext(filename)[0] + '.jpg'
                                                    base_name = f"{uuid.uuid4().hex}_{filename}"
                                                    target_path = os.path.join(imported_folder, base_name)
                                                    img.save(target_path, 'JPEG')
                                                else:
                                                    # Copy file as is
                                                    shutil.copy2(file_path, target_path)
                                                
                                                # Create database entry
                                                tags = ['uncropped']
                                                new_image = Image(
                                                    original_path=f'/static/uploads/imported/{base_name}',
                                                    filename=filename,
                                                    is_cropped=False,
                                                    is_favorite=False,
                                                    tags=json.dumps(tags),
                                                    recognition_suggestions=json.dumps([])
                                                )
                                                
                                                # Set width and height
                                                try:
                                                    with PILImage.open(target_path) as im:
                                                        new_image.width = im.width
                                                        new_image.height = im.height
                                                except Exception as e:
                                                    print(f"Error setting image dimensions: {e}")
                                                
                                                db.session.add(new_image)
                                                db.session.commit()
                                                uploaded_count += 1
                                                
                                            except Exception as e:
                                                errors.append(f"Error processing {file_in_zip} from {file.filename}: {str(e)}")
                        
                        except zipfile.BadZipFile:
                            errors.append(f"{file.filename} is not a valid zip file")
                        except Exception as e:
                            errors.append(f"Error processing zip file {file.filename}: {str(e)}")
                    
                    # Check if it's an image file
                    elif allowed_file(file.filename):
                        try:
                            file.seek(0, 2)
                            file_size = file.tell()
                            file.seek(0)
                            if file_size > MAX_FILE_SIZE:
                                errors.append(f"{file.filename} is too large (max 16MB)")
                                continue
                            
                            ext = os.path.splitext(file.filename)[1].lower()
                            if ext == '.webp':
                                img = PILImage.open(file.stream).convert('RGB')
                                filename = secure_filename(os.path.splitext(file.filename)[0] + '.jpg')
                                base_name = f"{uuid.uuid4().hex}_{filename}"
                                file_path = os.path.join(imported_folder, base_name)
                                img.save(file_path, 'JPEG')
                            else:
                                filename = secure_filename(file.filename)
                                base_name = f"{uuid.uuid4().hex}_{filename}"
                                file_path = os.path.join(imported_folder, base_name)
                                file.save(file_path)
                            
                            # Create database entry
                            tags = ['uncropped']
                            new_image = Image(
                                original_path=f'/static/uploads/imported/{base_name}',
                                filename=filename,
                                is_cropped=False,
                                is_favorite=False,
                                tags=json.dumps(tags),
                                recognition_suggestions=json.dumps([])
                            )
                            
                            # Set width and height
                            try:
                                with PILImage.open(file_path) as im:
                                    new_image.width = im.width
                                    new_image.height = im.height
                            except Exception as e:
                                print(f"Error setting image dimensions: {e}")
                            
                            db.session.add(new_image)
                            db.session.commit()
                            uploaded_count += 1
                            
                        except Exception as e:
                            errors.append(f"Error processing {file.filename}: {str(e)}")
                    else:
                        errors.append(f"{file.filename} is not a valid image or zip file")
        
        if uploaded_count > 0:
            flash(f'Successfully imported {uploaded_count} image(s)', 'success')
        if errors:
            flash('Errors: ' + ', '.join(errors), 'error')
        
        return redirect(url_for('import_workflow'))
    
    # Get counts for workflow progress
    imported_pending = Image.query.filter(Image.tags.contains('uncropped')).all()
    crop_pending = Image.query.filter(Image.tags.contains('uncropped')).all()
    name_pending = Image.query.filter(Image.tags.contains('uncropped')).all()
    
    return render_template('import_workflow.html', 
                         step='import',
                         imported_pending=imported_pending,
                         crop_pending=crop_pending,
                         name_pending=name_pending)

@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    """Handle folder uploads to imported directory, with .webp to .jpg conversion"""
    if 'files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('import_workflow'))
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('No files selected', 'error')
        return redirect(url_for('import_workflow'))
    imported_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'imported')
    os.makedirs(imported_folder, exist_ok=True)
    uploaded_count = 0
    errors = []
    for file in files:
        if file and allowed_file(file.filename):
            try:
                file.seek(0, 2)
                file_size = file.tell()
                file.seek(0)
                if file_size > MAX_FILE_SIZE:
                    errors.append(f"{file.filename} is too large (max 16MB)")
                    continue
                ext = os.path.splitext(file.filename)[1].lower()
                if ext == '.webp':
                    img = PILImage.open(file.stream).convert('RGB')
                    filename = secure_filename(os.path.splitext(file.filename)[0] + '.jpg')
                    base_name = f"{uuid.uuid4().hex}_{filename}"
                    file_path = os.path.join(imported_folder, base_name)
                    img.save(file_path, 'JPEG')
                else:
                    filename = secure_filename(file.filename)
                    base_name = f"{uuid.uuid4().hex}_{filename}"
                    file_path = os.path.join(imported_folder, base_name)
                    file.save(file_path)
                new_image = Image(
                    original_path=f'/static/uploads/imported/{base_name}',
                    filename=filename,
                    is_cropped=False,
                    is_favorite=False,
                    tags=json.dumps([]),
                    recognition_suggestions=json.dumps([])
                )
                # Set width and height
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(os.path.join(app.config['UPLOAD_FOLDER'], 'imported', base_name)) as im:
                        new_image.width = im.width
                        new_image.height = im.height
                except Exception as e:
                    print(f"Error setting image dimensions: {e}")
                db.session.add(new_image)
                db.session.commit()
                uploaded_count += 1
            except Exception as e:
                errors.append(f"Error processing {file.filename}: {str(e)}")
        else:
            errors.append(f"{file.filename} is not a valid image file")
    if uploaded_count > 0:
        flash(f'Successfully imported {uploaded_count} image(s) to imported folder', 'success')
        uncropped_count = Image.query.filter(Image.tags.contains('uncropped')).count()
        if uncropped_count > 0:
            return redirect(url_for('cropper'))
    if errors:
        flash('Errors: ' + ', '.join(errors), 'error')
    return redirect(url_for('import_workflow'))

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    if 'files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('import_page'))
    
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        flash('No files selected', 'error')
        return redirect(url_for('import_page'))
    
    uploaded_count = 0
    errors = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Check file size
                file.seek(0, 2)  # Seek to end
                file_size = file.tell()
                file.seek(0)  # Reset to beginning
                
                if file_size > MAX_FILE_SIZE:
                    errors.append(f"{file.filename} is too large (max 16MB)")
                    continue
                
                # Process the image
                result = process_uploaded_image(file)
                if result:
                    # Get or create unidentified person
                    unidentified_person = Person.query.filter_by(name='Unidentified').first()
                    if not unidentified_person:
                        unidentified_person = Person(name='Unidentified', is_confirmed=False)
                        db.session.add(unidentified_person)
                        db.session.commit()
                    
                    # Create new image record
                    new_image = Image(
                        original_path=result['original_path'],
                        cropped_path=result['cropped_path'],
                        filename=result['filename'],
                        is_favorite=False,
                        tags=json.dumps([]),
                        recognition_suggestions=json.dumps([])
                    )
                    # Set width and height
                    try:
                        from PIL import Image as PILImage
                        with PILImage.open(result['original_path'].replace('/static/', 'static/')) as im:
                            new_image.width = im.width
                            new_image.height = im.height
                    except Exception as e:
                        print(f"Error setting image dimensions: {e}")
                    db.session.add(new_image)
                    db.session.commit()
                    
                    # Associate image with unidentified person
                    new_image.people.append(unidentified_person)
                    db.session.commit()
                    
                    uploaded_count += 1
                else:
                    errors.append(f"Failed to process {file.filename}")
                    
            except Exception as e:
                errors.append(f"Error processing {file.filename}: {str(e)}")
        else:
            errors.append(f"{file.filename} is not a valid image file")
    
    if uploaded_count > 0:
        flash(f'Successfully uploaded {uploaded_count} image(s)', 'success')
    
    if errors:
        flash('Errors: ' + ', '.join(errors), 'error')
    
    return redirect(url_for('import_page'))

@app.route('/person/<int:person_id>')
def person_gallery(person_id):
    """Person Gallery page"""
    from models import Video
    person = Person.query.get_or_404(person_id)
    images = person.images  # Use the many-to-many relationship
    videos = person.videos  # Get videos for this person
    return render_template('person_gallery.html', 
                         person=person.to_dict(), 
                         images=[img.to_dict() for img in images],
                         videos=[video.to_dict() for video in videos])

@app.route('/images')
def all_images():
    """All Images page with filters (show all images in the database, only if file exists)"""
    page_size = 50
    sort_by = request.args.get('sort_by', 'id-desc')
    images_query = Image.query
    # Sorting logic
    if sort_by == 'id-desc':
        images_query = images_query.order_by(Image.id.desc())
    elif sort_by == 'id-asc':
        images_query = images_query.order_by(Image.id.asc())
    elif sort_by == 'name-asc':
        images_query = images_query.join(Image.people, isouter=True).order_by(Person.name.asc().nullslast(), Image.id.asc())
    elif sort_by == 'name-desc':
        images_query = images_query.join(Image.people, isouter=True).order_by(Person.name.desc().nullslast(), Image.id.desc())
    elif sort_by == 'date-newest':
        images_query = images_query.order_by(Image.created_at.desc(), Image.id.desc())
    elif sort_by == 'date-oldest':
        images_query = images_query.order_by(Image.created_at.asc(), Image.id.asc())
    elif sort_by == 'size-large':
        images_query = images_query.order_by((Image.width * Image.height).desc().nullslast(), Image.id.desc())
    elif sort_by == 'size-small':
        images_query = images_query.order_by((Image.width * Image.height).asc().nullslast(), Image.id.asc())
    elif sort_by == 'favorites':
        images_query = images_query.order_by(Image.is_favorite.desc(), Image.id.desc())
    else:
        images_query = images_query.order_by(Image.id.desc())
        sort_by = 'id-desc'
    # Only include images whose file exists on disk
    all_images = images_query.all()
    def file_exists(img):
        path = img.original_path
        if not path:
            return False
        file_path = path.replace('/static/', 'static/')
        return os.path.exists(file_path)
    filtered_images = [img for img in all_images if file_exists(img)]
    total = len(filtered_images)
    images = filtered_images[:page_size]
    # Only include people with >0 images
    people = [p for p in Person.query.order_by(Person.name).all() if len(p.images) > 0]
    return render_template('all_images.html', images=[img.to_dict() for img in images], people=[p.to_dict() for p in people], total_images=total, page_size=page_size, current_sort=sort_by)

@app.route('/image/<int:image_id>')
def image_detail(image_id):
    """Image Detail modal/page"""
    image = Image.query.get_or_404(image_id)
    # Get all people associated with this image with their IDs
    people_data = [{'id': person.id, 'name': person.name} for person in image.people]
    return render_template('image_detail.html', image=image.to_dict(), people=people_data)

# Global variable to store the face recognition database instance
face_recognition_db = None

def get_next_image_data():
    """Helper function to get the next image data for the name page"""
    import glob
    
    # Get all images from database with cropped paths, sorted by ID descending
    db_cropped_images = Image.query.filter(Image.cropped_path.like('%/cropped/%')).order_by(Image.id.desc()).all()
    
    image_to_show = None
    for db_image in db_cropped_images:
        if db_image.cropped_path:
            # Convert web path to file path
            file_path = db_image.cropped_path.replace('/static/', 'static/')
            if os.path.exists(file_path):
                # Skip if already moved to named_cropped or skipped_cropped
                if '/named_cropped/' in file_path or '/skipped_cropped/' in file_path:
                    continue
                # Try to open as image
                try:
                    with PILImage.open(file_path) as im:
                        im.verify()
                    image_to_show = file_path
                    break
                except Exception:
                    continue
    
    # If no database images found, fall back to scanning the folder directly
    if not image_to_show:
        cropped_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped')
        image_files = glob.glob(os.path.join(cropped_dir, '*'))
        image_files = [f for f in image_files if os.path.isfile(f)]
        
        for img_path in image_files:
            # Skip if already moved to named_cropped or skipped_cropped
            if '/named_cropped/' in img_path or '/skipped_cropped/' in img_path:
                continue
            # Try to open as image
            try:
                with PILImage.open(img_path) as im:
                    im.verify()
                image_to_show = img_path
                break
            except Exception:
                continue
    
    if not image_to_show:
        return {'image': None, 'face_db_ready': face_recognition_db is not None}
    
    # Debug output
    print(f"Debug - image_to_show: {image_to_show}")
    # Convert file path to web URL, handling both forward and backward slashes
    web_url = image_to_show.replace('\\', '/').replace('static/', '/static/')
    print(f"Debug - web_url: {web_url}")
    
    # Prepare data for template
    image_data = {
        'image_url': web_url,
        'face_box': None,  # Will be filled by frontend API call
        'predicted_name': '',
        'path': image_to_show
    }
    return {'image': image_data, 'face_db_ready': face_recognition_db is not None}

@app.route('/name', methods=['GET', 'POST'])
def name_page():
    """Assign a name to a cropped image (now just an image in uploads with a tag)"""
    import glob
    import json
    from models import Image, Person
    from flask import request, flash, redirect, url_for, render_template
    import os
    if request.method == 'POST':
        image_path = request.form.get('image_path')
        name = request.form.get('name', '').strip()
        skip = request.form.get('skip')
        delete = request.form.get('delete')
        rotate = request.form.get('rotate')
        if image_path and os.path.exists(image_path.replace('/static/', 'static/')):
            file_path = image_path.replace('/static/', 'static/')
            if rotate:
                try:
                    with PILImage.open(file_path) as im:
                        rotated = im.rotate(-90, expand=True)
                        rotated.save(file_path)
                except Exception as e:
                    print(f"Error rotating image: {e}")
                return redirect(url_for('name_page'))
            elif delete:
                # Remove from database if exists
                image_record = Image.query.filter_by(original_path=image_path).first()
                if image_record:
                    db.session.delete(image_record)
                    db.session.commit()
                # Remove file from disk
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting image file: {e}")
                return redirect(url_for('name_page'))
            elif skip:
                # Add or update tag to 'skipped'
                image_record = Image.query.filter_by(original_path=image_path).first()
                if image_record:
                    tags = json.loads(image_record.tags) if image_record.tags else []
                    if 'skipped' not in tags:
                        tags.append('skipped')
                        image_record.tags = json.dumps(tags)
                        db.session.commit()
                return redirect(url_for('name_page'))
            else:
                # Assign name to person in database
                if name:
                    person = Person.query.filter_by(name=name).first()
                    if not person:
                        person = Person(name=name, is_confirmed=True)
                        db.session.add(person)
                        db.session.flush()
                    image_record = Image.query.filter_by(original_path=image_path).first()
                    if image_record:
                        image_record.people.clear()
                        image_record.people.append(person)
                        # Update tags
                        tags = json.loads(image_record.tags) if image_record.tags else []
                        if 'named' not in tags:
                            tags.append('named')
                        image_record.tags = json.dumps(tags)
                        db.session.commit()
                return redirect(url_for('name_page'))
    # Find the first image in uploads folder with tag 'cropped' and not yet named
    images_to_name = Image.query.filter(~Image.tags.contains('named'), Image.tags.contains('cropped')).all()
    image_to_show = images_to_name[0] if images_to_name else None
    return render_template('name.html', image=image_to_show)

@app.route('/stats')
def stats():
    """Stats page with charts and analytics"""
    # Get basic stats
    total_people = Person.query.count()
    total_images = Image.query.count()
    confirmed_people = Person.query.filter_by(is_confirmed=True).count()
    unidentified_people = total_people - confirmed_people
    favorite_images = Image.query.filter_by(is_favorite=True).count()
    
    # Get people with most images
    people_with_counts = []
    for person in Person.query.order_by(Person.name).all():
        people_with_counts.append({
            'name': person.name,
            'image_count': len(person.images),
            'is_confirmed': person.is_confirmed
        })
    
    # Sort by image count (descending) and take top 20
    people_with_counts.sort(key=lambda x: x['image_count'], reverse=True)
    top_people = people_with_counts[:20]
    
    # Get image size distribution
    images_with_sizes = Image.query.filter(Image.width.isnot(None), Image.height.isnot(None)).all()
    size_ranges = {
        'Small (< 100k pixels)': 0,
        'Medium (100k-500k pixels)': 0,
        'Large (500k-1M pixels)': 0,
        'Very Large (> 1M pixels)': 0
    }
    
    for img in images_with_sizes:
        pixels = img.width * img.height
        if pixels < 100000:
            size_ranges['Small (< 100k pixels)'] += 1
        elif pixels < 500000:
            size_ranges['Medium (100k-500k pixels)'] += 1
        elif pixels < 1000000:
            size_ranges['Large (500k-1M pixels)'] += 1
        else:
            size_ranges['Very Large (> 1M pixels)'] += 1
    
    # Get multi-person image stats
    multi_person_images = 0
    for img in Image.query.all():
        if len(img.people) > 1:
            multi_person_images += 1
    
    stats_data = {
        'total_people': total_people,
        'total_images': total_images,
        'confirmed_people': confirmed_people,
        'unidentified_people': unidentified_people,
        'favorite_images': favorite_images,
        'multi_person_images': multi_person_images,
        'top_people': top_people,
        'size_ranges': size_ranges
    }
    
    return render_template('stats.html', stats=stats_data)

# API endpoints for AJAX interactions
@app.route('/api/people')
def api_people():
    """API endpoint for people data"""
    try:
        name_filter = request.args.get('name', '').strip()
        
        if name_filter:
            # Filter by name (case-insensitive partial match)
            people = Person.query.filter(Person.name.ilike(f'%{name_filter}%')).order_by(Person.name).all()
        else:
            # Get all people
            people = Person.query.order_by(Person.name).all()
            
        return jsonify({
            'success': True,
            'people': [{
                'id': person.id,
                'name': person.name,
                'is_confirmed': person.is_confirmed,
                'image_count': person.image_count,
                'video_count': len(person.videos) if hasattr(person, 'videos') else 0
            } for person in people]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/images')
def api_images():
    """API endpoint for images data"""
    images = Image.query.all()
    return jsonify([img.to_dict() for img in images])

@app.route('/api/toggle_favorite/<int:person_id>', methods=['POST'])
def toggle_person_favorite(person_id):
    """Toggle person favorite status"""
    person = Person.query.get_or_404(person_id)
    person.is_favorite = not person.is_favorite
    db.session.commit()
    return jsonify({'success': True, 'is_favorite': person.is_favorite})

@app.route('/api/toggle_image_favorite/<int:image_id>', methods=['POST'])
def toggle_image_favorite(image_id):
    """Toggle image favorite status"""
    image = Image.query.get_or_404(image_id)
    image.is_favorite = not image.is_favorite
    db.session.commit()
    return jsonify({'success': True, 'is_favorite': image.is_favorite})

@app.route('/api/assign_image/<int:image_id>/<int:person_id>', methods=['POST'])
def assign_image_to_person(image_id, person_id):
    """Assign image to person and move to named_cropped folder"""
    image = Image.query.get_or_404(image_id)
    person = Person.query.get_or_404(person_id)
    # Clear existing people and assign to new person
    image.people.clear()
    image.people.append(person)
    db.session.commit()
    # Move file from cropped to named_cropped
    if image.cropped_path and '/cropped/' in image.cropped_path:
        src_path = image.cropped_path.replace('/static/', 'static/')
        named_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'named_cropped')
        os.makedirs(named_dir, exist_ok=True)
        dest_path = os.path.join(named_dir, os.path.basename(src_path))
        try:
            shutil.move(src_path, dest_path)
            image.cropped_path = dest_path.replace('static/', '/static/')
            db.session.commit()
        except Exception as e:
            print(f"Error moving named image: {e}")
    remove_people_with_no_images()
    return jsonify({'success': True, 'message': f'Image assigned to {person.name}'})

@app.route('/api/create_person', methods=['POST'])
def create_person():
    """Create new person"""
    data = request.get_json()
    name = data.get('name', '').strip()
    
    if not name:
        return jsonify({'success': False, 'error': 'Name is required'}), 400
    
    # Check if person already exists
    existing_person = Person.query.filter_by(name=name).first()
    if existing_person:
        return jsonify({'success': False, 'error': 'Person already exists'}), 400
    
    person = Person(name=name, is_confirmed=True)
    db.session.add(person)
    db.session.commit()
    
    return jsonify({'success': True, 'person': person.to_dict()})

@app.route('/api/update_person/<int:person_id>', methods=['POST'])
def update_person(person_id):
    """Update person name"""
    try:
        person = Person.query.get_or_404(person_id)
        data = request.get_json()
        new_name = data.get('name', '').strip()
        
        if not new_name:
            return jsonify({'success': False, 'error': 'Name is required'}), 400
        
        # Check if the new name already exists (excluding current person)
        existing_person = Person.query.filter_by(name=new_name).first()
        if existing_person and existing_person.id != person_id:
            return jsonify({'success': False, 'error': 'A person with this name already exists'}), 400
        
        # Update the name
        person.name = new_name
        db.session.commit()
        
        return jsonify({'success': True, 'person': person.to_dict()})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/face_suggestions/<int:image_id>')
def get_face_suggestions(image_id):
    """Get face recognition suggestions for an image"""
    try:
        image = Image.query.get_or_404(image_id)
        
        # Get the image path - handle hash-prefixed filenames
        if image.cropped_path and os.path.exists(image.cropped_path):
            image_path = image.cropped_path
        else:
            # Search for the actual file with hash prefix
            base_filename = image.filename
            uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
            
            # Look for files that contain the base filename
            matching_files = []
            if os.path.exists(uploads_dir):
                for file in os.listdir(uploads_dir):
                    if base_filename in file:
                        matching_files.append(file)
            
            if matching_files:
                # Use the first matching file (prefer cropped versions)
                cropped_files = [f for f in matching_files if f.startswith('cropped_')]
                if cropped_files:
                    image_path = os.path.join(uploads_dir, cropped_files[0])
                else:
                    image_path = os.path.join(uploads_dir, matching_files[0])
            else:
                return jsonify({'success': False, 'error': f'No matching files found for {base_filename}'}), 404
        
        # Check if image file exists
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': f'Image file not found: {image_path}'}), 404
        
        # Get face recognition
        face_db = get_face_db()
        
        # Get similar faces (no threshold, get more candidates)
        similar_faces = face_db.find_similar_faces(image_path, threshold=0.0, top_k=50)
        
        # Format suggestions - ensure unique names
        seen_names = set()
        suggestions = []
        for face in similar_faces:
            if face['person_name'] not in seen_names and len(suggestions) < 6:
                seen_names.add(face['person_name'])
                suggestions.append({
                    'person_id': face['person_id'],
                    'name': face['person_name'],
                    'confidence': face['similarity'],
                    'is_confirmed': face['is_confirmed']
                })
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detect_faces/<int:image_id>')
def detect_faces(image_id):
    """Detect faces in an image and return bounding box coordinates"""
    try:
        image = Image.query.get_or_404(image_id)
        # Get the image path - handle hash-prefixed filenames
        if hasattr(image, 'cropped_path') and image.cropped_path and os.path.exists(image.cropped_path):
            image_path = image.cropped_path
        else:
            base_filename = image.filename
            uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
            matching_files = []
            if os.path.exists(uploads_dir):
                for file in os.listdir(uploads_dir):
                    if base_filename in file:
                        matching_files.append(file)
            if matching_files:
                cropped_files = [f for f in matching_files if f.startswith('cropped_')]
                if cropped_files:
                    image_path = os.path.join(uploads_dir, cropped_files[0])
                else:
                    image_path = os.path.join(uploads_dir, matching_files[0])
            else:
                print(f"[detect_faces] No matching files found for {base_filename}")
                return jsonify({'success': False, 'error': f'No matching files found for {base_filename}'}), 404
        print(f"[detect_faces] Using image path: {image_path}")
        if not os.path.exists(image_path):
            print(f"[detect_faces] Image file not found: {image_path}")
            return jsonify({'success': False, 'error': f'Image file not found: {image_path}'}), 404
        face_db = get_face_db()
        try:
            faces = face_db.detect_faces(image_path)
        except Exception as e:
            print(f"[detect_faces] Error during face detection: {e}")
            return jsonify({'success': False, 'error': f'Face detection error: {e}'}), 500
        print(f"[detect_faces] Faces detected: {faces}")
        return jsonify({
            'success': True,
            'faces': faces
        })
    except Exception as e:
        print(f"[detect_faces] Outer exception: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/detect_faces_from_image', methods=['POST'])
def detect_faces_from_image():
    """Detect faces from uploaded image data and return face detection results"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        video_id = request.form.get('video_id')
        
        # video_id is optional - can be used for video frames from profile picture management
        # If not provided, we'll just proceed with face detection
        
        # Save the uploaded image temporarily
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_frame_{uuid4().hex}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        try:
            image_file.save(temp_path)
            
            # Get face database
            face_db = get_face_db()
            
            # Detect faces in the image
            print(f"[detect_faces_from_image] Detecting faces in temporary image: {temp_path}")
            faces = face_db.detect_faces(temp_path)
            print(f"[detect_faces_from_image] Faces detected: {faces}")
            
            if not faces:
                return jsonify({
                    'success': True,
                    'faces': [],
                    'face_bbox': None,
                    'all_faces': [],
                    'face_best_guesses': [],
                    'suggestions': []
                })
            
            # Process each face individually for recognition (like photo namer)
            suggestions = []
            face_best_guesses = []
            
            for i, face in enumerate(faces):
                print(f"[detect_faces_from_image] Processing face {i + 1} individually: {face}")
                
                try:
                    # Extract individual face region from image (like photo namer)
                    from PIL import Image as PILImage
                    
                    # Load the image
                    im = PILImage.open(temp_path)
                    
                    # Add padding around face (like photo namer)
                    padding_x = int(face['width'] * 0.5)
                    padding_y = int(face['height'] * 0.5)
                    
                    # Calculate crop box with padding
                    x1 = max(0, face['x'] - padding_x)
                    y1 = max(0, face['y'] - padding_y)
                    x2 = min(im.width, face['x'] + face['width'] + padding_x)
                    y2 = min(im.height, face['y'] + face['height'] + padding_y)
                    
                    # Crop the individual face
                    face_crop = im.crop((x1, y1, x2, y2))
                    
                    # Ensure minimum size for face recognition (at least 112x112)
                    min_size = 112
                    if face_crop.width < min_size or face_crop.height < min_size:
                        # Resize to minimum size while maintaining aspect ratio
                        ratio = max(min_size / face_crop.width, min_size / face_crop.height)
                        new_width = int(face_crop.width * ratio)
                        new_height = int(face_crop.height * ratio)
                        face_crop = face_crop.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                    
                    # Save individual face as temporary file
                    temp_crop_path = os.path.join(tempfile.gettempdir(), f'temp_face_crop_{i}_{uuid4().hex}.jpg')
                    face_crop.save(temp_crop_path)
                    
                    # Process THIS specific face individually
                    face_suggestions = face_db.find_similar_faces(temp_crop_path, threshold=0.0, top_k=5)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_crop_path)
                    except Exception:
                        pass
                    
                    if face_suggestions:
                        # Get the best suggestion for this specific face
                        best_suggestion = face_suggestions[0]
                        best_guess = best_suggestion['person_name']
                        face_best_guesses.append(best_guess)
                        
                        # Add to overall suggestions if not already present
                        if not any(s['name'] == best_guess for s in suggestions):
                            suggestions.append({
                                'person_id': best_suggestion['person_id'],
                                'name': best_suggestion['person_name'],
                                'confidence': best_suggestion['similarity'],
                                'is_confirmed': best_suggestion['is_confirmed']
                            })
                        
                        print(f"[detect_faces_from_image] Face {i + 1} individually matched: {best_guess} (confidence: {best_suggestion['similarity']:.3f})")
                    else:
                        # Try fallback with original thumbnail for this specific face (only if video_id provided)
                        if video_id and video_id != '0':
                            print(f"[detect_faces_from_image] No suggestions for face {i + 1}, trying fallback")
                            try:
                                from models import Video
                                video = Video.query.get(video_id)
                                if video and video.thumbnail_path:
                                    thumbnail_path = video.thumbnail_path
                                    if thumbnail_path.startswith('/static/'):
                                        thumbnail_path = thumbnail_path[1:]
                                    
                                    if os.path.exists(thumbnail_path):
                                        # Process original thumbnail for this face
                                        fallback_suggestions = face_db.find_similar_faces(thumbnail_path, threshold=0.0, top_k=5)
                                        if fallback_suggestions:
                                            best_suggestion = fallback_suggestions[0]
                                            best_guess = best_suggestion['person_name']
                                            face_best_guesses.append(best_guess)
                                            
                                            if not any(s['name'] == best_guess for s in suggestions):
                                                suggestions.append({
                                                    'person_id': best_suggestion['person_id'],
                                                    'name': best_suggestion['person_name'],
                                                    'confidence': best_suggestion['similarity'],
                                                    'is_confirmed': best_suggestion['is_confirmed']
                                                })
                                            
                                            print(f"[detect_faces_from_image] Face {i + 1} fallback matched: {best_guess}")
                                        else:
                                            face_best_guesses.append(f"Face {i + 1}")
                                            print(f"[detect_faces_from_image] Face {i + 1} fallback failed, using default")
                                    else:
                                        face_best_guesses.append(f"Face {i + 1}")
                                        print(f"[detect_faces_from_image] Face {i + 1} fallback thumbnail not found")
                                else:
                                    face_best_guesses.append(f"Face {i + 1}")
                                    print(f"[detect_faces_from_image] Face {i + 1} fallback video not found")
                            except Exception as e:
                                face_best_guesses.append(f"Face {i + 1}")
                                print(f"[detect_faces_from_image] Face {i + 1} fallback error: {e}")
                        else:
                            face_best_guesses.append(f"Face {i + 1}")
                            print(f"[detect_faces_from_image] Face {i + 1} no video_id, skipping fallback")
                
                except Exception as e:
                    face_best_guesses.append(f"Face {i + 1}")
                    print(f"[detect_faces_from_image] Error processing face {i + 1} individually: {e}")
            
            print(f"[detect_faces_from_image] Final individual face assignments: {face_best_guesses}")
            print(f"[detect_faces_from_image] Final suggestions: {len(suggestions)}")
            
            # Prepare response data
            response_data = {
                'success': True,
                'faces': faces,
                'face_bbox': faces[0] if faces else None,  # Use first face as primary
                'all_faces': faces,
                'face_best_guesses': face_best_guesses,
                'suggestions': suggestions[:5]  # Limit to 5 suggestions total
            }
            
            print(f"[detect_faces_from_image] Returning face data: {len(faces)} faces, {len(suggestions)} suggestions")
            return jsonify(response_data)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        print(f"[detect_faces_from_image] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get_face_suggestions', methods=['POST'])
def api_get_face_suggestions():
    """Get suggestions for a specific face in an image"""
    try:
        data = request.get_json() or {}
        image_data = data.get('image_data')
        face_index = data.get('face_index')
        video_id = data.get('video_id')
        
        if not image_data or face_index is None or not video_id:
            return jsonify({'success': False, 'error': 'Missing required parameters'}), 400
        
        # Save the uploaded image temporarily
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_face_suggestions_{uuid4().hex}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        try:
            # Convert data URL to image file
            import base64
            if image_data.startswith('data:image'):
                # Remove data URL prefix
                image_data = image_data.split(',')[1]
            
            # Decode base64 and save
            with open(temp_path, 'wb') as f:
                f.write(base64.b64decode(image_data))
            
            # Get face database
            face_db = get_face_db()
            
            # Detect faces in the image
            faces = face_db.detect_faces(temp_path)
            
            if not faces or face_index >= len(faces):
                return jsonify({'success': True, 'suggestions': []})
            
            # Extract the specific face
            face = faces[face_index]
            
            # Extract individual face region (like photo namer)
            from PIL import Image as PILImage
            im = PILImage.open(temp_path)
            
            # Add padding around face
            padding_x = int(face['width'] * 0.5)
            padding_y = int(face['height'] * 0.5)
            
            # Calculate crop box with padding
            x1 = max(0, face['x'] - padding_x)
            y1 = max(0, face['y'] - padding_y)
            x2 = min(im.width, face['x'] + face['width'] + padding_x)
            y2 = min(im.height, face['y'] + face['height'] + padding_y)
            
            # Crop the individual face
            face_crop = im.crop((x1, y1, x2, y2))
            
            # Ensure minimum size for face recognition
            min_size = 112
            if face_crop.width < min_size or face_crop.height < min_size:
                ratio = max(min_size / face_crop.width, min_size / face_crop.height)
                new_width = int(face_crop.width * ratio)
                new_height = int(face_crop.height * ratio)
                face_crop = face_crop.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
            
            # Save individual face as temporary file
            temp_crop_path = os.path.join(temp_dir, f'temp_face_crop_{face_index}_{uuid4().hex}.jpg')
            face_crop.save(temp_crop_path)
            
            # Get suggestions for this specific face
            face_suggestions = face_db.find_similar_faces(temp_crop_path, threshold=0.0, top_k=10)
            
            # Clean up temp files
            try:
                os.remove(temp_path)
                os.remove(temp_crop_path)
            except Exception:
                pass
            
            # Format suggestions
            suggestions = []
            if face_suggestions:
                for suggestion in face_suggestions:
                    suggestions.append({
                        'name': suggestion['person_name'],
                        'confidence': suggestion['similarity'],
                        'person_id': suggestion['person_id'],
                        'is_confirmed': suggestion['is_confirmed']
                    })
            
            return jsonify({'success': True, 'suggestions': suggestions})
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                    
    except Exception as e:
        print(f"[get_face_suggestions] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/remove_all_people_from_video/<int:video_id>', methods=['POST'])
def remove_all_people_from_video(video_id):
    """Remove all people associations from a video"""
    try:
        print(f"[remove_all_people_from_video] Removing all people from video {video_id}")
        
        # Import Video model
        from models import Video
        
        # Get the video
        video = Video.query.get(video_id)
        if not video:
            return jsonify({'success': False, 'error': 'Video not found'}), 404
        
        # Clear all people associations
        video.people.clear()
        
        # Update video tags to remove "named" tag
        if video.tags:
            try:
                tags = json.loads(video.tags) if isinstance(video.tags, str) else video.tags
                if isinstance(tags, list):
                    # Remove "named" tag if present
                    tags = [tag for tag in tags if tag != "named"]
                    video.tags = json.dumps(tags)
                elif isinstance(tags, dict):
                    # Remove "named" key if present
                    tags.pop("named", None)
                    video.tags = json.dumps(tags)
            except (json.JSONDecodeError, TypeError):
                # If tags can't be parsed, set to empty list
                video.tags = json.dumps([])
        else:
            video.tags = json.dumps([])
        
        # Commit changes
        db.session.commit()
        
        print(f"[remove_all_people_from_video] Successfully removed all people from video {video_id}")
        
        return jsonify({'success': True, 'message': 'All people removed from video'})
        
    except Exception as e:
        print(f"[remove_all_people_from_video] Error: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/viewer/images')
def api_viewer_images():
    """API endpoint to load images for viewer with pagination"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    favorites_only = request.args.get('favorites') == 'true'
    people_param = request.args.get('people', '')
    person_id = request.args.get('person_id', type=int)
    
    # Build query based on parameters
    if person_id:
        images_query = Image.query.join(image_people).filter(image_people.c.person_id == person_id)
    elif favorites_only:
        images_query = Image.query.filter_by(is_favorite=True)
    elif people_param:
        try:
            person_ids = [int(pid) for pid in people_param.split(',') if pid.strip()]
            images_query = Image.query.join(image_people).filter(image_people.c.person_id.in_(person_ids))
        except (ValueError, TypeError):
            images_query = Image.query
    else:
        images_query = Image.query
    
    # Get total count first
    total_images = images_query.count()
    
    # If per_page is set to total_images or higher, load all images
    if per_page >= total_images:
        images = images_query.order_by(Image.id.desc()).all()
        page = 1
    else:
        # Apply pagination
        images = images_query.order_by(Image.id.desc()).offset((page - 1) * per_page).limit(per_page).all()
    
    return jsonify({
        'success': True,
        'images': [img.to_dict() for img in images],
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total_images,
            'pages': (total_images + per_page - 1) // per_page
        }
    })

@app.route('/api/people')
def get_people():
    try:
        people = Person.query.order_by(Person.name).all()
        return jsonify({
            'success': True,
            'people': [{
                'id': person.id,
                'name': person.name,
                'is_confirmed': person.is_confirmed,
                'image_count': person.image_count
            } for person in people]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/viewer')
@app.route('/viewer/<int:person_id>')
@app.route('/viewer/favorites')
def viewer(person_id=None):
    """Advanced image viewer with navigation and display modes"""
    
    # Get query parameters for multiple person selection
    favorites_only = request.args.get('favorites') == 'true'
    people_param = request.args.get('people', '')
    page = request.args.get('page', 1, type=int)
    per_page = 100  # Load first 100 images initially
    
    # Determine what images to show
    if person_id:
        # Specific person's images (backward compatibility)
        person = Person.query.get_or_404(person_id)
        images_query = Image.query.join(image_people).filter(image_people.c.person_id == person_id)
        title = f"Viewer - {person.name}"
        mode = "person"
    elif favorites_only:
        # Favorite images only
        images_query = Image.query.filter_by(is_favorite=True)
        title = "Viewer - Favorites"
        mode = "favorites"
    elif people_param:
        # Multiple people selected
        try:
            person_ids = [int(pid) for pid in people_param.split(',') if pid.strip()]
            people = Person.query.filter(Person.id.in_(person_ids)).all()
            
            if len(people) == 1:
                title = f"Viewer - {people[0].name}"
            else:
                title = f"Viewer - {len(people)} People"
            
            # Get images that contain any of the selected people
            images_query = Image.query.join(image_people).filter(image_people.c.person_id.in_(person_ids))
            mode = "multiple_people"
        except (ValueError, TypeError):
            # Fallback to all images if parsing fails
            images_query = Image.query
            title = "Viewer - All Images"
            mode = "all"
    else:
        # All images
        images_query = Image.query
        title = "Viewer - All Images"
        mode = "all"
    
    # Apply pagination - load all images initially for better experience
    total_images = images_query.count()
    images = images_query.order_by(Image.id.desc()).all()  # Load all images
    
    # Convert images to dictionaries using the to_dict() method for consistency
    images_data = [img.to_dict() for img in images]
    
    # Get people for the person selector (limit to first 100 for performance)
    people = Person.query.order_by(Person.name).limit(100).all()
    
    return render_template('viewer.html', 
                         images=images_data, 
                         people=[p.to_dict() for p in people], 
                         title=title, 
                         mode=mode, 
                         person_id=person_id,
                         current_sort='id-desc',
                         pagination={
                             'page': page,
                             'per_page': per_page,
                             'total': total_images,
                             'pages': (total_images + per_page - 1) // per_page
                         })

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        password = request.form.get('password')
        if password == ADMIN_PASSWORD:
            session['logged_in'] = True
            next_url = request.args.get('next') or url_for('home')
            return redirect(next_url)
        else:
            error = 'Invalid password.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Protect all routes except login/logout and API endpoints
for rule in list(app.url_map.iter_rules()):
    if rule.endpoint not in ('login', 'logout', 'static', 'detect_faces_from_image', 'api_get_face_suggestions', 'remove_all_people_from_video'):
        view_func = app.view_functions[rule.endpoint]
        app.view_functions[rule.endpoint] = login_required(view_func)

# Utility function to remove people with no images
def remove_people_with_no_images():
    people_to_remove = Person.query.filter(~Person.images.any()).all()
    for person in people_to_remove:
        db.session.delete(person)
    if people_to_remove:
        db.session.commit()

@app.route('/quick-slideshow')
def quick_slideshow():
    """Fullscreen quick slideshow of all people images, random, auto-next, 1s interval, minimal UI"""
    images = Image.query.order_by(Image.id.desc()).all()
    return render_template('quick_slideshow.html', images=[img.to_dict() for img in images])

@app.route('/mobile-slideshow')
def mobile_slideshow():
    """Optimized slideshow for mobile/tablet devices with touch controls and network-friendly loading"""
    sort_by = request.args.get('sort_by', 'id-desc')
    images_query = Image.query
    
    # Sorting logic (same as all_images route)
    if sort_by == 'id-desc':
        images_query = images_query.order_by(Image.id.desc())
    elif sort_by == 'id-asc':
        images_query = images_query.order_by(Image.id.asc())
    elif sort_by == 'name-asc':
        images_query = images_query.join(Image.people, isouter=True).order_by(Person.name.asc().nullslast(), Image.id.asc())
    elif sort_by == 'name-desc':
        images_query = images_query.join(Image.people, isouter=True).order_by(Person.name.desc().nullslast(), Image.id.desc())
    elif sort_by == 'date-newest':
        images_query = images_query.order_by(Image.created_at.desc(), Image.id.desc())
    elif sort_by == 'date-oldest':
        images_query = images_query.order_by(Image.created_at.asc(), Image.id.asc())
    elif sort_by == 'size-large':
        images_query = images_query.order_by((Image.width * Image.height).desc().nullslast(), Image.id.desc())
    elif sort_by == 'size-small':
        images_query = images_query.order_by((Image.width * Image.height).asc().nullslast(), Image.id.asc())
    elif sort_by == 'favorites':
        images_query = images_query.order_by(Image.is_favorite.desc(), Image.id.desc())
    elif sort_by == 'random':
        import random
        images = images_query.all()
        random.shuffle(images)
        return render_template('mobile_slideshow.html', images=[img.to_dict() for img in images], sort_by=sort_by)
    else:
        images_query = images_query.order_by(Image.id.desc())
        sort_by = 'id-desc'
    
    images = images_query.all()
    return render_template('mobile_slideshow.html', images=[img.to_dict() for img in images], sort_by=sort_by)

@app.route('/random-grid')
def random_grid():
    """Simple grid of 6 random images that refreshes on 'n' press"""
    import random
    # Get all images that exist on disk
    all_images = Image.query.all()
    def file_exists(img):
        if img.original_path:
            file_path = img.original_path.replace('/static/', 'static/')
            if os.path.exists(file_path):
                return True
        uploads_path = os.path.join('static', 'uploads', img.filename)
        if os.path.exists(uploads_path):
            return True
        return False
    
    filtered_images = [img for img in all_images if file_exists(img)]
    
    # Select 6 random images
    random.shuffle(filtered_images)
    random_images = filtered_images[:6]
    
    return render_template('random_grid.html', images=[img.to_dict() for img in random_images])

@app.route('/workflow')
def workflow():
    """Complete workflow page"""
    return render_template('workflow.html')

@app.route('/experimental')
def experimental_page():
    """Standalone experimental justified grid page"""
    return render_template('experimental.html')

@app.route('/api/delete_image/<int:image_id>', methods=['POST'])
def delete_image(image_id):
    """Delete an image by ID, including DB record and files"""
    image = Image.query.get_or_404(image_id)
    # Remove files
    for path in [image.original_path, getattr(image, 'cropped_path', None)]:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    # Remove uploaded file if not already covered
    uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
    if image.filename:
        for file in os.listdir(uploads_dir):
            if image.filename in file:
                try:
                    os.remove(os.path.join(uploads_dir, file))
                except Exception:
                    pass
    # Remove from DB
    db.session.delete(image)
    db.session.commit()
    # Find the next uncropped image (by ID)
    import json
    next_image = Image.query.filter(Image.tags.contains('uncropped'), Image.id > image_id).order_by(Image.id.asc()).first()
    next_id = next_image.id if next_image else None
    return jsonify({'success': True, 'next_image_id': next_id})

@app.route('/api/yolo/detect', methods=['POST'])
def yolo_detect():
    """YOLO person detection endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save uploaded file temporarily
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_path)
        
        # Get YOLO detector
        detector = get_yolo_detector()
        
        # Detect persons
        persons = detector.detect_persons(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'persons_found': len(persons),
            'detections': persons
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/yolo/process', methods=['POST'])
def yolo_process():
    """Process image with YOLO and save crops"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        base_name = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], base_name)
        file.save(file_path)
        
        # Get YOLO detector
        detector = get_yolo_detector()
        
        # Process image
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped_magic')
        result = detector.process_image(file_path, output_dir, confidence_threshold=0.5)
        
        # Convert paths to web URLs
        web_paths = []
        for crop_path in result['cropped_paths']:
            web_path = crop_path.replace('static/', '/static/')
            web_paths.append(web_path)
        
        return jsonify({
            'success': True,
            'original_path': f'/static/uploads/{base_name}',
            'persons_found': result['persons_found'],
            'crops': result['crops'],
            'cropped_paths': web_paths
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/yolo/process_imported', methods=['POST'])
def yolo_process_imported():
    """Process imported image with YOLO and save crops to cropped folder"""
    try:
        data = request.get_json()
        if not data or 'image_path' not in data:
            return jsonify({'success': False, 'error': 'No image path provided'}), 400
        
        image_path = data['image_path']
        filename = data.get('filename', 'unknown')
        
        # Convert web path to file path
        file_path = image_path.replace('/static/', 'static/')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Image file not found'}), 404
        
        # Get YOLO detector
        detector = get_yolo_detector()
        
        # Create cropped folder if it doesn't exist
        cropped_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped')
        os.makedirs(cropped_folder, exist_ok=True)
        
        # Process image and save to cropped folder
        result = detector.process_image(file_path, cropped_folder, confidence_threshold=0.5)
        
        # Convert paths to web URLs and create database records
        web_paths = []
        for i, crop_path in enumerate(result['cropped_paths']):
            web_path = crop_path.replace('static/', '/static/')
            web_paths.append(web_path)
            
            # Create database record for cropped image
            cropped_filename = f"cropped_{filename}_{i+1}.jpg"
            new_image = Image(
                original_path=image_path,  # Reference to original imported image
                cropped_path=web_path,
                filename=cropped_filename,
                is_favorite=False,
                tags=json.dumps([]),
                recognition_suggestions=json.dumps([])
            )
            # Set width and height
            try:
                from PIL import Image as PILImage
                with PILImage.open(crop_path) as im:
                    new_image.width = im.width
                    new_image.height = im.height
            except Exception as e:
                print(f"Error setting image dimensions: {e}")
            db.session.add(new_image)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'original_path': image_path,
            'persons_found': result['persons_found'],
            'crops': result['crops'],
            'cropped_paths': web_paths
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/face_suggestions/<path:image_path>')
def get_face_suggestions_by_path(image_path):
    """Get face recognition suggestions for an image by path"""
    try:
        # Convert web path to file path
        file_path = image_path.replace('/static/', 'static/')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        # Get face recognition suggestions
        face_db = get_face_db()
        
        # Get similar faces (no threshold, get more candidates)
        similar_faces = face_db.find_similar_faces(file_path, threshold=0.0, top_k=50)
        
        # Format suggestions - ensure unique names
        seen_names = set()
        suggestions = []
        for face in similar_faces:
            if face['person_name'] not in seen_names and len(suggestions) < 6:
                seen_names.add(face['person_name'])
                suggestions.append({
                    'person_id': face['person_id'],
                    'name': face['person_name'],
                    'confidence': face['similarity'],
                    'is_confirmed': face['is_confirmed']
                })
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/workflow/add_person', methods=['POST'])
def add_person_to_workflow():
    """Add a person from workflow to the database"""
    try:
        data = request.get_json()
        name = data.get('name')
        image_path = data.get('image_path')
        original_filename = data.get('original_filename')
        
        if not name or not image_path:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Get or create person
        person = Person.query.filter_by(name=name).first()
        if not person:
            person = Person(name=name, is_confirmed=True)
            db.session.add(person)
            db.session.commit()
        
        # Create image record
        new_image = Image(
            original_path=image_path,
            cropped_path=image_path,  # For workflow, cropped is the same as original
            filename=original_filename or 'workflow_import',
            is_favorite=False,
            tags=json.dumps([]),
            recognition_suggestions=json.dumps([])
        )
        # Set width and height
        try:
            from PIL import Image as PILImage
            with PILImage.open(image_path.replace('/static/', 'static/')) as im:
                new_image.width = im.width
                new_image.height = im.height
        except Exception as e:
            print(f"Error setting image dimensions: {e}")
        db.session.add(new_image)
        db.session.commit()
        
        # Associate image with person
        new_image.people.append(person)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'person_id': person.id,
            'image_id': new_image.id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload_temp', methods=['POST'])
def upload_temp():
    """Upload multiple images to a unique session directory and return session_id and URLs"""
    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'error': 'No files uploaded'}), 400

    session_id = str(uuid4())
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', session_id)
    os.makedirs(session_dir, exist_ok=True)
    urls = []
    for file in files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(session_dir, filename)
        file.save(save_path)
        url = f'/static/uploads/temp/{session_id}/{filename}'
        urls.append(url)
    return jsonify({'success': True, 'session_id': session_id, 'urls': urls})

@app.route('/api/yolo/process_url', methods=['POST'])
def yolo_process_url():
    """Process image by server URL with YOLO and save crops"""
    try:
        image_url = request.form.get('image_url')
        if not image_url:
            return jsonify({'success': False, 'error': 'No image_url provided'}), 400
        # Convert web path to file path
        if image_url.startswith('/static/'):
            file_path = image_url.replace('/static/', 'static/')
        else:
            return jsonify({'success': False, 'error': 'Invalid image_url'}), 400
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        # Get YOLO detector
        detector = get_yolo_detector()
        # Process image
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped_magic')
        result = detector.process_image(file_path, output_dir, confidence_threshold=0.5)
        # Convert paths to web URLs
        web_paths = []
        for crop_path in result['cropped_paths']:
            web_path = crop_path.replace('static/', '/static/')
            web_paths.append(web_path)
        return jsonify({
            'success': True,
            'original_path': image_url,
            'persons_found': result['persons_found'],
            'crops': result['crops'],
            'cropped_paths': web_paths
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/crop_progress/<session_id>', methods=['POST'])
def save_crop_progress(session_id):
    """Save crop progress for a session as JSON in the session's temp directory"""
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', session_id)
    os.makedirs(session_dir, exist_ok=True)
    progress_path = os.path.join(session_dir, 'progress.json')
    try:
        progress = request.get_json()
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress, f)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/crop_progress/<session_id>', methods=['GET'])
def load_crop_progress(session_id):
    """Load crop progress for a session from JSON in the session's temp directory"""
    session_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp', session_id)
    progress_path = os.path.join(session_dir, 'progress.json')
    try:
        if not os.path.exists(progress_path):
            return jsonify({'success': True, 'progress': {}})
        with open(progress_path, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        return jsonify({'success': True, 'progress': progress})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/yolo/batch_process', methods=['POST'])
def yolo_batch_process():
    """Batch process images in parallel and return those needing user input (0 or >1 person)"""
    try:
        data = request.get_json()
        image_urls = data.get('image_urls', [])
        if not image_urls:
            return jsonify({'success': False, 'error': 'No image_urls provided'}), 400
        
        # For now, process just the first few images to avoid freezing
        # We'll implement proper async processing later
        image_urls = image_urls[:10]  # Limit to first 10 images for testing
        
        # Ensure output directory exists
        output_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped_magic')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get YOLO detector (shared instance)
        detector = get_yolo_detector()
        
        results = []
        
        def process_one(image_url):
            try:
                # Convert web path to file path
                if image_url.startswith('/static/'):
                    file_path = image_url.replace('/static/', 'static/')
                else:
                    return None
                
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    return None
                
                # Process image with YOLO
                result = detector.process_image(file_path, output_dir, confidence_threshold=0.5)
                
                # Convert paths to web URLs
                web_paths = []
                for crop_path in result['cropped_paths']:
                    web_path = crop_path.replace('static/', '/static/')
                    web_paths.append(web_path)
                
                return {
                    'image_url': image_url,
                    'persons_found': result['persons_found'],
                    'crops': result['crops'],
                    'cropped_paths': web_paths
                }
            except Exception as e:
                print(f"Error processing {image_url}: {str(e)}")
                return None
        
        # Process images sequentially for now to avoid freezing
        print(f"Processing {len(image_urls)} images...")
        for i, image_url in enumerate(image_urls):
            print(f"Processing image {i+1}/{len(image_urls)}: {image_url}")
            res = process_one(image_url)
            if res:
                print(f"Processed {res['image_url']}: {res['persons_found']} persons found")
                if res['persons_found'] != 1:
                    results.append(res)
            else:
                print(f"Failed to process image")
        
        print(f"Batch processing complete. {len(results)} images need user input.")
        
        return jsonify({'success': True, 'queue': results})
        
    except Exception as e:
        print(f"Batch process error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/report_bug', methods=['POST'])
def submit_bug_report():
    """Submit a bug report"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Basic validation
        required_fields = ['description', 'severity']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400

        # Get user context
        user_context = {
            'page': data.get('page', request.referrer),
            'url': data.get('url', request.url),
            'user_agent': data.get('user_agent', request.headers.get('User-Agent')),
            'timestamp': data.get('timestamp', datetime.now().isoformat())
        }

        # Save the report
        success = save_bug_report(data | user_context)
        if success:
            return jsonify({'success': True, 'message': 'Bug report submitted successfully!'})
        else:
            return jsonify({'success': False, 'error': 'Failed to save bug report.'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/manual_crop', methods=['POST'])
def manual_crop():
    """Crop a region from an imported image and save to uploads folder, then update DB"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        filename = data.get('filename', 'manual_crop.jpg')
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        width = int(data.get('width', 0))
        height = int(data.get('height', 0))
        if not image_path or width <= 0 or height <= 0:
            return jsonify({'success': False, 'error': 'Invalid crop parameters'}), 400
        file_path = image_path.replace('/static/', 'static/')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Image file not found'}), 404
        # Open and crop the image
        with PILImage.open(file_path) as img:
            crop_box = (x, y, x + width, y + height)
            cropped_img = img.crop(crop_box)
            # Save to uploads folder
            cropped_filename = f"manualcrop_{uuid.uuid4().hex}_{filename}"
            cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
            cropped_img.save(cropped_path)
        # Add to database
        web_path = f'/static/uploads/{cropped_filename}'
        new_image = Image(
            original_path=web_path,
            filename=cropped_filename,
            is_favorite=False,
            tags=json.dumps(['cropped']),
            recognition_suggestions=json.dumps([])
        )
        # Set width and height
        try:
            from PIL import Image as PILImage
            with PILImage.open(cropped_path) as im:
                new_image.width = im.width
                new_image.height = im.height
        except Exception as e:
            print(f"Error setting image dimensions: {e}")
        
        # CRITICAL: Ensure the new crop image has NO people assigned
        # This prevents any automatic assignment from the original image
        new_image.people.clear()
        
        db.session.add(new_image)
        db.session.commit()
        return jsonify({'success': True, 'cropped_path': web_path})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/yolo/save_crops', methods=['POST'])
def yolo_save_crops():
    """Save selected crops, update DB"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        filename = data.get('filename', 'unknown.jpg')
        crops = data.get('crops', [])  # list of {box: [x1, y1, x2, y2]}
        if not image_path or not crops:
            return jsonify({'success': False, 'error': 'Missing data'}), 400
        file_path = image_path.replace('/static/', 'static/')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Image file not found'}), 404
        # Open image
        with PILImage.open(file_path) as img:
            saved_paths = []
            for idx, crop in enumerate(crops):
                box = crop['box']
                crop_img = img.crop((box[0], box[1], box[2], box[3]))
                cropped_filename = f"crop_{uuid.uuid4().hex}_{filename}"
                cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
                crop_img.save(cropped_path)
                web_path = f'/static/uploads/{cropped_filename}'
                new_image = Image(
                    original_path=web_path,
                    filename=cropped_filename,
                    is_favorite=False,
                    tags=json.dumps(['cropped']),
                    recognition_suggestions=json.dumps([])
                )
                # Set width and height
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(cropped_path) as im:
                        new_image.width = im.width
                        new_image.height = im.height
                except Exception as e:
                    print(f"Error setting image dimensions: {e}")
                
                # CRITICAL: Ensure the new crop image has NO people assigned
                # This prevents any automatic assignment from the original image
                new_image.people.clear()
                
                db.session.add(new_image)
                saved_paths.append(web_path)
            db.session.commit()
        return jsonify({'success': True, 'saved': saved_paths})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/sync_cropped')
def admin_sync_cropped():
    """Scan static/uploads/cropped for images not in the DB and add them as Image entries."""
    import glob
    added = 0
    cropped_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped')
    if not os.path.exists(cropped_dir):
        return 'Cropped folder does not exist.'
    # Get all image files in cropped folder
    image_files = glob.glob(os.path.join(cropped_dir, '*'))
    from models import Image
    for img_path in image_files:
        # Convert to web path
        web_path = img_path.replace('static/', '/static/')
        # Check if already in DB
        exists = Image.query.filter_by(cropped_path=web_path).first()
        if not exists:
            filename = os.path.basename(img_path)
            new_image = Image(
                original_path=None,
                cropped_path=web_path,
                filename=filename,
                is_favorite=False,
                tags='[]',
                recognition_suggestions='[]'
            )
            db.session.add(new_image)
            added += 1
    db.session.commit()
    return f'Synced. Added {added} new cropped images to the database.'

@app.route('/api/rotate_image', methods=['POST'])
def rotate_image():
    """Rotate an image by 90 degrees in the specified direction"""
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        direction = data.get('direction', 'cw')  # 'cw' or 'ccw'
        
        if not image_path:
            return jsonify({'success': False, 'error': 'No image_path provided'}), 400
        
        # Convert web path to file path
        file_path = image_path.replace('/static/', 'static/')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Image file not found'}), 404
        
        # Determine rotation angle
        if direction == 'cw':
            angle = -90  # Clockwise
        elif direction == 'ccw':
            angle = 90   # Counter-clockwise
        else:
            return jsonify({'success': False, 'error': 'Invalid direction. Use "cw" or "ccw"'}), 400
        
        # Open and rotate the image
        with PILImage.open(file_path) as img:
            rotated = img.rotate(angle, expand=True)
            rotated.save(file_path)
        
        # Return the same image URL (file was updated in place)
        return jsonify({
            'success': True,
            'image_url': image_path,
            'direction': direction
        })
        
    except Exception as e:
        print(f"Error rotating image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rotate_image_by_id/<int:image_id>', methods=['POST'])
def rotate_image_by_id(image_id):
    """Rotate an image by 90 degrees in the specified direction using image ID"""
    try:
        image = Image.query.get_or_404(image_id)
        data = request.get_json()
        direction = data.get('direction', 'cw')  # 'cw' or 'ccw'
        
        # Get the image path
        image_path = image.original_path
        if not image_path:
            return jsonify({'success': False, 'error': 'No image path found'}), 404
        
        # Convert web path to file path
        file_path = image_path.replace('/static/', 'static/')
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'error': 'Image file not found'}), 404
        
        # Determine rotation angle
        if direction == 'cw':
            angle = -90  # Clockwise
        elif direction == 'ccw':
            angle = 90   # Counter-clockwise
        else:
            return jsonify({'success': False, 'error': 'Invalid direction. Use "cw" or "ccw"'}), 400
        
        # Open and rotate the image
        with PILImage.open(file_path) as img:
            rotated = img.rotate(angle, expand=True)
            # Save without EXIF data to avoid orientation issues
            rotated.save(file_path, quality=95)
        
        # Update image dimensions
        try:
            with PILImage.open(file_path) as img:
                image.width = img.width
                image.height = img.height
                db.session.commit()
        except Exception as e:
            print(f"Warning: Could not update image dimensions: {e}")
        
        return jsonify({
            'success': True,
            'image_url': image_path,
            'direction': direction,
            'message': f'Image rotated {direction.upper()} successfully'
        })
        
    except Exception as e:
        print(f"Error rotating image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/recrop_image/<int:image_id>', methods=['POST'])
def recrop_image(image_id):
    """Recrop an image by taking it back to the original if it was cropped by this program"""
    try:
        image = Image.query.get_or_404(image_id)
        
        # Check if this image has an original path (was cropped by this program)
        if not image.original_path:
            return jsonify({'success': False, 'error': 'This image was not cropped by this program. No original available.'}), 400
        
        # Check if original file exists
        original_file_path = image.original_path.replace('/static/', 'static/')
        if not os.path.exists(original_file_path):
            return jsonify({'success': False, 'error': 'Original image file not found.'}), 404
        
        # Move the original back to replace the cropped version
        cropped_file_path = image.cropped_path.replace('/static/', 'static/')
        
        # Backup the current cropped version
        backup_path = cropped_file_path + '.backup'
        try:
            shutil.copy2(cropped_file_path, backup_path)
        except Exception as e:
            print(f"Warning: Could not create backup: {e}")
        
        # Copy original to replace cropped
        try:
            shutil.copy2(original_file_path, cropped_file_path)
            
            # Update image dimensions
            with PILImage.open(cropped_file_path) as img:
                image.width = img.width
                image.height = img.height
                db.session.commit()
            
            return jsonify({
                'success': True,
                'message': 'Image recropped from original',
                'image_url': image.cropped_path
            })
            
        except Exception as e:
            # Restore from backup if copy failed
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, cropped_file_path)
                except Exception:
                    pass
            raise e
        
    except Exception as e:
        print(f"Error recropping image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/face_recognition')
@time_function("api.face_recognition")
def api_face_recognition():
    """API endpoint for face recognition on a specific image"""
    global face_recognition_db
    import json
    from face_recognition_module import FaceRecognitionDB
    
    with time_operation("api.face_recognition.parse_params"):
        # Get image path from query parameter
        image_path = request.args.get('image_path')
        auto_rotate = request.args.get('auto_rotate', 'false').lower() == 'true'
        
        if not image_path:
            return jsonify({'error': 'No image_path provided'}), 400
        
        # Initialize face recognition database if needed
        if face_recognition_db is None:
            try:
                face_recognition_db = FaceRecognitionDB(app)
                face_recognition_db.initialize_insightface()
            except Exception as e:
                print(f"Error initializing face recognition: {e}")
                return jsonify({'error': 'Face recognition system not available'}), 500
        
        # Convert web path to file path
        file_path = image_path.replace('/static/', 'static/')
        if not os.path.exists(file_path):
            return jsonify({'error': 'Image file not found'}), 404
    
    try:
        # Auto-rotate if requested
        if auto_rotate:
            try:
                with PILImage.open(file_path) as img:
                    # Check if auto-rotation would help
                    rotated_img = autorotate_image_pil(img)
                    if rotated_img != img:  # If rotation was applied
                        rotated_img.save(file_path)
                        print(f"Auto-rotated image: {file_path}")
            except Exception as e:
                print(f"Error during auto-rotation: {e}")
        
        # OPTIMIZATION: Detect faces once
        with time_operation("api.face_recognition.detect_faces"):
            face_boxes = face_recognition_db.detect_faces(file_path)
        print(f"Found {len(face_boxes)} faces in {file_path}")
        
        best_face = None
        best_confidence = 0
        best_name = ""
        
        if face_boxes:
            print(f"Testing {len(face_boxes)} faces for best recognition match...")
            
            # OPTIMIZATION: Process each face once
            with time_operation("api.face_recognition.process_faces", faces_count=len(face_boxes)):
                for i, face_box in enumerate(face_boxes):
                    print(f"Testing face {i+1}: {face_box}")
                    
                    with time_operation("api.face_recognition.crop_face", face_index=i):
                        # Crop the face with padding and ensure minimum size
                        with PILImage.open(file_path) as im:
                            # Add padding around the face (50% of face size)
                            padding_x = int(face_box['width'] * 0.5)
                            padding_y = int(face_box['height'] * 0.5)
                            
                            # Calculate crop box with padding
                            x1 = max(0, face_box['x'] - padding_x)
                            y1 = max(0, face_box['y'] - padding_y)
                            x2 = min(im.width, face_box['x'] + face_box['width'] + padding_x)
                            y2 = min(im.height, face_box['y'] + face_box['height'] + padding_y)
                            
                            face_crop = im.crop((x1, y1, x2, y2))
                            
                            # Ensure minimum size for face recognition (at least 112x112)
                            min_size = 112
                            if face_crop.width < min_size or face_crop.height < min_size:
                                # Resize to minimum size while maintaining aspect ratio
                                ratio = max(min_size / face_crop.width, min_size / face_crop.height)
                                new_width = int(face_crop.width * ratio)
                                new_height = int(face_crop.height * ratio)
                                face_crop = face_crop.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                            
                            temp_crop_path = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped', f'temp_face_crop_{i}.jpg')
                            face_crop.save(temp_crop_path)
                    
                    # OPTIMIZATION: Get face suggestions once per face
                    with time_operation("api.face_recognition.get_suggestions", face_index=i):
                        suggestions = face_recognition_db.get_face_suggestions_from_path(temp_crop_path, threshold=0.0, top_k=1)
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_crop_path)
                    except Exception:
                        pass
                    
                    if suggestions:
                        confidence = suggestions[0]['confidence']
                        name = suggestions[0]['name']
                        print(f"Face {i+1} matched: {name} (confidence: {confidence:.3f})")
                        
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_face = face_box
                            best_name = name
                    else:
                        print(f"Face {i+1}: No match found")
                    
                    if best_face:
                        print(f"Best match: {best_name} (confidence: {best_confidence:.3f})")
                    else:
                        # If no recognition matches, use the largest face
                        best_face = max(face_boxes, key=lambda b: b['width'] * b['height'])
                        best_name = ""
                        print("No recognition matches found, using largest face")
        
        return jsonify({
            'face_box': best_face,
            'predicted_name': best_name,
            'confidence': best_confidence if best_face else 0,
            'auto_rotated': auto_rotate
        })
        
    except Exception as e:
        print(f"Error in face recognition API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/undo_last', methods=['POST'])
def undo_last():
    """Undo the last name assignment: move the most recently named image back to cropped and clear its person assignment."""
    from sqlalchemy import desc
    # Find the most recently updated image in named_cropped
    last_named = Image.query.filter(Image.cropped_path.like('%/named_cropped/%')).order_by(desc(Image.id)).first()
    if not last_named:
        return jsonify({'success': False, 'error': 'No recently named image found.'}), 404
    # Move file from named_cropped to cropped
    src_path = last_named.cropped_path.replace('/static/', 'static/')
    cropped_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'cropped')
    os.makedirs(cropped_dir, exist_ok=True)
    dest_path = os.path.join(cropped_dir, os.path.basename(src_path))
    try:
        shutil.move(src_path, dest_path)
        last_named.cropped_path = dest_path.replace('static/', '/static/')
        # Remove person assignment
        last_named.people.clear()
        db.session.commit()
        return jsonify({'success': True, 'image_path': last_named.cropped_path})
    except Exception as e:
        print(f"Error undoing last assignment: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/face-matches')
def face_matches():
    """Display groups of similar faces"""
    # Load current state
    state = load_face_matching_state()
    return render_template('face_matches.html', state=state)

# Global variable to store the face matcher instance
_face_matcher_instance = None

def get_face_matcher():
    """Get or create the singleton face matcher instance"""
    global _face_matcher_instance
    if _face_matcher_instance is None:
        try:
            from standalone_face_matcher import HighPerformanceFaceMatcher
            _face_matcher_instance = HighPerformanceFaceMatcher()
        except ImportError as e:
            print(f"Could not import HighPerformanceFaceMatcher: {e}")
            _face_matcher_instance = None
    return _face_matcher_instance

@app.route('/api/face-matches/process', methods=['POST'])
def process_face_matches():
    """Process all images to extract faces and find matches"""
    try:
        # Reset state
        state = {
            'last_processed_group': 0,
            'total_groups': 0,
            'processing_complete': False,
            'last_update': datetime.now().isoformat()
        }
        save_face_matching_state(state)
        
        def process_background():
            try:
                matcher = get_face_matcher()
                if matcher:
                    success = matcher.run_full_processing()
                    if success:
                        print("Face processing completed successfully")
                        # Update state
                        state['processing_complete'] = True
                        state['last_update'] = datetime.now().isoformat()
                        save_face_matching_state(state)
                    else:
                        print("Face processing failed")
                else:
                    # Fallback to subprocess approach
                    import subprocess
                    import sys
                    
                    result = subprocess.run([
                        sys.executable, 'standalone_face_matcher.py'
                    ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
                    
                    if result.returncode == 0:
                        print("Face processing completed successfully")
                        state['processing_complete'] = True
                        state['last_update'] = datetime.now().isoformat()
                        save_face_matching_state(state)
                    else:
                        print(f"Face processing failed: {result.stderr}")
                
            except Exception as e:
                print(f"Error in background processing: {e}")
        
        # Start background processing
        thread = threading.Thread(target=process_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'message': 'Face processing started in background'})
        
    except Exception as e:
        print(f"Error starting face processing: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/face-matches/clear', methods=['POST'])
def clear_face_matches():
    """Clear all face matching data and restart"""
    try:
        success = clear_face_matching_data()
        if success:
            # Reset state
            state = {
                'last_processed_group': 0,
                'total_groups': 0,
                'processing_complete': False,
                'last_update': datetime.now().isoformat()
            }
            save_face_matching_state(state)
            return jsonify({'success': True, 'message': 'Face matching data cleared successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to clear data'}), 500
    except Exception as e:
        print(f"Error clearing face matches: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/face-matches/progress')
def get_face_matches_progress():
    """Get current progress of face processing"""
    try:
        matcher = get_face_matcher()
        if matcher:
            progress = matcher.get_progress()
            # Update state with current progress
            state = load_face_matching_state()
            if progress.get('stats', {}).get('face_groups', 0) > 0:
                state['total_groups'] = progress['stats']['face_groups']
                state['last_update'] = datetime.now().isoformat()
                save_face_matching_state(state)
            return jsonify({'success': True, 'progress': progress})
        else:
            return jsonify({'success': True, 'progress': {
                'overall': 0.0,
                'fitpexport': {'processed': 0, 'total': 0, 'percent': 0.0},
                'visage': {'processed': 0, 'total': 0, 'percent': 0.0},
                'matching': {'processed': 0, 'total': 0, 'percent': 0.0},
                'stats': {'total_images': 0, 'processed_images': 0, 'faces_extracted': 0, 'face_groups': 0},
                'current_activity': 'Face matcher not available',
                'status': 'Not Available'
            }})
    except Exception as e:
        print(f"Error getting progress: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/face-matches/groups')
def get_face_match_groups():
    """Get groups of similar faces with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 1, type=int)  # Show one group at a time
        group_id = request.args.get('group_id', None, type=int)
        
        # Try to use the high-performance matcher first
        matcher = get_face_matcher()
        if matcher:
            result = matcher.get_similar_face_groups(min_group_size=2)
            groups = result['groups']
        else:
            # Fall back to reading from database directly with dHash support
            import sqlite3
            from pathlib import Path
            
            db_path = Path("face_crops/face_crops.db")
            if not db_path.exists():
                return jsonify({'success': True, 'groups': [], 'total_groups': 0, 'current_page': page})
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get total count of groups using face_groups table
            cursor.execute("SELECT COUNT(*) FROM face_groups")
            total_groups = cursor.fetchone()[0] or 0
            
            # Get groups with pagination
            offset = (page - 1) * per_page
            cursor.execute("""
                SELECT group_id, crop_ids, similarity_score 
                FROM face_groups 
                ORDER BY similarity_score DESC
                LIMIT ? OFFSET ?
            """, (per_page, offset))
            
            groups_data = cursor.fetchall()
            conn.close()
            
            # Create groups from database
            groups = []
            for group_id, crop_ids_str, similarity_score in groups_data:
                crop_ids = [int(id) for id in crop_ids_str.split(',')]
                
                # Get face details for this group
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                face_details = []
                for crop_id in crop_ids:
                    cursor.execute("""
                        SELECT crop_path, original_path, source_type, bbox_x, bbox_y, bbox_w, bbox_h, confidence
                        FROM face_crops WHERE id = ?
                    """, (crop_id,))
                    face_row = cursor.fetchone()
                    if face_row:
                        face_details.append({
                            'crop_path': face_row[0],
                            'original_path': face_row[1],
                            'source_type': face_row[2],
                            'bbox': face_row[3:7],
                            'confidence': face_row[7],
                            'similarity': similarity_score
                        })
                conn.close()
                
                if len(face_details) >= 2:  # Only show groups with 2+ faces
                    groups.append({
                        'group_id': group_id,
                        'faces': face_details,
                        'similarity_score': similarity_score,
                        'count': len(face_details)
                    })
        
        # Convert paths to web URLs
        for group in groups:
            for face in group['faces']:
                # Convert crop path to web URL
                if face['crop_path'].startswith('face_crops/'):
                    face['crop_url'] = f"/static/{face['crop_path']}"
                else:
                    face['crop_url'] = f"/static/{face['crop_path']}"
                
                # Convert original path to web URL
                if face['original_path'].startswith('static/'):
                    face['original_url'] = f"/{face['original_path']}"
                elif face['original_path'].startswith('fitpexport/'):
                    face['original_url'] = f"/static/{face['original_path']}"
                else:
                    face['original_url'] = f"/static/{face['original_path']}"
        
        # Update state
        state = load_face_matching_state()
        state['total_groups'] = len(groups)
        state['last_processed_group'] = page
        state['last_update'] = datetime.now().isoformat()
        save_face_matching_state(state)
        
        return jsonify({
            'success': True, 
            'groups': groups, 
            'total_groups': len(groups), 
            'current_page': page,
            'has_next': page < len(groups),
            'has_prev': page > 1
        })
        
    except Exception as e:
        print(f"Error getting face match groups: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/face-matches/state')
def get_face_matches_state():
    """Get current face matching state"""
    try:
        state = load_face_matching_state()
        return jsonify({'success': True, 'state': state})
    except Exception as e:
        print(f"Error getting face matching state: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/images_paginated')
def api_images_paginated():
    """Keyset/cursor paginated API endpoint for images, only including images whose file exists on disk"""
    page_size = int(request.args.get('page_size', 50))
    favorites_only = request.args.get('favorites_only', 'false') == 'true'
    person_filter = request.args.get('person_filter', '')
    sort_by = request.args.get('sort_by', 'id-desc')
    last_id = request.args.get('last_id', type=int)
    last_name = request.args.get('last_name', type=str)
    last_created_at = request.args.get('last_created_at', type=str)
    last_size = request.args.get('last_size', type=int)
    last_fav = request.args.get('last_fav', type=int)

    query = Image.query

    # Filter: favorites only
    if favorites_only:
        query = query.filter(Image.is_favorite == True)

    # Filter: by person
    if person_filter:
        if person_filter == 'unidentified':
            query = query.filter(~Image.people.any())
        else:
            try:
                person_id = int(person_filter)
                query = query.filter(Image.people.any(Person.id == person_id))
            except Exception:
                pass

    # Sorting and keyset pagination (unchanged)
    if sort_by == 'id-desc':
        query = query.order_by(Image.id.desc())
        if last_id is not None:
            query = query.filter(Image.id < last_id)
    elif sort_by == 'id-asc':
        query = query.order_by(Image.id.asc())
        if last_id is not None:
            query = query.filter(Image.id > last_id)
    elif sort_by == 'name-asc':
        query = query.join(Image.people, isouter=True).order_by(Person.name.asc().nullslast(), Image.id.asc())
        if last_name is not None and last_id is not None:
            query = query.filter(
                (Person.name > last_name) |
                ((Person.name == last_name) & (Image.id > last_id))
            )
    elif sort_by == 'name-desc':
        query = query.join(Image.people, isouter=True).order_by(Person.name.desc().nullslast(), Image.id.desc())
        if last_name is not None and last_id is not None:
            query = query.filter(
                (Person.name < last_name) |
                ((Person.name == last_name) & (Image.id < last_id))
            )
    elif sort_by == 'date-newest':
        query = query.order_by(Image.created_at.desc(), Image.id.desc())
        if last_created_at is not None and last_id is not None:
            query = query.filter(
                (Image.created_at < last_created_at) |
                ((Image.created_at == last_created_at) & (Image.id < last_id))
            )
    elif sort_by == 'date-oldest':
        query = query.order_by(Image.created_at.asc(), Image.id.asc())
        if last_created_at is not None and last_id is not None:
            query = query.filter(
                (Image.created_at > last_created_at) |
                ((Image.created_at == last_created_at) & (Image.id > last_id))
            )
    elif sort_by == 'size-large':
        query = query.order_by((Image.width * Image.height).desc().nullslast(), Image.id.desc())
        if last_size is not None and last_id is not None:
            query = query.filter(
                ((Image.width * Image.height) < last_size) |
                (((Image.width * Image.height) == last_size) & (Image.id < last_id))
            )
    elif sort_by == 'size-small':
        query = query.order_by((Image.width * Image.height).asc().nullslast(), Image.id.asc())
        if last_size is not None and last_id is not None:
            query = query.filter(
                ((Image.width * Image.height) > last_size) |
                (((Image.width * Image.height) == last_size) & (Image.id > last_id))
            )
    elif sort_by == 'favorites':
        query = query.order_by(Image.is_favorite.desc(), Image.id.desc())
        if last_fav is not None and last_id is not None:
            query = query.filter(
                (Image.is_favorite < last_fav) |
                ((Image.is_favorite == last_fav) & (Image.id < last_id))
            )
    elif sort_by == 'random':
        # For random sorting, we need to get all images first and then shuffle
        # This is less efficient but necessary for true randomness
        pass  # We'll handle this after the query
    else:
        query = query.order_by(Image.id.desc())
        if last_id is not None:
            query = query.filter(Image.id < last_id)

    # Get more than page_size in case some are missing files
    if sort_by == 'random':
        # For random sorting, get all images and shuffle them
        candidates = query.all()
        def file_exists(img):
            path = img.original_path
            if not path:
                return False
            file_path = path.replace('/static/', 'static/')
            return os.path.exists(file_path)
        filtered_images = [img for img in candidates if file_exists(img)]
        
        # Shuffle the filtered images
        import random
        random.shuffle(filtered_images)
        
        # Take the first page_size images
        images = filtered_images[:page_size]
    else:
        # For other sorting methods, use keyset pagination
        candidates = query.limit(page_size * 3).all()
        def file_exists(img):
            path = img.original_path
            if not path:
                return False
            file_path = path.replace('/static/', 'static/')
            return os.path.exists(file_path)
        filtered_images = [img for img in candidates if file_exists(img)]
        images = filtered_images[:page_size]

    # For total, count all images matching file existence (slow for large sets)
    all_candidates = Image.query.order_by(Image.id.desc()).all()
    total = sum(1 for img in all_candidates if file_exists(img))

    # Prepare cursor fields for the last image
    last_cursor = {}
    if images and sort_by != 'random':  # Don't use cursors for random sorting
        last_img = images[-1]
        last_cursor['last_id'] = last_img.id
        if sort_by.startswith('name'):
            last_cursor['last_name'] = last_img.people[0].name if last_img.people else ''
        if sort_by.startswith('date'):
            last_cursor['last_created_at'] = last_img.created_at.isoformat() if last_img.created_at else ''
        if sort_by.startswith('size'):
            last_cursor['last_size'] = (last_img.width or 0) * (last_img.height or 0)
        if sort_by == 'favorites':
            last_cursor['last_fav'] = 1 if last_img.is_favorite else 0

    return jsonify({
        'images': [img.to_dict() for img in images],
        'total': total,
        'page_size': page_size,
        **last_cursor
    })

@app.route('/import_folder', methods=['GET', 'POST'])
def import_folder():
    """Import a folder of images, always tag as uncropped, and convert .webp to .jpg"""
    import glob
    import json
    from models import Image, Person
    from flask import request, flash, redirect, url_for, render_template
    import os
    if request.method == 'POST':
        files = request.files.getlist('images')
        uploaded_count = 0
        errors = []
        for file in files:
            if file and allowed_file(file.filename):
                try:
                    file.seek(0, 2)
                    file_size = file.tell()
                    file.seek(0)
                    if file_size > MAX_FILE_SIZE:
                        errors.append(f"{file.filename} is too large (max 16MB)")
                        continue
                    ext = os.path.splitext(file.filename)[1].lower()
                    if ext == '.webp':
                        img = PILImage.open(file.stream).convert('RGB')
                        filename = secure_filename(os.path.splitext(file.filename)[0] + '.jpg')
                        base_name = f"{uuid.uuid4().hex}_{filename}"
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], base_name)
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        img.save(file_path, 'JPEG')
                    else:
                        filename = secure_filename(file.filename)
                        base_name = f"{uuid.uuid4().hex}_{filename}"
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], base_name)
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        file.save(file_path)
                    tags = ['uncropped']
                    new_image = Image(
                        original_path=f'/static/uploads/{base_name}',
                        filename=filename,
                        is_favorite=False,
                        tags=json.dumps(tags),
                        recognition_suggestions=json.dumps([])
                    )
                    # Set width and height
                    try:
                        from PIL import Image as PILImage
                        with PILImage.open(os.path.join(app.config['UPLOAD_FOLDER'], 'imported', base_name)) as im:
                            new_image.width = im.width
                            new_image.height = im.height
                    except Exception as e:
                        print(f"Error setting image dimensions: {e}")
                    db.session.add(new_image)
                    db.session.commit()
                    uploaded_count += 1
                except Exception as e:
                    errors.append(f"Error processing {file.filename}: {str(e)}")
            else:
                errors.append(f"{file.filename} is not a valid image file")
        if uploaded_count > 0:
            flash(f'Successfully imported {uploaded_count} image(s)', 'success')
            uncropped_count = Image.query.filter(Image.tags.contains('uncropped')).count()
            if uncropped_count > 0:
                return redirect(url_for('cropper'))
        if errors:
            flash('Errors: ' + ', '.join(errors), 'error')
        return redirect(url_for('import_workflow'))
    return render_template('import_folder.html')

@app.route('/import_zip', methods=['GET', 'POST'])
def import_zip():
    """Import images from a zip file"""
    import zipfile
    import tempfile
    import shutil
    import json
    from models import Image, Person
    from flask import request, flash, redirect, url_for, render_template
    import os
    
    if request.method == 'POST':
        if 'zip_file' not in request.files:
            flash('No zip file selected', 'error')
            return redirect(url_for('import_zip'))
        
        zip_file = request.files['zip_file']
        if zip_file.filename == '':
            flash('No zip file selected', 'error')
            return redirect(url_for('import_zip'))
        
        if not is_zip_file(zip_file.filename):
            flash('Please select a valid zip file', 'error')
            return redirect(url_for('import_zip'))
        
        # Check file size
        zip_file.seek(0, 2)
        file_size = zip_file.tell()
        zip_file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            flash('Zip file is too large (max 16MB)', 'error')
            return redirect(url_for('import_zip'))
        
        uploaded_count = 0
        errors = []
        
        try:
            # Create a temporary directory to extract files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save zip file temporarily
                temp_zip_path = os.path.join(temp_dir, 'upload.zip')
                zip_file.save(temp_zip_path)
                
                # Extract zip file
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    # Check for zip bomb (too many files)
                    file_list = zip_ref.namelist()
                    if len(file_list) > 1000:  # Limit to 1000 files
                        flash('Zip file contains too many files (max 1000)', 'error')
                        return redirect(url_for('import_zip'))
                    
                    # Extract files
                    zip_ref.extractall(temp_dir)
                
                # Process extracted files
                imported_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'imported')
                os.makedirs(imported_folder, exist_ok=True)
                
                # Walk through extracted files
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_ext = os.path.splitext(file)[1].lower()
                        
                        # Check if it's an image file
                        if file_ext in {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}:
                            try:
                                # Generate unique filename
                                filename = secure_filename(file)
                                base_name = f"{uuid.uuid4().hex}_{filename}"
                                target_path = os.path.join(imported_folder, base_name)
                                
                                # Process the image
                                if file_ext == '.webp':
                                    # Convert webp to jpg
                                    img = PILImage.open(file_path).convert('RGB')
                                    filename = os.path.splitext(filename)[0] + '.jpg'
                                    base_name = f"{uuid.uuid4().hex}_{filename}"
                                    target_path = os.path.join(imported_folder, base_name)
                                    img.save(target_path, 'JPEG')
                                else:
                                    # Copy file as is
                                    shutil.copy2(file_path, target_path)
                                
                                # Create database entry
                                tags = ['uncropped']
                                new_image = Image(
                                    original_path=f'/static/uploads/imported/{base_name}',
                                    filename=filename,
                                    is_cropped=False,
                                    is_favorite=False,
                                    tags=json.dumps(tags),
                                    recognition_suggestions=json.dumps([])
                                )
                                
                                # Set width and height
                                try:
                                    with PILImage.open(target_path) as im:
                                        new_image.width = im.width
                                        new_image.height = im.height
                                except Exception as e:
                                    print(f"Error setting image dimensions: {e}")
                                
                                db.session.add(new_image)
                                db.session.commit()
                                uploaded_count += 1
                                
                            except Exception as e:
                                errors.append(f"Error processing {file}: {str(e)}")
                
                if uploaded_count > 0:
                    flash(f'Successfully imported {uploaded_count} image(s) from zip file', 'success')
                    uncropped_count = Image.query.filter(Image.tags.contains('uncropped')).count()
                    if uncropped_count > 0:
                        return redirect(url_for('cropper'))
                
                if errors:
                    flash('Errors: ' + ', '.join(errors), 'error')
                    
        except zipfile.BadZipFile:
            flash('Invalid zip file format', 'error')
        except Exception as e:
            flash(f'Error processing zip file: {str(e)}', 'error')
        
        return redirect(url_for('import_workflow'))
    
    return render_template('import_zip.html')

@app.route('/api/images/cleanup', methods=['POST'])
def cleanup_images():
    try:
        # Find the 'Unidentified' person
        unidentified = Person.query.filter_by(name='Unidentified').first()
        # Query all images
        images = Image.query.options(joinedload(Image.people)).all()
        to_delete = []
        for img in images:
            people = img.people
            if not people:
                to_delete.append(img)
            elif len(people) == 1 and unidentified and people[0].id == unidentified.id:
                to_delete.append(img)
        deleted_count = 0
        for img in to_delete:
            # Remove image file from disk if exists
            if img.cropped_path and os.path.exists(img.cropped_path):
                try:
                    os.remove(img.cropped_path)
                except Exception:
                    pass
            if img.original_path and os.path.exists(img.original_path):
                try:
                    os.remove(img.original_path)
                except Exception:
                    pass
            db.session.delete(img)
            deleted_count += 1
        db.session.commit()
        return jsonify({'success': True, 'deleted': deleted_count})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/cropper')
def cropper():
    """Show the uncropped image with the lowest ID for processing."""
    from models import Image
    import json
    image = Image.query.filter(Image.tags.contains('uncropped')).order_by(Image.id.desc()).first()
    
    # If no uncropped images, check if there are unnamed images and redirect to namer
    if not image:
        unnamed_count = Image.query.filter(~Image.people.any()).count()
        if unnamed_count > 0:
            return redirect(url_for('namer'))
        else:
            # No uncropped or unnamed images, redirect to all images
            return redirect(url_for('all_images'))
    
    image_data = image.to_dict()
    # Find the next uncropped image (by ID)
    next_image_id = None
    if image:
        next_image = Image.query.filter(Image.tags.contains('uncropped'), Image.id > image.id).order_by(Image.id.desc()).first()
        if next_image:
            next_image_id = next_image.id
        else:
            # Wrap to first if at end
            first_image = Image.query.filter(Image.tags.contains('uncropped')).order_by(Image.id.desc()).first()
            if first_image and first_image.id != image.id:
                next_image_id = first_image.id
    return render_template('cropper.html', image=image_data, next_image_id=next_image_id, current_sort='id-desc')

@app.route('/cropper/<int:image_id>')
def cropper_image(image_id):
    """Show a specific uncropped image by ID, and provide next uncropped image navigation."""
    from models import Image
    import json
    image = Image.query.get_or_404(image_id)
    if 'uncropped' not in (json.loads(image.tags) if image.tags else []):
        # If not uncropped, redirect to /cropper
        from flask import redirect, url_for
        return redirect(url_for('cropper'))
    image_data = image.to_dict()
    # Find the next uncropped image (by ID)
    next_image = Image.query.filter(Image.tags.contains('uncropped'), Image.id > image.id).order_by(Image.id.desc()).first()
    if next_image:
        next_image_id = next_image.id
    else:
        # Wrap to first if at end
        first_image = Image.query.filter(Image.tags.contains('uncropped')).order_by(Image.id.desc()).first()
        if first_image and first_image.id != image.id:
            next_image_id = first_image.id
        else:
            next_image_id = None
    return render_template('cropper.html', image=image_data, next_image_id=next_image_id, current_sort='id-desc')

@app.route('/api/cropper/yolo', methods=['POST'])
def api_cropper_yolo():
    from yolo_detector import get_yolo_detector
    import base64
    import io
    data = request.get_json()
    image_id = data.get('image_id')
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'success': False, 'error': 'Image not found'})
    file_path = image.original_path.replace('/static/', 'static/')
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'})
    
    try:
        # First check if the image can be opened with PIL
        with PILImage.open(file_path) as img:
            # Verify the image is valid by trying to access its size
            img.verify()
        
        # Reopen for actual processing
        with PILImage.open(file_path) as img:
            detector = get_yolo_detector()
            result = detector.detect_persons(file_path)
            crops = []
            for box in result:
                coords = box['bbox']
                crop_img = img.crop((coords[0], coords[1], coords[2], coords[3]))
                buffered = io.BytesIO()
                crop_img.save(buffered, format="JPEG")
                img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                crops.append({'box': coords, 'base64': img_b64})
            return jsonify({'success': True, 'crops': crops})
            
    except PILImage.UnidentifiedImageError:
        print(f"Corrupted or invalid image file: {file_path}")
        return jsonify({'success': False, 'error': 'Image file is corrupted or invalid'})
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return jsonify({'success': False, 'error': f'Error processing image: {str(e)}'})

@app.route('/api/cropper/save', methods=['POST'])
def api_cropper_save():
    import json
    import shutil
    data = request.get_json()
    image_id = data.get('image_id')
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'success': False, 'error': 'Image not found'})
    file_path = image.original_path.replace('/static/', 'static/')
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'})
    crop_type = data.get('type')
    crop_idx = data.get('crop_idx')
    def remove_uncropped_and_names(img, keep_names=False):
        tags = json.loads(img.tags) if img.tags else []
        if 'uncropped' in tags:
            tags.remove('uncropped')
        if not keep_names:
            img.people.clear()
            if 'named' in tags:
                tags.remove('named')
        img.tags = json.dumps(tags)
    def backup_original(img):
        orig_path = img.original_path.replace('/static/', 'static/')
        if os.path.exists(orig_path):
            backup_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'original_backups')
            os.makedirs(backup_dir, exist_ok=True)
            backup_name = f"{img.id}_{img.filename}_originalbackup.jpg"
            backup_path = os.path.join(backup_dir, backup_name)
            shutil.copy2(orig_path, backup_path)
    def remove_and_hide_original(img):
        orig_path = img.original_path.replace('/static/', 'static/')
        if os.path.exists(orig_path):
            removed_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'removed')
            os.makedirs(removed_dir, exist_ok=True)
            removed_name = f"{img.id}_{img.filename}"
            removed_path = os.path.join(removed_dir, removed_name)
            shutil.move(orig_path, removed_path)
        db.session.delete(img)
    from yolo_detector import get_yolo_detector
    detector = get_yolo_detector()
    result = detector.detect_persons(file_path)
    next_image = Image.query.filter(Image.tags.contains('uncropped'), Image.id > image.id).order_by(Image.id.asc()).first()
    
    # Helper function to determine redirect when no more uncropped images
    def get_redirect_url():
        if next_image:
            return None  # No redirect needed
        # Check if there are unnamed images
        unnamed_count = Image.query.filter(~Image.people.any()).count()
        if unnamed_count > 0:
            return url_for('namer')
        else:
            return url_for('all_images')
    
    if crop_type == 'crop':
        with PILImage.open(file_path) as img_pil:
            idx = int(crop_idx)
            if idx < 0 or idx >= len(result):
                return jsonify({'success': False, 'error': 'Invalid crop index'})
            coords = result[idx]['bbox']
            crop_img = img_pil.crop((coords[0], coords[1], coords[2], coords[3]))
            cropped_filename = f"crop_{uuid.uuid4().hex}_{image.filename}"
            cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
            crop_img.save(cropped_path)
            web_path = f'/static/uploads/{cropped_filename}'
            new_image = Image(
                original_path=web_path,
                filename=cropped_filename,
                is_favorite=False,
                tags=json.dumps([]),
                recognition_suggestions=json.dumps([])
            )
            # Set width and height
            try:
                with PILImage.open(cropped_path) as im:
                    new_image.width = im.width
                    new_image.height = im.height
            except Exception as e:
                print(f"Error setting image dimensions: {e}")
            
            # CRITICAL: Ensure the new crop image has NO people assigned
            # This prevents any automatic assignment from the original image
            new_image.people.clear()
            
            db.session.add(new_image)
            db.session.flush()  # Assigns new_image.id
            old_id = image.id
            db.session.delete(image)
            db.session.flush()
            db.session.execute(text(f"UPDATE images SET id = {old_id} WHERE id = {new_image.id}"))
            db.session.commit()
        next_id = next_image.id if next_image else None
        saved_crop = {'id': old_id, 'image_url': web_path, 'filename': cropped_filename}
        redirect_url = get_redirect_url()
        response_data = {'success': True, 'next_image_id': next_id, 'saved_crop': saved_crop}
        if redirect_url:
            response_data['redirect_to'] = redirect_url
        return jsonify(response_data)
    elif crop_type == 'original':
        backup_original(image)
        remove_uncropped_and_names(image)
        db.session.commit()
        next_id = next_image.id if next_image else None
        redirect_url = get_redirect_url()
        response_data = {'success': True, 'next_image_id': next_id}
        if redirect_url:
            response_data['redirect_to'] = redirect_url
        return jsonify(response_data)
    elif crop_type == 'manual_crop':
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        width = int(data.get('width', 0))
        height = int(data.get('height', 0))
        if width <= 0 or height <= 0:
            return jsonify({'success': False, 'error': 'Invalid crop parameters'})
        with PILImage.open(file_path) as img_pil:
            crop_box = (x, y, x + width, y + height)
            cropped_img = img_pil.crop(crop_box)
            cropped_filename = f"manualcrop_{uuid.uuid4().hex}_{image.filename}"
            cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
            cropped_img.save(cropped_path)
            web_path = f'/static/uploads/{cropped_filename}'
            new_image = Image(
                original_path=web_path,
                filename=cropped_filename,
                is_favorite=False,
                tags=json.dumps([]),
                recognition_suggestions=json.dumps([])
            )
            # Set width and height
            try:
                with PILImage.open(cropped_path) as im:
                    new_image.width = im.width
                    new_image.height = im.height
            except Exception as e:
                print(f"Error setting image dimensions: {e}")
            db.session.add(new_image)
            db.session.flush()
            old_id = image.id
            db.session.delete(image)
            db.session.flush()
            db.session.execute(text(f"UPDATE images SET id = {old_id} WHERE id = {new_image.id}"))
            db.session.commit()
        next_id = next_image.id if next_image else None
        saved_crop = {'id': old_id, 'image_url': web_path, 'filename': cropped_filename}
        redirect_url = get_redirect_url()
        response_data = {'success': True, 'next_image_id': next_id, 'saved_crop': saved_crop}
        if redirect_url:
            response_data['redirect_to'] = redirect_url
        return jsonify(response_data)
    else:
        return jsonify({'success': False, 'error': 'Invalid type'})

@app.route('/api/cropper/save_multi', methods=['POST'])
def api_cropper_save_multi():
    import json
    import shutil
    data = request.get_json()
    image_id = data.get('image_id')
    crops = data.get('crops', [])
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'success': False, 'error': 'Image not found'})
    file_path = image.original_path.replace('/static/', 'static/')
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'})
    from yolo_detector import get_yolo_detector
    detector = get_yolo_detector()
    result = detector.detect_persons(file_path)
    def remove_uncropped_and_names(img, keep_names=False):
        tags = json.loads(img.tags) if img.tags else []
        if 'uncropped' in tags:
            tags.remove('uncropped')
        if not keep_names:
            img.people.clear()
            if 'named' in tags:
                tags.remove('named')
        img.tags = json.dumps(tags)
    def backup_original(img):
        orig_path = img.original_path.replace('/static/', 'static/')
        if os.path.exists(orig_path):
            backup_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'original_backups')
            os.makedirs(backup_dir, exist_ok=True)
            backup_name = f"{img.id}_{img.filename}_originalbackup.jpg"
            backup_path = os.path.join(backup_dir, backup_name)
            shutil.copy2(orig_path, backup_path)
    def remove_and_hide_original(img):
        orig_path = img.original_path.replace('/static/', 'static/')
        if os.path.exists(orig_path):
            removed_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'removed')
            os.makedirs(removed_dir, exist_ok=True)
            removed_name = f"{img.id}_{img.filename}"
            removed_path = os.path.join(removed_dir, removed_name)
            shutil.move(orig_path, removed_path)
        db.session.delete(img)
    original_included = any(c.get('type') == 'original' for c in crops)
    new_crops = []
    with PILImage.open(file_path) as img_pil:
        for c in crops:
            if c.get('type') == 'original':
                backup_original(image)
                remove_uncropped_and_names(image, keep_names=True)
            elif c.get('type') == 'crop':
                idx = int(c.get('crop_idx'))
                if idx < 0 or idx >= len(result):
                    continue
                coords = result[idx]['bbox']
                crop_img = img_pil.crop((coords[0], coords[1], coords[2], coords[3]))
                cropped_filename = f"crop_{uuid.uuid4().hex}_{image.filename}"
                cropped_path = os.path.join(app.config['UPLOAD_FOLDER'], cropped_filename)
                crop_img.save(cropped_path)
                web_path = f'/static/uploads/{cropped_filename}'
                new_image = Image(
                    original_path=web_path,
                    filename=cropped_filename,
                    is_favorite=False,
                    tags=json.dumps([]),
                    recognition_suggestions=json.dumps([])
                )
                # Set width and height
                try:
                    with PILImage.open(cropped_path) as im:
                        new_image.width = im.width
                        new_image.height = im.height
                except Exception as e:
                    print(f"Error setting image dimensions: {e}")
                
                # CRITICAL: Ensure the new crop image has NO people assigned
                # This prevents any automatic assignment from the original image
                new_image.people.clear()
                
                db.session.add(new_image)
                db.session.flush()
                
                # CRITICAL FIX: Clear people assignment immediately after flush
                # The assignment happens during flush, so we need to clear it right after
                # Check if people were assigned during flush
                if new_image.people:
                    new_image.people.clear()
                    # Use ORM method instead of raw SQL to avoid StaleDataError
                    db.session.flush()
                
                new_crops.append(new_image)
    # Prepare dicts BEFORE deleting or reassigning
    def crop_to_dict(crop):
        return {
            'id': crop.id,
            'image_url': crop.original_path or f'/static/uploads/{crop.filename}',
            'filename': crop.filename
        }
    saved_crops = [crop_to_dict(c) for c in new_crops] if new_crops else []

    # Now do deletion/reassignment
    if not original_included and new_crops:
        old_id = image.id
        first_crop_id = new_crops[0].id
        
        # Store the original image's people associations before deletion
        original_people = list(image.people)
        
        # Delete the original image
        db.session.delete(image)
        db.session.flush()
        
        # Reassign the first crop's ID to the original image's ID
        db.session.execute(text(f"UPDATE images SET id = {old_id} WHERE id = {first_crop_id}"))
        
        # CRITICAL FIX: Clear people associations from the reassigned crop
        # The ID reassignment can cause the crop to inherit the original image's people
        # Use ORM method instead of raw SQL to avoid StaleDataError
        reassigned_image = Image.query.get(old_id)
        if reassigned_image:
            reassigned_image.people.clear()
        
        # Remove the first crop from the new_crops list and expunge it from session
        # to prevent SQLAlchemy from trying to update it
        reassigned_crop = new_crops.pop(0)
        db.session.expunge(reassigned_crop)  # Remove from session tracking
        
        # Update the saved_crops list to reflect the ID change
        saved_crops = [{
            'id': old_id,
            'image_url': reassigned_crop.original_path or f'/static/uploads/{reassigned_crop.filename}',
            'filename': reassigned_crop.filename
        }] + [crop_to_dict(crop) for crop in new_crops]
        
    elif not original_included:
        remove_and_hide_original(image)
    
    # CRITICAL FIX: Ensure ALL remaining crops stay unnamed
    # Clear people associations from ALL crops to prevent automatic assignment
    for crop in new_crops:
        crop.people.clear()
    
    db.session.commit()
    next_image = Image.query.filter(Image.tags.contains('uncropped'), Image.id > image.id).order_by(Image.id.asc()).first()
    next_id = next_image.id if next_image else None
    
    # Check if we should redirect when no more uncropped images
    redirect_url = None
    if not next_image:
        # Check if there are unnamed images
        unnamed_count = Image.query.filter(~Image.people.any()).count()
        if unnamed_count > 0:
            redirect_url = url_for('namer')
        else:
            redirect_url = url_for('all_images')
    
    response_data = {'success': True, 'next_image_id': next_id, 'saved_crops': saved_crops}
    if redirect_url:
        response_data['redirect_to'] = redirect_url
    return jsonify(response_data)

@app.route('/api/cropper/skip', methods=['POST'])
def api_cropper_skip():
    import json
    data = request.get_json()
    image_id = data.get('image_id')
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'success': False, 'error': 'Image not found'})
    # Do NOT change any tags or files, just return the next uncropped image
    next_image = Image.query.filter(Image.tags.contains('uncropped'), Image.id > image.id).order_by(Image.id.asc()).first()
    next_id = next_image.id if next_image else None
    
    # Check if we should redirect when no more uncropped images
    redirect_url = None
    if not next_image:
        # Check if there are unnamed images
        unnamed_count = Image.query.filter(~Image.people.any()).count()
        if unnamed_count > 0:
            redirect_url = url_for('namer')
        else:
            redirect_url = url_for('all_images')
    
    response_data = {'success': True, 'next_image_id': next_id}
    if redirect_url:
        response_data['redirect_to'] = redirect_url
    return jsonify(response_data)

@app.route('/api/set_uncropped/<int:image_id>', methods=['POST'])
def set_image_uncropped(image_id):
    """Set the 'uncropped' tag on the image"""
    import json
    image = Image.query.get_or_404(image_id)
    tags = json.loads(image.tags) if image.tags else []
    if 'uncropped' not in tags:
        tags.append('uncropped')
        image.tags = json.dumps(tags)
        db.session.commit()
    return jsonify({'success': True, 'tags': tags})

@app.route('/api/cropper/image_data')
def api_cropper_image_data():
    from models import Image
    import json
    image_id = request.args.get('image_id', type=int)
    if image_id:
        image = Image.query.get(image_id)
        if not image or 'uncropped' not in (json.loads(image.tags) if image.tags else []):
            # If not uncropped, return first uncropped
            image = Image.query.filter(Image.tags.contains('uncropped')).order_by(Image.id.asc()).first()
    else:
        image = Image.query.filter(Image.tags.contains('uncropped')).order_by(Image.id.asc()).first()
    if not image:
        return jsonify({'success': True, 'image': None, 'next_image_id': None, 'next_images': [], 'uncropped_count': 0})
    image_data = image.to_dict()
    # Find the next uncropped image (by ID)
    next_image = Image.query.filter(Image.tags.contains('uncropped'), Image.id > image.id).order_by(Image.id.asc()).first()
    if next_image:
        next_image_id = next_image.id
    else:
        # Wrap to first if at end
        first_image = Image.query.filter(Image.tags.contains('uncropped')).order_by(Image.id.asc()).first()
        if first_image and first_image.id != image.id:
            next_image_id = first_image.id
        else:
            next_image_id = None
    # Get next 10 uncropped images after current
    next_images_query = Image.query.filter(Image.tags.contains('uncropped'), Image.id > image.id).order_by(Image.id.asc()).limit(10).all()
    next_images = [
        {
            'id': img.id,
            'image_url': img.original_path or f'/static/uploads/{img.filename}',
            'filename': img.filename
        }
        for img in next_images_query
    ]
    uncropped_count = Image.query.filter(Image.tags.contains('uncropped')).count()
    return jsonify({'success': True, 'image': image_data, 'next_image_id': next_image_id, 'next_images': next_images, 'uncropped_count': uncropped_count})

@app.route('/api/remove_person_from_image/<int:image_id>/<int:person_id>', methods=['POST'])
def remove_person_from_image(image_id, person_id):
    import json
    image = Image.query.get(image_id)
    if not image:
        return jsonify({'success': False, 'error': 'Image not found'})
    person = None
    for p in image.people:
        if p.id == person_id:
            person = p
            break
    if not person:
        return jsonify({'success': False, 'error': 'Person not found on image'})
    image.people.remove(person)
    # Remove 'named' tag if no people remain
    tags = json.loads(image.tags) if image.tags else []
    if not image.people and 'named' in tags:
        tags.remove('named')
    image.tags = json.dumps(tags)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/namer-selection')
def namer_selection():
    """Show the namer selection page with photo and video options."""
    from models import Image, Video
    
    # Get photo stats
    photo_stats = {
        'unnamed_count': Image.query.filter(~Image.people.any()).count()
    }
    
    # Get video stats (videos without people associated)
    video_stats = {
        'unnamed_count': Video.query.filter(~Video.people.any()).count()
    }
    
    return render_template('namer_selection.html', 
                         photo_stats=photo_stats, 
                         video_stats=video_stats)

@app.route('/test-video-face-detection')
def test_video_face_detection():
    """Test face detection on video thumbnails."""
    from models import Video
    import os
    
    # Get a random video with a thumbnail
    video = Video.query.filter(Video.thumbnail_path.isnot(None)).first()
    
    if not video:
        return "No videos with thumbnails found"
    
    thumbnail_path = video.thumbnail_path.replace('/static/', 'static/')
    
    if not os.path.exists(thumbnail_path):
        return f"Thumbnail not found: {thumbnail_path}"
    
    try:
        # Test face detection
        faces = face_db.detect_faces(thumbnail_path)
        suggestions = face_db.get_suggestions_for_faces(faces, thumbnail_path) if faces else []
        
        result = f"""
        Video ID: {video.id}
        Thumbnail: {thumbnail_path}
        Faces detected: {len(faces) if faces else 0}
        Suggestions: {len(suggestions) if suggestions else 0}
        """
        
        if faces:
            result += f"\nFace details: {faces}"
        if suggestions:
            result += f"\nSuggestion details: {suggestions}"
            
        return result
        
    except Exception as e:
        return f"Error: {e}"

@app.route('/video-namer')
@time_function("video_namer.main_page")
def video_namer():
    """Show the next unidentified video that exists on disk, with insightface suggestions."""
    from models import Video
    import os, json
    
    # Check if a specific video ID was requested
    requested_video_id = request.args.get('video_id')
    
    with time_operation("video_namer.file_exists_check"):
        def file_exists(vid):
            if vid.file_path:
                file_path = vid.file_path.replace('/static/', 'static/')
                if os.path.exists(file_path):
                    return True
            return False
    
    with time_operation("video_namer.query_videos_to_name"):
        if requested_video_id:
            # Show specific video
            video_to_show = Video.query.get(requested_video_id)
            if not video_to_show or not file_exists(video_to_show):
                return redirect(url_for('video_namer'))
        else:
            # Get videos without people associated
            videos_to_name = Video.query.filter(
                ~Video.people.any()  # No people associated
            ).order_by(Video.id.desc()).all()
            
            # Filter for file existence and skipped tags
            filtered_videos = []
            for vid in videos_to_name:
                if file_exists(vid) and 'skipped' not in (json.loads(vid.tags) if vid.tags else []):
                    filtered_videos.append(vid)
            
            video_to_show = filtered_videos[0] if filtered_videos else None
        
        unnamed_count = Video.query.filter(~Video.people.any()).count()
        
        # If no unnamed videos, redirect to video tab
        if not video_to_show:
            return redirect(url_for('video_tab'))

    suggestions = []
    best_suggestion = None
    faces = []
    face_bbox = None
    all_faces = []
    face_best_guesses = []
    
    if video_to_show:
        # Run face detection on the video thumbnail
        with time_operation("video_namer.face_detection"):
            try:
                from face_recognition_module import get_face_db
                face_db = get_face_db()
                
                # First try the cached thumbnail if it exists
                thumbnail_path = video_to_show.thumbnail_path
                detection_path = None
                
                if thumbnail_path and os.path.exists(thumbnail_path.replace('/static/', 'static/')):
                    detection_path = thumbnail_path.replace('/static/', 'static/')
                else:
                    # Fallback: extract a frame from the video
                    video_path = video_to_show.file_path.replace('/static/', 'static/')
                    
                    if os.path.exists(video_path):
                        try:
                            import cv2
                            cap = cv2.VideoCapture(video_path)
                            
                            if cap.isOpened():
                                # Get a frame from 1 second into the video
                                cap.set(cv2.CAP_PROP_POS_MSEC, 1000)  # 1 second
                                ret, frame = cap.read()
                                
                                if ret and frame is not None:
                                    # Save the frame temporarily for face detection
                                    temp_frame_path = f"temp_frame_{video_to_show.id}.jpg"
                                    cv2.imwrite(temp_frame_path, frame)
                                    detection_path = temp_frame_path
                                    print(f"Extracted frame from video for face detection: {detection_path}")
                                
                                cap.release()
                        except ImportError:
                            print("OpenCV not available, skipping frame extraction")
                        except Exception as e:
                            print(f"Error extracting frame: {e}")
                
                if detection_path and os.path.exists(detection_path):
                    # Run face detection
                    detected_faces = face_db.detect_faces(detection_path)
                    
                    if detected_faces:
                        all_faces = detected_faces
                        faces = detected_faces  # Keep for backward compatibility
                        face_bbox = detected_faces[0]  # Use first face for initial display
                        
                        # Process each face individually for recognition (like reprocessing)
                        suggestions = []
                        face_best_guesses = []
                        
                        for i, face in enumerate(detected_faces):
                            print(f"[video_namer] Processing face {i + 1} individually: {face}")
                            
                            try:
                                # Extract individual face region from image (like reprocessing)
                                from PIL import Image as PILImage
                                
                                # Load the image
                                im = PILImage.open(detection_path)
                                
                                # Add padding around face (like reprocessing)
                                padding_x = int(face['width'] * 0.5)
                                padding_y = int(face['height'] * 0.5)
                                
                                # Calculate crop box with padding
                                x1 = max(0, face['x'] - padding_x)
                                y1 = max(0, face['y'] - padding_y)
                                x2 = min(im.width, face['x'] + face['width'] + padding_x)
                                y2 = min(im.height, face['y'] + face['height'] + padding_y)
                                
                                # Crop the individual face
                                face_crop = im.crop((x1, y1, x2, y2))
                                
                                # Ensure minimum size for face recognition (at least 112x112)
                                min_size = 112
                                if face_crop.width < min_size or face_crop.height < min_size:
                                    ratio = max(min_size / face_crop.width, min_size / face_crop.height)
                                    new_width = int(face_crop.width * ratio)
                                    new_height = int(face_crop.height * ratio)
                                    face_crop = face_crop.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                                
                                # Save individual face as temporary file
                                temp_crop_path = os.path.join(tempfile.gettempdir(), f'temp_face_crop_{i}_{uuid4().hex}.jpg')
                                face_crop.save(temp_crop_path)
                                
                                # Process THIS specific face individually
                                face_suggestions = face_db.find_similar_faces(temp_crop_path, threshold=0.0, top_k=5)
                                
                                # Clean up temp file
                                try:
                                    os.remove(temp_crop_path)
                                except Exception:
                                    pass
                                
                                if face_suggestions:
                                    # Get the best suggestion for this specific face
                                    best_suggestion = face_suggestions[0]
                                    best_guess = best_suggestion['person_name']
                                    face_best_guesses.append(best_guess)
                                    
                                    # Add to overall suggestions if not already present
                                    if not any(s['name'] == best_guess for s in suggestions):
                                        suggestions.append({
                                            'person_id': best_suggestion['person_id'],
                                            'name': best_suggestion['person_name'],
                                            'confidence': best_suggestion['similarity'],
                                            'is_confirmed': best_suggestion['is_confirmed']
                                        })
                                    
                                    print(f"[video_namer] Face {i + 1} individually matched: {best_guess} (confidence: {best_suggestion['similarity']:.3f})")
                                else:
                                    # Try fallback with original thumbnail for this specific face
                                    print(f"[video_namer] No suggestions for face {i + 1}, trying fallback")
                                    try:
                                        if video_to_show and video_to_show.thumbnail_path:
                                            thumbnail_path = video_to_show.thumbnail_path
                                            if thumbnail_path.startswith('/static/'):
                                                thumbnail_path = thumbnail_path[1:]
                                            
                                            if os.path.exists(thumbnail_path):
                                                # Process original thumbnail for this face
                                                fallback_suggestions = face_db.find_similar_faces(thumbnail_path, threshold=0.0, top_k=5)
                                                if fallback_suggestions:
                                                    best_suggestion = fallback_suggestions[0]
                                                    best_guess = best_suggestion['person_name']
                                                    face_best_guesses.append(best_guess)
                                                    
                                                    if not any(s['name'] == best_guess for s in suggestions):
                                                        suggestions.append({
                                                            'person_id': best_suggestion['person_id'],
                                                            'name': best_suggestion['person_name'],
                                                            'confidence': best_suggestion['similarity'],
                                                            'is_confirmed': best_suggestion['is_confirmed']
                                                        })
                                                    
                                                    print(f"[video_namer] Face {i + 1} fallback matched: {best_guess}")
                                                else:
                                                    face_best_guesses.append('Unknown')
                                                    print(f"[video_namer] Face {i + 1} fallback failed, using Unknown")
                                            else:
                                                face_best_guesses.append('Unknown')
                                                print(f"[video_namer] Face {i + 1} fallback thumbnail not found, using Unknown")
                                        else:
                                            face_best_guesses.append('Unknown')
                                            print(f"[video_namer] Face {i + 1} no fallback thumbnail, using Unknown")
                                    except Exception as e:
                                        face_best_guesses.append('Unknown')
                                        print(f"[video_namer] Face {i + 1} fallback error: {e}")
                                
                            except Exception as e:
                                print(f"[video_namer] Error processing face {i + 1}: {e}")
                                face_best_guesses.append('Unknown')
                        
                        # Set best suggestion for the first face
                        best_suggestion = suggestions[0] if suggestions else None
                    else:
                        all_faces = []
                        faces = []
                        face_best_guesses = []
                        suggestions = []
                        best_suggestion = None
                    
                    # Clean up temporary file if it was created
                    if detection_path.startswith("temp_frame_"):
                        try:
                            os.remove(detection_path)
                        except:
                            pass
                else:
                    faces = []
                    all_faces = []
                    face_best_guesses = []
                    suggestions = []
                    best_suggestion = None
                    
            except Exception as e:
                print(f"Face detection error for video {video_to_show.id}: {e}")
                import traceback
                traceback.print_exc()
                suggestions = []
                best_suggestion = None
                faces = []
                all_faces = []
                face_best_guesses = []
    
    return render_template('video_namer.html',
                         video=video_to_show,
                         suggestions=suggestions,
                         best_suggestion=best_suggestion,
                         faces=faces,
                         face_bbox=face_bbox,
                         all_faces=all_faces,
                         face_best_guesses=face_best_guesses,
                         unnamed_count=unnamed_count,
                         progress={'total': unnamed_count},
                         current_sort='id-desc')

@app.route('/namer')
@time_function("namer.main_page")
def namer():
    """Show the next unidentified image that exists on disk, with insightface suggestions."""
    from models import Image
    import os, json
    
    with time_operation("namer.file_exists_check"):
        def file_exists(img):
            if img.original_path:
                file_path = img.original_path.replace('/static/', 'static/')
                if os.path.exists(file_path):
                    return True
            uploads_path = os.path.join('static', 'uploads', img.filename)
            if os.path.exists(uploads_path):
                return True
            return False
    with time_operation("namer.query_images_to_name"):
        # OPTIMIZATION: Use database-level filtering instead of lazy loading
        # This avoids the N+1 query problem where each image triggers a separate query for people
        images_to_name = Image.query.filter(
            ~Image.people.any()  # No people associated
        ).order_by(Image.id.desc()).all()
        
        # Filter for file existence and skipped tags
        filtered_images = []
        for img in images_to_name:
            if file_exists(img) and 'skipped' not in (json.loads(img.tags) if img.tags else []):
                filtered_images.append(img)
        
        image_to_show = filtered_images[0] if filtered_images else None
        unnamed_count = len(filtered_images)
        
        # If no unnamed images, redirect to all images
        if not image_to_show:
            return redirect(url_for('all_images'))

    suggestions = []
    best_suggestion = None
    face_bbox = None
    all_faces = []
    face_best_guesses = []
    
    if image_to_show:
        # Determine image path
        with time_operation("namer.determine_image_path"):
            if image_to_show.original_path and os.path.exists(image_to_show.original_path.replace('/static/', 'static/')):
                image_path = image_to_show.original_path.replace('/static/', 'static/')
            else:
                # Search for the actual file with hash prefix
                base_filename = image_to_show.filename
                uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
                
                # Look for files that contain the base filename
                matching_files = []
                if os.path.exists(uploads_dir):
                    for file in os.listdir(uploads_dir):
                        if base_filename in file:
                            matching_files.append(file)
                
                if matching_files:
                    # Use the first matching file (prefer cropped versions)
                    cropped_files = [f for f in matching_files if f.startswith('cropped_')]
                    if cropped_files:
                        image_path = os.path.join(uploads_dir, cropped_files[0])
                    else:
                        image_path = os.path.join(uploads_dir, matching_files[0])
                else:
                    image_path = None

        if image_path and os.path.exists(image_path):
            print(f'[NAMER] Image found: {image_path}')
            try:
                with time_operation("namer.face_recognition_processing", image_path=image_path):
                    from face_recognition_module import get_face_db
                    face_db = get_face_db()
                    
                    # OPTIMIZATION: Detect faces ONCE and process each face ONCE
                    with time_operation("namer.detect_faces_once"):
                        detected_faces = face_db.detect_faces(image_path)
                    
                    if detected_faces:
                        all_faces = detected_faces
                        face_bbox = detected_faces[0]  # Use first face for initial display
                        
                        # OPTIMIZATION: Process each face exactly once for recognition
                        with time_operation("namer.process_all_faces_once", faces_count=len(detected_faces)):
                            from PIL import Image as PILImage
                            import tempfile
                            
                            # Process all faces in one pass
                            face_suggestions = []
                            face_embeddings = []
                            
                            for idx, face in enumerate(detected_faces):
                                try:
                                    with time_operation("namer.crop_face", face_index=idx):
                                        # Crop the face with padding
                                        with PILImage.open(image_path) as im:
                                            padding_x = int(face['width'] * 0.5)
                                            padding_y = int(face['height'] * 0.5)
                                            x1 = max(0, face['x'] - padding_x)
                                            y1 = max(0, face['y'] - padding_y)
                                            x2 = min(im.width, face['x'] + face['width'] + padding_x)
                                            y2 = min(im.height, face['y'] + face['height'] + padding_y)
                                            face_crop = im.crop((x1, y1, x2, y2))
                                            
                                            # Ensure minimum size
                                            min_size = 112
                                            if face_crop.width < min_size or face_crop.height < min_size:
                                                ratio = max(min_size / face_crop.width, min_size / face_crop.height)
                                                new_width = int(face_crop.width * ratio)
                                                new_height = int(face_crop.height * ratio)
                                                face_crop = face_crop.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                                            
                                            # Save to temporary file
                                            temp_crop_path = f'static/uploads/temp/temp_face_crop_{image_to_show.id}_{idx}.jpg'
                                            os.makedirs(os.path.dirname(temp_crop_path), exist_ok=True)
                                            face_crop.save(temp_crop_path)
                                            print(f"[DEBUG] Created temp crop file: {temp_crop_path}")
                                            print(f"[DEBUG] File exists after save: {os.path.exists(temp_crop_path)}")
                                    
                                    # OPTIMIZATION: Get recognition for this face ONCE
                                    with time_operation("namer.recognize_face", face_index=idx):
                                        similar_faces = face_db.find_similar_faces(temp_crop_path, threshold=0.0, top_k=1)
                                    
                                    # Store results
                                    if similar_faces and similar_faces[0]['person_name']:
                                        face_suggestions.append(similar_faces[0]['person_name'])
                                    else:
                                        face_suggestions.append('Unknown')
                                    
                                    # DO NOT clean up temp file here - keep it for face_suggestions API
                                        
                                except Exception as e:
                                    print(f'[NAMER] Error processing face {idx}: {e}')
                                    face_suggestions.append('Unknown')
                            
                            # Set results
                            face_best_guesses = face_suggestions
                            
                            # Use the first face's suggestions for the main suggestions list
                            temp_crop_path_first = f'static/uploads/temp/temp_face_crop_{image_to_show.id}_0.jpg'
                            if os.path.exists(temp_crop_path_first):
                                with time_operation("namer.get_detailed_suggestions_first_face"):
                                    similar_faces = face_db.find_similar_faces(temp_crop_path_first, threshold=0.0, top_k=20)
                                    seen_names = set()
                                    suggestions = []
                                    for face in similar_faces:
                                        if face['person_name'] and face['person_name'] not in seen_names:
                                            seen_names.add(face['person_name'])
                                            suggestions.append({
                                                'person_id': face['person_id'],
                                                'name': face['person_name'],
                                                'confidence': face['similarity'],
                                                'is_confirmed': face['is_confirmed']
                                            })
                                        if len(suggestions) == 5:
                                            break
                                    best_suggestion = suggestions[0] if suggestions else None
                                    while len(suggestions) < 5:
                                        suggestions.append({'person_id': None, 'name': 'No match', 'confidence': 0.0, 'is_confirmed': False})
                                # DO NOT clean up temp file for first face here - keep it for face_suggestions API
                            else:
                                # Fallback if temp file doesn't exist
                                suggestions = [{'person_id': None, 'name': 'No match', 'confidence': 0.0, 'is_confirmed': False}] * 5
                                best_suggestion = None
                    else:
                        all_faces = []
                        face_best_guesses = []
                        suggestions = [{'person_id': None, 'name': 'No match', 'confidence': 0.0, 'is_confirmed': False}] * 5
                        best_suggestion = None
                        
                    print(f'[NAMER] Face bbox: {face_bbox}')
                    print(f'[NAMER] All faces: {all_faces}')
                    print(f'[NAMER] Face best guesses: {face_best_guesses}')
                    print(f'[NAMER] Suggestions: {suggestions}')
                    
            except Exception as e:
                print(f'[NAMER] Face recognition error: {e}')
                suggestions = []
                best_suggestion = None
                face_bbox = None
                all_faces = []
                face_best_guesses = []
        else:
            print(f'[NAMER] Image file not found: {image_path}')
    else:
        print('[NAMER] No images to name.')
    
    # Get saved rotation from metadata
    saved_rotation = 0
    if image_to_show and image_to_show.metadata:
        try:
            metadata = json.loads(image_to_show.metadata)
            saved_rotation = metadata.get('rotation', 0)
        except:
            saved_rotation = 0

    return render_template('namer.html', image=image_to_show.to_dict() if image_to_show else None, suggestions=suggestions, best_suggestion=best_suggestion, unnamed_count=unnamed_count, face_bbox=face_bbox, all_faces=all_faces, saved_rotation=saved_rotation, face_best_guesses=face_best_guesses, progress={'total': unnamed_count}, current_sort='id-desc')

@app.route('/api/namer/next_image', methods=['POST'])
def api_namer_next_image():
    from models import Image
    import os, json
    data = request.get_json() or {}
    skipped_ids = set(data.get('skipped_ids', []))
    def file_exists(img):
        if img.original_path:
            file_path = img.original_path.replace('/static/', 'static/')
            if os.path.exists(file_path):
                return True
        uploads_path = os.path.join('static', 'uploads', img.filename)
        if os.path.exists(uploads_path):
            return True
        return False
    # OPTIMIZATION: Use database-level filtering instead of lazy loading
    images_to_name = Image.query.filter(
        ~Image.people.any()  # No people associated
    ).order_by(Image.id.asc()).all()
    
    # Filter for file existence and skipped IDs
    filtered_images = []
    for img in images_to_name:
        if file_exists(img) and str(img.id) not in skipped_ids:
            filtered_images.append(img)
    
    image_to_show = filtered_images[0] if filtered_images else None
    return jsonify({'image': image_to_show.to_dict() if image_to_show else None})

@app.route('/api/namer/suggestions/<int:image_id>')
@time_function("api.namer.suggestions")
def api_namer_suggestions(image_id):
    # Use the same logic as /api/face_suggestions/<int:image_id>
    try:
        with time_operation("api.namer.query_image"):
            from models import Image
            image = Image.query.get_or_404(image_id)
        
        # Get the image path - handle hash-prefixed filenames
        if image.original_path and os.path.exists(image.original_path.replace('/static/', 'static/')):
            image_path = image.original_path.replace('/static/', 'static/')
        else:
            # Search for the actual file with hash prefix
            base_filename = image.filename
            uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
            
            # Look for files that contain the base filename
            matching_files = []
            if os.path.exists(uploads_dir):
                for file in os.listdir(uploads_dir):
                    if base_filename in file:
                        matching_files.append(file)
            
            if matching_files:
                # Use the first matching file (prefer cropped versions)
                cropped_files = [f for f in matching_files if f.startswith('cropped_')]
                if cropped_files:
                    image_path = os.path.join(uploads_dir, cropped_files[0])
                else:
                    image_path = os.path.join(uploads_dir, matching_files[0])
            else:
                return jsonify({'success': False, 'error': f'No matching files found for {base_filename}'}), 404
        
        # Check if image file exists
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': f'Image file not found: {image_path}'}), 404
        
        # Get face recognition
        face_db = get_face_db()
        
        # Get similar faces (no threshold, get more candidates)
        similar_faces = face_db.find_similar_faces(image_path, threshold=0.0, top_k=50)
        
        # Format suggestions - ensure unique names
        seen_names = set()
        suggestions = []
        for face in similar_faces:
            if face['person_name'] not in seen_names and len(suggestions) < 6:
                seen_names.add(face['person_name'])
                suggestions.append({
                    'person_id': face['person_id'],
                    'name': face['person_name'],
                    'confidence': face['similarity'],
                    'is_confirmed': face['is_confirmed']
                })
        
        return jsonify({
            'success': True,
            'suggestions': suggestions
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/namer/save', methods=['POST'])
@time_function("api.namer.save")
def api_namer_save():
    from models import db, Image, Person
    import json
    data = request.get_json()
    image_id = data.get('image_id')
    name = data.get('name')
    face_index = data.get('face_index', 0)  # Default to first face if not specified
    if not image_id or not name:
        return {'success': False, 'error': 'Missing image_id or name'}, 400

    with time_operation("api.namer.query_database"):
        image = Image.query.get(image_id)
        if not image:
            return {'success': False, 'error': 'Image not found'}, 404
        person = Person.query.filter_by(name=name).first()
        if not person:
            person = Person(name=name, is_confirmed=True)
            db.session.add(person)
            db.session.flush()

    with time_operation("api.namer.update_image"):
        # For single face save, we should preserve existing people associations
        # Only add the new person if they're not already associated
        if person not in image.people:
            image.people.append(person)
        # Remove 'named' tag if present
        tags = json.loads(image.tags) if image.tags else []
        if 'named' in tags:
            tags.remove('named')
        image.tags = json.dumps(tags)

    with time_operation("api.namer.commit_changes"):
        db.session.commit()

    # Add face to face recognition database
    try:
        # Get image path
        if image.original_path and os.path.exists(image.original_path.replace('/static/', 'static/')):
            image_path = image.original_path.replace('/static/', 'static/')
        else:
            # Search for the actual file with hash prefix
            base_filename = image.filename
            uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
            matching_files = []
            if os.path.exists(uploads_dir):
                for file in os.listdir(uploads_dir):
                    if base_filename in file:
                        matching_files.append(file)
            if matching_files:
                cropped_files = [f for f in matching_files if f.startswith('cropped_')]
                if cropped_files:
                    image_path = os.path.join(uploads_dir, cropped_files[0])
                else:
                    image_path = os.path.join(uploads_dir, matching_files[0])
            else:
                image_path = None
        num_faces = 1
        if image_path and os.path.exists(image_path):
            from face_recognition_module import get_face_db
            face_db = get_face_db()
            detected_faces = face_db.detect_faces(image_path)
            num_faces = len(detected_faces) if detected_faces else 1
            
            # Add face to recognition database
            if detected_faces and len(detected_faces) > 0:
                # Use the first detected face for single face save
                face = detected_faces[0]
                
                # Process individual face (same as video namer logic)
                from PIL import Image as PILImage
                
                # Load the image
                im = PILImage.open(image_path)
                
                # Add padding around face
                padding_x = int(face['width'] * 0.5)
                padding_y = int(face['height'] * 0.5)
                
                # Calculate crop box with padding
                x1 = max(0, face['x'] - padding_x)
                y1 = max(0, face['y'] - padding_y)
                x2 = min(im.width, face['x'] + face['width'] + padding_x)
                y2 = min(im.height, face['y'] + face['height'] + padding_y)
                
                # Crop the individual face
                face_crop = im.crop((x1, y1, x2, y2))
                
                # Ensure minimum size for face recognition (112x112)
                min_size = 112
                if face_crop.width < min_size or face_crop.height < min_size:
                    ratio = max(min_size / face_crop.width, min_size / face_crop.height)
                    new_width = int(face_crop.width * ratio)
                    new_height = int(face_crop.height * ratio)
                    face_crop = face_crop.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                
                # Save individual face as temporary file
                import tempfile
                from uuid import uuid4
                temp_crop_path = os.path.join(tempfile.gettempdir(), f'image_face_crop_{image_id}_{uuid4().hex}.jpg')
                face_crop.save(temp_crop_path)
                
                # Add THIS specific face to recognition database
                success = face_db.add_face_to_database(
                    image_path=temp_crop_path,
                    person_id=person.id,
                    person_name=person.name,
                    is_confirmed=person.is_confirmed
                )
                
                # Clean up temp file
                try:
                    os.remove(temp_crop_path)
                except Exception:
                    pass
                
                if success:
                    print(f"✓ Added image face for {person.name} to recognition database")
                else:
                    print(f"✗ Failed to add image face for {person.name}")
                
                # Reload face database to make new faces immediately available
                reload_face_db()
        
        # Clean up temp crops for this image
        cleanup_temp_crops(image_id, num_faces)
    except Exception as e:
        print(f"Error adding face to recognition database: {e}")
    
    # Check if there are more unnamed images
    remaining_unnamed = Image.query.filter(~Image.people.any()).count()
    if remaining_unnamed == 0:
        return {'success': True, 'redirect_to': url_for('all_images')}
    
    return {'success': True}

@app.route('/api/namer/skip', methods=['POST'])
def api_namer_skip():
    from models import db, Image
    import json
    data = request.get_json()
    image_id = data.get('image_id')
    if not image_id:
        return {'success': False, 'error': 'Missing image_id'}, 400
    image = Image.query.get(image_id)
    if not image:
        return {'success': False, 'error': 'Image not found'}, 404
    # Add 'skipped' tag to mark this image as skipped
    tags = json.loads(image.tags) if image.tags else []
    if 'skipped' not in tags:
        tags.append('skipped')
    image.tags = json.dumps(tags)
    db.session.commit()
    # Clean up temp crops for this image
    # Try to get number of faces
    num_faces = 1
    try:
        if image.original_path and os.path.exists(image.original_path.replace('/static/', 'static/')):
            image_path = image.original_path.replace('/static/', 'static/')
        else:
            base_filename = image.filename
            uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
            matching_files = []
            if os.path.exists(uploads_dir):
                for file in os.listdir(uploads_dir):
                    if base_filename in file:
                        matching_files.append(file)
            if matching_files:
                cropped_files = [f for f in matching_files if f.startswith('cropped_')]
                if cropped_files:
                    image_path = os.path.join(uploads_dir, cropped_files[0])
                else:
                    image_path = os.path.join(uploads_dir, matching_files[0])
            else:
                image_path = None
        if image_path and os.path.exists(image_path):
            from face_recognition_module import get_face_db
            face_db = get_face_db()
            detected_faces = face_db.detect_faces(image_path)
            num_faces = len(detected_faces) if detected_faces else 1
    except Exception:
        pass
    cleanup_temp_crops(image_id, num_faces)
    return {'success': True}

@app.route('/api/namer/face_suggestions', methods=['POST'])
@time_function("api.namer.face_suggestions")
def api_namer_face_suggestions():
    """Get face recognition suggestions for a specific face in an image"""
    from face_recognition_module import get_face_db
    try:
        with time_operation("api.namer.face_suggestions.parse_request"):
            data = request.get_json()
            image_id = data.get('image_id')
            face_index = data.get('face_index', 0)
            
            if not image_id:
                return jsonify({'success': False, 'error': 'Missing image_id'}), 400
        
        with time_operation("api.namer.face_suggestions.query_image"):
            image = Image.query.get(image_id)
            if not image:
                return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        # OPTIMIZATION: Use the same temp crop file that was created during main page load
        temp_crop_path = f'static/uploads/temp/temp_face_crop_{image_id}_{face_index}.jpg'
        
        # Debug: Check what temp files exist
        import glob
        temp_files = glob.glob(f'static/uploads/temp/temp_face_crop_{image_id}_*.jpg')
        print(f"[DEBUG] Temp files for image {image_id}: {temp_files}")
        print(f"[DEBUG] Looking for: {temp_crop_path}")
        print(f"[DEBUG] File exists: {os.path.exists(temp_crop_path)}")
        
        if not os.path.exists(temp_crop_path):
            return jsonify({'success': False, 'error': f'Face crop not found for index {face_index}. Available files: {temp_files}. Please refresh the page.'}), 404
        
        # Get face recognition
        face_db = get_face_db()
        
        # OPTIMIZATION: Get suggestions directly from the existing crop file
        with time_operation("api.namer.face_suggestions.get_suggestions", face_index=face_index):
            similar_faces = face_db.find_similar_faces(temp_crop_path, threshold=0.0, top_k=20)
        
        # Format suggestions - ensure unique names
        with time_operation("api.namer.face_suggestions.format_results"):
            seen_names = set()
            suggestions = []
            for face in similar_faces:
                if face['person_name'] and face['person_name'] not in seen_names and len(suggestions) < 5:
                    seen_names.add(face['person_name'])
                    suggestions.append({
                        'person_id': face['person_id'],
                        'name': face['person_name'],
                        'confidence': face['similarity'],
                        'is_confirmed': face['is_confirmed']
                    })
            
            # Fill remaining slots
            while len(suggestions) < 5:
                suggestions.append({'person_id': None, 'name': 'No match', 'confidence': 0.0, 'is_confirmed': False})
            
            best_suggestion = suggestions[0] if suggestions and suggestions[0]['person_id'] else None
            if not best_suggestion:
                best_suggestion = {'person_id': None, 'name': 'No match', 'confidence': 0.0, 'is_confirmed': False}
        
        # Get face bbox from the main page data (this should be available)
        # For now, we'll return a placeholder - in a real implementation, you might want to store this in session
        face_bbox = {'x': 0, 'y': 0, 'width': 100, 'height': 100}  # Placeholder
        
        return jsonify({
            'success': True,
            'suggestions': suggestions,
            'best_suggestion': best_suggestion,
            'face_bbox': face_bbox
        })
        
    except Exception as e:
        print(f"Error in face suggestions API: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/namer/save_rotation', methods=['POST'])
def api_namer_save_rotation():
    from models import db, Image
    import json
    import os
    from PIL import Image as PILImage
    
    try:
        data = request.get_json()
        image_id = data.get('image_id')
        rotation = data.get('rotation', 0)
        
        if not image_id:
            return jsonify({'success': False, 'error': 'Missing image_id'}), 400
        
        image = Image.query.get(image_id)
        if not image:
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        # Get the image file path
        image_path = None
        if image.original_path and os.path.exists(image.original_path.replace('/static/', 'static/')):
            image_path = image.original_path.replace('/static/', 'static/')
        else:
            # Search for the actual file with hash prefix
            base_filename = image.filename
            uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
            
            # Look for files that contain the base filename
            matching_files = []
            if os.path.exists(uploads_dir):
                for file in os.listdir(uploads_dir):
                    if base_filename in file:
                        matching_files.append(file)
            
            if matching_files:
                # Use the first matching file (prefer cropped versions)
                cropped_files = [f for f in matching_files if f.startswith('cropped_')]
                if cropped_files:
                    image_path = os.path.join(uploads_dir, cropped_files[0])
                else:
                    image_path = os.path.join(uploads_dir, matching_files[0])
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'success': False, 'error': f'Image file not found: {image_path}'}), 404
        
        # Physically rotate the image file
        try:
            with PILImage.open(image_path) as pil_img:
                # Convert to RGB if necessary
                if pil_img.mode in ('RGBA', 'LA', 'P'):
                    pil_img = pil_img.convert('RGB')
                
                # Rotate the image
                rotated_img = pil_img.rotate(-rotation, expand=True)  # Negative for clockwise
                
                # Save the rotated image back to the same file
                rotated_img.save(image_path, quality=95, optimize=True)
                print(f"✓ Physically rotated image {image_id} by {rotation} degrees")
                
        except Exception as e:
            print(f"Error rotating image file: {e}")
            return jsonify({'success': False, 'error': f'Failed to rotate image file: {str(e)}'}), 500
        
        # Store rotation in image metadata (set to 0 since we physically rotated)
        try:
            if image.metadata and isinstance(image.metadata, str):
                metadata = json.loads(image.metadata)
            else:
                metadata = {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        
        metadata['rotation'] = 0  # Reset to 0 since we physically rotated
        image.metadata = json.dumps(metadata)
        
        try:
            db.session.commit()
            print(f"✓ Saved rotation metadata for image {image_id}")
            return jsonify({'success': True})
        except Exception as e:
            print(f"Error committing to database: {e}")
            db.session.rollback()
            return jsonify({'success': False, 'error': 'Database error'}), 500
            
    except Exception as e:
        print(f"Error in save_rotation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/namer/save_multiple', methods=['POST'])
@time_function("api.namer.save_multiple")
def api_namer_save_multiple():
    from models import db, Image, Person
    import json
    data = request.get_json()
    image_id = data.get('image_id')
    names = data.get('names', [])
    
    if not image_id or not names:
        return {'success': False, 'error': 'Missing image_id or names data'}, 400

    with time_operation("api.namer.query_database"):
        image = Image.query.get(image_id)
        if not image:
            return {'success': False, 'error': 'Image not found'}, 404

    with time_operation("api.namer.update_image"):
        # Clear existing people associations
        image.people.clear()
        
        # Add each person for the names
        for name in names:
            name = name.strip()
            if name:
                person = Person.query.filter_by(name=name).first()
                if not person:
                    person = Person(name=name, is_confirmed=True)
                    db.session.add(person)
                    db.session.flush()
                
                if person not in image.people:
                    image.people.append(person)
        
        # Remove 'named' tag if present
        tags = json.loads(image.tags) if image.tags else []
        if 'named' in tags:
            tags.remove('named')
        image.tags = json.dumps(tags)

    with time_operation("api.namer.commit_changes"):
        db.session.commit()

    # Add faces to face recognition database
    try:
        # Get image path
        if image.original_path and os.path.exists(image.original_path.replace('/static/', 'static/')):
            image_path = image.original_path.replace('/static/', 'static/')
        else:
            # Search for the actual file with hash prefix
            base_filename = image.filename
            uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
            matching_files = []
            if os.path.exists(uploads_dir):
                for file in os.listdir(uploads_dir):
                    if base_filename in file:
                        matching_files.append(file)
            if matching_files:
                cropped_files = [f for f in matching_files if f.startswith('cropped_')]
                if cropped_files:
                    image_path = os.path.join(uploads_dir, cropped_files[0])
                else:
                    image_path = os.path.join(uploads_dir, matching_files[0])
            else:
                image_path = None
        
        num_faces = len(names)
        if image_path and os.path.exists(image_path):
            from face_recognition_module import get_face_db
            face_db = get_face_db()
            detected_faces = face_db.detect_faces(image_path)
            num_faces = len(detected_faces) if detected_faces else len(names)
            
            # Add faces to recognition database
            if detected_faces and len(detected_faces) > 0:
                # Process each face individually (same as video namer logic)
                from PIL import Image as PILImage
                
                # Load the image
                im = PILImage.open(image_path)
                
                for i, face in enumerate(detected_faces):
                    if i < len(names) and names[i].strip():
                        person_name = names[i].strip()
                        person = Person.query.filter_by(name=person_name).first()
                        if person:
                            try:
                                # Add padding around face
                                padding_x = int(face['width'] * 0.5)
                                padding_y = int(face['height'] * 0.5)
                                
                                # Calculate crop box with padding
                                x1 = max(0, face['x'] - padding_x)
                                y1 = max(0, face['y'] - padding_y)
                                x2 = min(im.width, face['x'] + face['width'] + padding_x)
                                y2 = min(im.height, face['y'] + face['height'] + padding_y)
                                
                                # Crop the individual face
                                face_crop = im.crop((x1, y1, x2, y2))
                                
                                # Ensure minimum size for face recognition (112x112)
                                min_size = 112
                                if face_crop.width < min_size or face_crop.height < min_size:
                                    ratio = max(min_size / face_crop.width, min_size / face_crop.height)
                                    new_width = int(face_crop.width * ratio)
                                    new_height = int(face_crop.height * ratio)
                                    face_crop = face_crop.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                                
                                # Save individual face as temporary file
                                import tempfile
                                from uuid import uuid4
                                temp_crop_path = os.path.join(tempfile.gettempdir(), f'image_face_crop_{image_id}_{i}_{uuid4().hex}.jpg')
                                face_crop.save(temp_crop_path)
                                
                                # Add THIS specific face to recognition database
                                success = face_db.add_face_to_database(
                                    image_path=temp_crop_path,
                                    person_id=person.id,
                                    person_name=person.name,
                                    is_confirmed=person.is_confirmed
                                )
                                
                                # Clean up temp file
                                try:
                                    os.remove(temp_crop_path)
                                except Exception:
                                    pass
                                
                                if success:
                                    print(f"✓ Added image face {i+1} for {person_name} to recognition database")
                                else:
                                    print(f"✗ Failed to add image face {i+1} for {person_name}")
                                    
                            except Exception as face_error:
                                print(f"Error processing image face {i+1} for {person_name}: {face_error}")
                
                # Reload face database to make new faces immediately available
                reload_face_db()
        
        # Clean up temp crops for this image
        cleanup_temp_crops(image_id, num_faces)
    except Exception as e:
        print(f"Error adding faces to recognition database: {e}")
    
    # Check if there are more unnamed images
    remaining_unnamed = Image.query.filter(~Image.people.any()).count()
    if remaining_unnamed == 0:
        return {'success': True, 'redirect_to': url_for('all_images')}
    
    return {'success': True}

@app.route('/api/import_google_photos', methods=['POST'])
def import_google_photos():
    """Import photos from Google Photos API"""
    import requests
    from urllib.parse import urlparse
    import tempfile
    
    try:
        data = request.get_json()
        photo_urls = data.get('photo_urls', [])
        
        if not photo_urls:
            return jsonify({'success': False, 'error': 'No photo URLs provided'}), 400
        
        # Create imported directory if it doesn't exist
        imported_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'imported')
        os.makedirs(imported_folder, exist_ok=True)
        
        imported_count = 0
        errors = []
        
        for photo_url in photo_urls:
            try:
                # Download the photo
                response = requests.get(photo_url, timeout=30)
                response.raise_for_status()
                
                # Get content type and validate it's an image
                content_type = response.headers.get('content-type', '')
                if not content_type.startswith('image/'):
                    errors.append(f"Invalid content type: {content_type}")
                    continue
                
                # Generate unique filename
                parsed_url = urlparse(photo_url)
                original_filename = os.path.basename(parsed_url.path) or 'google_photo.jpg'
                filename = secure_filename(original_filename)
                base_name = f"{uuid.uuid4().hex}_{filename}"
                file_path = os.path.join(imported_folder, base_name)
                
                # Save the image
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                # Create image record
                new_image = Image(
                    original_path=f'/static/uploads/imported/{base_name}',
                    filename=filename,
                    is_cropped=False,
                    is_favorite=False,
                    tags=json.dumps([]),
                    recognition_suggestions=json.dumps([])
                )
                # Set width and height
                try:
                    from PIL import Image as PILImage
                    with PILImage.open(os.path.join(app.config['UPLOAD_FOLDER'], 'imported', base_name)) as im:
                        new_image.width = im.width
                        new_image.height = im.height
                except Exception as e:
                    print(f"Error setting image dimensions: {e}")
                db.session.add(new_image)
                db.session.commit()
                
                imported_count += 1
                
            except Exception as e:
                errors.append(f"Error processing {photo_url}: {str(e)}")
        
        if imported_count > 0:
            return jsonify({
                'success': True,
                'imported_count': imported_count,
                'errors': errors
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No photos were successfully imported',
                'errors': errors
            }), 400
            
    except Exception as e:
        print(f"Error in import_google_photos: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/performance/summary')
def api_performance_summary():
    """Get performance summary for debugging"""
    summary = get_performance_summary()
    return jsonify(summary)

@app.route('/performance')
def performance_page():
    """Performance monitoring page"""
    summary = get_performance_summary()
    return render_template('performance.html', summary=summary)

def cleanup_temp_crops(image_id, num_faces):
    import glob
    for idx in range(num_faces):
        temp_crop_path = f'static/uploads/temp/temp_face_crop_{image_id}_{idx}.jpg'
        if os.path.exists(temp_crop_path):
            try:
                os.remove(temp_crop_path)
            except Exception:
                pass
    # Also clean up any stray crops for this image_id
    for path in glob.glob(f'static/uploads/temp/temp_face_crop_{image_id}_*.jpg'):
        try:
            os.remove(path)
        except Exception:
            pass

@app.route('/missing_google_photos')
def missing_google_photos():
    """Page to find Google Photos images missing from visage."""
    return render_template('missing_google_photos.html')

@app.route('/api/build_phash_db', methods=['POST'])
def api_build_phash_db():
    """API endpoint to build pHash DB for both Google Photos and visage images."""
    # These paths should match your actual image roots
    google_photos_dir = 'fitpexport/Takeout/Google Photos/fitp/'
    visage_dirs = ['static/uploads/', 'face_crops/']
    all_dirs = [google_photos_dir] + visage_dirs
    build_phash_db(all_dirs)
    return jsonify({'success': True})

@app.route('/api/find_missing_google_photos')
def api_find_missing_google_photos():
    import os
    google_photos_dir = 'fitpexport\\Takeout\\Google Photos\\fitp\\'
    # Only use images that are named (have at least one associated person)
    visage_images = []
    for img in Image.query.filter(Image.people.any()).all():
        path = img.original_path
        if path and path.startswith('/static/'):
            file_path = path.replace('/static/', 'static/')
            if os.path.exists(file_path):
                visage_images.append(file_path)
        else:
            file_path = os.path.join('static', 'uploads', img.filename)
            if os.path.exists(file_path):
                visage_images.append(file_path)
    # Build a set of pHashes for these images
    from utils.phash_db import get_phash
    visage_phashes = set()
    for vpath in visage_images:
        phash = get_phash(vpath)
        if phash:
            visage_phashes.add(phash)
    # Now check Google Photos images
    all_phashes = dict(get_all_phashes())
    missing = []
    for path, phash in all_phashes.items():
        if path.startswith(google_photos_dir):
            found = False
            for vphash in visage_phashes:
                if phash and vphash:
                    dist = compute_phash_distance(phash, vphash)
                    if dist <= 6:
                        found = True
                        break
            if not found:
                missing.append(path)
    return jsonify({'missing': missing})

def compute_phash_distance(phash1, phash2):
    import imagehash
    return imagehash.hex_to_hash(phash1) - imagehash.hex_to_hash(phash2)

@app.route('/api/phash_db_stats')
def api_phash_db_stats():
    google_photos_dir = 'fitpexport\\Takeout\\Google Photos\\fitp\\'
    visage_dirs = ['static\\uploads\\', 'face_crops\\']
    all_phashes = dict(get_all_phashes())
    google_count = sum(1 for p in all_phashes if p.startswith(google_photos_dir))
    visage_count = sum(1 for p in all_phashes if any(p.startswith(vdir) for vdir in visage_dirs))
    return jsonify({'google_photos_count': google_count, 'visage_count': visage_count, 'total': len(all_phashes)})

@app.route('/api/phash_db_sample')
def api_phash_db_sample():
    rows = get_all_phashes()
    sample = [p for p, _ in rows[:20]]
    return jsonify({'sample': sample})

@app.route('/api/phash_db_visage_sample')
def api_phash_db_visage_sample():
    visage_dirs = ['static\\uploads\\', 'face_crops\\']
    rows = get_all_phashes()
    visage_paths = [p for p, _ in rows if any(p.startswith(vdir) for vdir in visage_dirs)]
    sample = visage_paths[:20]
    return jsonify({'sample': sample, 'count': len(visage_paths)})

@app.route('/api/visage_images_sample')
def api_visage_images_sample():
    import os
    visage_images = []
    for img in Image.query.all():
        path = img.original_path
        if path and path.startswith('/static/'):
            file_path = path.replace('/static/', 'static/')
            if os.path.exists(file_path):
                visage_images.append(file_path)
        else:
            file_path = os.path.join('static', 'uploads', img.filename)
            if os.path.exists(file_path):
                visage_images.append(file_path)
    sample = visage_images[:20]
    return jsonify({'sample': sample, 'count': len(visage_images)})

@app.route('/api/google_vs_visage_matches')
def api_google_vs_visage_matches():
    import os
    google_photos_dir = 'fitpexport\\Takeout\\Google Photos\\fitp\\'
    # Only use images that are named (have at least one associated person)
    visage_images = []
    for img in Image.query.filter(Image.people.any()).all():
        path = img.original_path
        if path and path.startswith('/static/'):
            file_path = path.replace('/static/', 'static/')
            if os.path.exists(file_path):
                visage_images.append((file_path, img))
        else:
            file_path = os.path.join('static', 'uploads', img.filename)
            if os.path.exists(file_path):
                visage_images.append((file_path, img))
    # Build a list of (path, phash, image info) for these images
    from utils.phash_db import get_phash
    visage_phash_list = []
    for vpath, vimg in visage_images:
        phash = get_phash(vpath)
        if phash:
            visage_phash_list.append((vpath, phash, vimg))
    # Now for each Google Photos image, find best match
    all_phashes = dict(get_all_phashes())
    results = []
    for path, phash in all_phashes.items():
        if path.startswith(google_photos_dir):
            best = None
            best_dist = None
            for vpath, vphash, vimg in visage_phash_list:
                if phash and vphash:
                    dist = compute_phash_distance(phash, vphash)
                    if best is None or dist < best_dist:
                        best = (vpath, vimg)
                        best_dist = dist
            results.append({
                'google_path': path,
                'google_url': '/'+path.replace('\\', '/'),
                'best_visage_path': best[0] if best else None,
                'best_visage_url': '/'+best[0].replace('\\', '/') if best else None,
                'best_dist': best_dist
            })
    return jsonify({'matches': results})

@app.route('/search')
def search():
    return render_template('search.html')

@app.route('/api/search')
def api_search():
    from models import Person, Image
    import re, json
    q = request.args.get('q', '').strip().lower()
    people_results = []
    image_results = []
    if q:
        # Case-insensitive partial match for people
        people = Person.query.filter(Person.name.ilike(f'%{q}%')).all()
        for person in people:
            people_results.append({
                'id': person.id,
                'name': person.name,
                'thumbnail': person.to_dict().get('thumbnail'),
                'image_count': person.image_count
            })
        # Search images by tags (clothing tags)
        images = Image.query.all()
        for img in images:
            tags = json.loads(img.tags) if img.tags else []
            if any(q in tag.lower() for tag in tags):
                image_results.append(img.to_dict())
    return jsonify({'people': people_results, 'images': image_results})

@app.route('/profile_pictures')
def profile_pictures():
    """People Management page"""
    from models import Person
    people = Person.query.order_by(Person.name).all()
    return render_template('profile_pictures.html', people=people)

@app.route('/api/person_images/<int:person_id>')
def api_person_images(person_id):
    from models import Person, Video, db
    import os
    import cv2
    from pathlib import Path
    person = Person.query.get_or_404(person_id)
    images = person.images
    
    # Build image list
    image_list = [
        {
            'id': img.id,
            'image_url': img.to_dict().get('image_url'),
            'filename': img.filename,
            'width': img.width,
            'height': img.height,
            'type': 'image'
        } for img in images
    ]
    
    # Always include video frames if videos exist
    if person.videos:
        try:
            # Extract frames from videos
            for video in person.videos:
                video_path = video.file_path.replace('/static/', 'static/')
                
                if not os.path.exists(video_path):
                    continue
                
                try:
                    # Try to use existing thumbnail first
                    if video.thumbnail_path and os.path.exists(video.thumbnail_path.replace('/static/', 'static/')):
                        image_list.append({
                            'id': f'video_{video.id}',
                            'image_url': video.thumbnail_path,
                            'filename': f'{video.filename}_frame.jpg',
                            'width': video.width,
                            'height': video.height,
                            'type': 'video_frame',
                            'video_id': video.id
                        })
                    else:
                        # Extract frame from video
                        cap = cv2.VideoCapture(video_path)
                        if cap.isOpened():
                            # Seek to 1 second
                            cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
                            ret, frame = cap.read()
                            cap.release()
                            
                            if ret and frame is not None:
                                # Save frame temporarily or generate thumbnail
                                video_filename = Path(video_path).stem
                                thumbnail_filename = f"{video_filename}_frame_{person_id}.jpg"
                                thumbnail_dir = "static/thumbnails/video_frames"
                                os.makedirs(thumbnail_dir, exist_ok=True)
                                thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
                                cv2.imwrite(thumbnail_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                
                                # Update video thumbnail_path if not set
                                if not video.thumbnail_path:
                                    video.thumbnail_path = f'/static/thumbnails/video_frames/{thumbnail_filename}'
                                    db.session.commit()
                                
                                image_list.append({
                                    'id': f'video_{video.id}',
                                    'image_url': f'/static/thumbnails/video_frames/{thumbnail_filename}',
                                    'filename': f'{video.filename}_frame.jpg',
                                    'width': frame.shape[1],
                                    'height': frame.shape[0],
                                    'type': 'video_frame',
                                    'video_id': video.id
                                })
                except Exception as e:
                    print(f"Error extracting frame from video {video.id}: {e}")
                    continue
        except ImportError:
            print("OpenCV not available, cannot extract video frames")
        except Exception as e:
            print(f"Error processing video frames: {e}")
    
    return jsonify({
        'success': True,
        'images': image_list
    })

@app.route('/api/set_profile_picture', methods=['POST'])
def api_set_profile_picture():
    from models import Person, Image, db
    import json
    data = request.get_json()
    person_id = data.get('person_id')
    image_id = data.get('image_id')
    if not person_id or not image_id:
        return jsonify({'success': False, 'error': 'Missing person_id or image_id'}), 400
    person = Person.query.get(person_id)
    image = Image.query.get(image_id)
    if not person or not image:
        return jsonify({'success': False, 'error': 'Person or image not found'}), 404
    # Set thumbnail_path to the image's image_url
    person.thumbnail_path = image.to_dict().get('image_url')
    db.session.commit()
    return jsonify({'success': True, 'thumbnail_path': person.thumbnail_path})

@app.route('/api/upload_profile_thumbnail', methods=['POST'])
def api_upload_profile_thumbnail():
    from models import Person, db
    import os
    person_id = request.form.get('person_id')
    file = request.files.get('file')
    if not person_id or not file:
        return jsonify({'success': False, 'error': 'Missing person_id or file'}), 400
    person = Person.query.get(person_id)
    if not person:
        return jsonify({'success': False, 'error': 'Person not found'}), 404
    # Save file
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'profile_thumbnails')
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    filename = f'profile_{person_id}{ext}'
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)
    # Update thumbnail_path
    person.thumbnail_path = f'/static/uploads/profile_thumbnails/{filename}'
    db.session.commit()
    return jsonify({'success': True, 'thumbnail_path': person.thumbnail_path})

@app.route('/api/cleanup_empty_people', methods=['POST'])
def api_cleanup_empty_people():
    """Remove all people who have no images or videos attached"""
    try:
        from models import Person, db
        from face_recognition_module import get_face_db
        import numpy as np
        
        # Find people with no images and no videos
        people_to_delete = []
        for person in Person.query.all():
            has_images = len(person.images) > 0
            has_videos = len(person.videos) > 0
            
            if not has_images and not has_videos:
                people_to_delete.append(person)
        
        deleted_count = 0
        
        # Remove from face database first, then delete from main database
        face_db = get_face_db()
        
        for person in people_to_delete:
            try:
                # Remove from face database
                if face_db and person.id in face_db.faces_db.get('people', {}):
                    # Find all embeddings for this person
                    person_embedding_indices = []
                    person_ids_list = face_db.faces_db.get('person_ids', [])
                    for i, pid in enumerate(person_ids_list):
                        if pid == person.id:
                            person_embedding_indices.append(i)
                    
                    # Remove embeddings in reverse order to maintain indices
                    for idx in reversed(sorted(person_embedding_indices)):
                        # Remove from embeddings array
                        embeddings = face_db.faces_db.get('embeddings', np.array([]))
                        if len(embeddings) > 0:
                            if len(embeddings.shape) == 1 or embeddings.shape[0] == 1:
                                face_db.faces_db['embeddings'] = np.array([])
                            else:
                                face_db.faces_db['embeddings'] = np.delete(embeddings, idx, axis=0)
                        # Remove from person_ids
                        if idx < len(person_ids_list):
                            face_db.faces_db['person_ids'].pop(idx)
                    
                    # Remove from people dictionary
                    del face_db.faces_db['people'][person.id]
                    
                    # Update metadata
                    face_db.faces_db['metadata']['total_faces'] = len(face_db.faces_db.get('embeddings', np.array([])))
                    if isinstance(face_db.faces_db['metadata']['total_faces'], np.ndarray):
                        face_db.faces_db['metadata']['total_faces'] = 0 if face_db.faces_db['metadata']['total_faces'].size == 0 else face_db.faces_db['metadata']['total_faces'].size
                    face_db.faces_db['metadata']['total_people'] = len(face_db.faces_db['people'])
                    
                    # Save face database
                    face_db._save_face_database()
                
                # Delete person from main database
                db.session.delete(person)
                deleted_count += 1
                
            except Exception as e:
                print(f"Error deleting person {person.id} ({person.name}): {e}")
                continue
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Successfully removed {deleted_count} people with no images or videos'
        })
        
    except Exception as e:
        print(f"Error in cleanup_empty_people: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/best_face_for_person', methods=['POST'])
def best_face_for_person():
    """
    Given an image_id and person_id, find the face in the image that best matches the person.
    Returns the face bounding box and similarity score.
    """
    from face_recognition_module import get_face_db
    import numpy as np
    data = request.get_json()
    image_id = data.get('image_id')
    person_id = data.get('person_id')
    if not image_id or not person_id:
        return jsonify({'success': False, 'error': 'Missing image_id or person_id'}), 400

    image = Image.query.get(image_id)
    if not image:
        return jsonify({'success': False, 'error': 'Image not found'}), 404

    # Get image path
    image_path = image.cropped_path if hasattr(image, 'cropped_path') and image.cropped_path and os.path.exists(image.cropped_path) else os.path.join('static', 'uploads', image.filename)
    if not os.path.exists(image_path):
        return jsonify({'success': False, 'error': 'Image file not found'}), 404

    face_db = get_face_db()
    faces = face_db.app_insightface.get(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    if not faces:
        return jsonify({'success': False, 'error': 'No faces detected'}), 404

    # Get person's embeddings from the DB (handle int/str keys)
    person_data = face_db.faces_db['people'].get(str(person_id)) or face_db.faces_db['people'].get(int(person_id))
    if not person_data or not person_data['embeddings']:
        return jsonify({'success': False, 'error': 'No embeddings for person'}), 404

    best_score = -1
    best_face = None
    for face in faces:
        # Compare this face's embedding to all of the person's embeddings, take the max similarity
        similarities = [float(np.dot(face.embedding, np.array(e)) / (np.linalg.norm(face.embedding) * np.linalg.norm(e))) for e in person_data['embeddings']]
        score = max(similarities)
        if score > best_score:
            best_score = score
            best_face = face

    if not best_face:
        return jsonify({'success': False, 'error': 'No matching face found'}), 404

    bbox = best_face.bbox.astype(int)
    return jsonify({
        'success': True,
        'face': {
            'x': int(bbox[0]),
            'y': int(bbox[1]),
            'width': int(bbox[2] - bbox[0]),
            'height': int(bbox[3] - bbox[1]),
            'confidence': float(best_score)
        }
    })

@app.route('/admin/fix_missing_dimensions', methods=['POST'])
def admin_fix_missing_dimensions():
    from models import Image, db
    from PIL import Image as PILImage
    import os
    fixed = 0
    failed = 0
    images = Image.query.filter((Image.width == None) | (Image.height == None) | (Image.width == 0) | (Image.height == 0)).all()
    for img in images:
        # Try original_path, then cropped_path, then filename in uploads
        file_paths = []
        if img.original_path:
            file_paths.append(img.original_path.replace('/static/', 'static/'))
        if getattr(img, 'cropped_path', None):
            file_paths.append(img.cropped_path.replace('/static/', 'static/'))
        file_paths.append(os.path.join('static', 'uploads', img.filename))
        found = False
        for path in file_paths:
            if os.path.exists(path):
                try:
                    with PILImage.open(path) as im:
                        img.width = im.width
                        img.height = im.height
                        db.session.commit()
                        fixed += 1
                        found = True
                        break
                except Exception as e:
                    print(f"Error fixing dimensions for {img.id}: {e}")
                    failed += 1
                    found = True
                    break
        if not found:
            failed += 1
    return {'success': True, 'fixed': fixed, 'failed': failed, 'total_missing': len(images)}

@app.route('/clothing_tagger')
def clothing_tagger():
    from models import Image
    images = [img.to_dict() for img in Image.query.order_by(Image.id.asc()).all()]
    return render_template('clothing_tagger.html', images=images)

@app.route('/api/image/<int:image_id>/clothing_tags', methods=['POST'])
def update_clothing_tags(image_id):
    from models import Image, db
    import json
    image = Image.query.get_or_404(image_id)
    data = request.get_json()
    tags = data.get('tags', [])
    # Only update the tags field
    image.tags = json.dumps(tags)
    db.session.commit()
    return {'success': True, 'tags': tags}

@app.route('/api/detect_clothing/<int:image_id>', methods=['POST'])
def detect_clothing(image_id):
    from models import Image, db
    image = Image.query.get_or_404(image_id)
    file_path = image.original_path.replace('/static/', 'static/')
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'Image file not found'}), 404
    from yolo_detector import get_yolo_detector
    detector = get_yolo_detector()
    clothing_items = detector.detect_clothing(file_path)
    tags = list({item['class_name'] for item in clothing_items})
    if not tags:
        tags = []
    import json
    image.tags = json.dumps(tags)
    db.session.commit()
    return jsonify({'success': True, 'tags': tags, 'detections': clothing_items})

@app.route('/api/detect_clothing/batch', methods=['POST'])
def detect_clothing_batch():
    from models import Image, db
    import json
    from yolo_detector import get_yolo_detector
    detector = get_yolo_detector()
    images = Image.query.all()
    results = []
    for image in images:
        file_path = image.original_path.replace('/static/', 'static/')
        if not os.path.exists(file_path):
            continue
        clothing_items = detector.detect_clothing(file_path)
        tags = list({item['class_name'] for item in clothing_items})
        image.tags = json.dumps(tags)
        results.append({'image_id': image.id, 'tags': tags, 'detections': clothing_items})
    db.session.commit()
    return jsonify({'success': True, 'results': results})

@app.route('/api/detect_clothing/batch_fast', methods=['POST'])
def detect_clothing_batch_fast():
    from models import Image, db
    import json
    from yolo_detector import get_yolo_detector
    detector = get_yolo_detector()
    images = Image.query.all()
    log_file = 'clothing_batch.log'
    def log(msg):
        with open(log_file, 'a') as f:
            f.write(f"{msg}\n")
    start_time = time.time()
    log(f"=== Batch clothing tagging started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    # Only process images with no tags or only the secret tag
    def needs_processing(img):
        tags = img.tags
        if isinstance(tags, str):
            try: tags = json.loads(tags)
            except: tags = []
        if not isinstance(tags, list): tags = [];
        return len(tags) == 0 or all(t.startswith('__no_clothing_detected__') for t in tags)
    images_to_process = [img for img in images if needs_processing(img)]
    results = []
    def process_image(img):
        t0 = time.time()
        file_path = img.original_path.replace('/static/', 'static/')
        thread_id = threading.get_ident()
        if not os.path.exists(file_path):
            log(f"[Thread {thread_id}] Image {img.id}: file not found ({file_path})")
            return (img.id, [], [])
        clothing_items = detector.detect_clothing(file_path)
        tags = list({item['class_name'] for item in clothing_items})
        if not tags:
            tags = ['__no_clothing_detected__']
        img.tags = json.dumps(tags)
        elapsed = int((time.time() - t0) * 1000)
        log(f"[Thread {thread_id}] Image {img.id}: processed in {elapsed} ms, tags: {tags}")
        return (img.id, tags, clothing_items)
    max_workers = multiprocessing.cpu_count()
    log(f"Using {max_workers} workers for {len(images_to_process)} images.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {executor.submit(process_image, img): img for img in images_to_process}
        for future in as_completed(future_to_img):
            img = future_to_img[future]
            try:
                img_id, tags, clothing_items = future.result()
                results.append({'image_id': img_id, 'tags': tags, 'detections': clothing_items})
            except Exception as e:
                log(f"[Thread {threading.get_ident()}] Image {img.id}: ERROR {e}")
                results.append({'image_id': img.id, 'tags': [], 'detections': [], 'error': str(e)})
    db.session.commit()
    total_time = int((time.time() - start_time) * 1000)
    log(f"=== Batch finished in {total_time} ms, processed {len(images_to_process)} images ===\n")
    return jsonify({'success': True, 'processed': len(images_to_process), 'results': results})

@app.route('/api/detect_clothing/batch_progress', methods=['GET'])
def detect_clothing_batch_progress():
    from models import Image
    import json
    images = Image.query.all()
    total = len(images)
    processed = 0
    for img in images:
        tags = img.tags
        if isinstance(tags, str):
            try: tags = json.loads(tags)
            except: tags = []
        if not isinstance(tags, list): tags = []
        # Consider processed if tags is not empty and not just the secret tag
        if len(tags) > 0 and not all(t.startswith('__no_clothing_detected__') for t in tags):
            processed += 1
    return jsonify({'success': True, 'processed': processed, 'total': total})

@app.route('/api/detect_clothing/clear_tags', methods=['POST'])
def clear_clothing_tags():
    from models import Image, db
    images = Image.query.all()
    for img in images:
        img.tags = '[]'
    db.session.commit()
    return jsonify({'success': True, 'message': 'All clothing tags cleared.'})

@app.route('/api/detect_clothing/set_model', methods=['POST'])
def set_clothing_model():
    from yolo_detector import YOLODetector
    data = request.get_json()
    model_url = data.get('model_url')
    model_path = data.get('model_path')
    if not model_url or not model_path:
        return jsonify({'success': False, 'error': 'model_url and model_path required'}), 400
    YOLODetector.set_clothing_model(model_path, model_url)
    # Clear all tags after changing model
    from models import Image, db
    images = Image.query.all()
    for img in images:
        img.tags = '[]'
    db.session.commit()
    return jsonify({'success': True, 'message': 'Clothing model updated and all tags cleared.'})

@app.route('/color_editor/<int:image_id>')
def color_editor(image_id):
    """Color editor page for adjusting image colors"""
    image = Image.query.get_or_404(image_id)
    return render_template('color_editor.html', image=image.to_dict())

@app.route('/api/apply_color_edits/<int:image_id>', methods=['POST'])
def apply_color_edits(image_id):
    """Apply color edits to an image"""
    try:
        image = Image.query.get_or_404(image_id)
        data = request.get_json()
        
        # Get color adjustment parameters
        hue = data.get('hue', 0)
        saturation = data.get('saturation', 100)
        brightness = data.get('brightness', 100)
        contrast = data.get('contrast', 100)
        gamma = data.get('gamma', 1.0)
        shadows = data.get('shadows', 0)
        highlights = data.get('highlights', 0)
        sharpness = data.get('sharpness', 0)
        vibrance = data.get('vibrance', 0)
        clarity = data.get('clarity', 0)
        temperature = data.get('temperature', 0)
        
        # Process the image with color adjustments
        from PIL import Image, ImageEnhance, ImageOps, ImageFilter
        import numpy as np
        
        # Load the image
        if image.original_path:
            image_path = image.original_path.replace('/static/', 'static/')
        else:
            image_path = os.path.join('static', 'uploads', image.filename)
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Image file not found'}), 404
        
        # Open and process the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array for advanced processing
            img_array = np.array(img)
            
            # Apply color adjustments
            if hue != 0:
                # Convert to HSV, adjust hue, convert back
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0] + hue) % 180
                img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Apply saturation
            if saturation != 100:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (saturation / 100.0), 0, 255)
                img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Apply brightness
            if brightness != 100:
                img_array = np.clip(img_array * (brightness / 100.0), 0, 255).astype(np.uint8)
            
            # Apply contrast
            if contrast != 100:
                factor = contrast / 100.0
                offset = 128 * (1 - factor)
                img_array = np.clip(img_array * factor + offset, 0, 255).astype(np.uint8)
            
            # Apply gamma correction
            if gamma != 1.0:
                img_array = np.power(img_array / 255.0, gamma) * 255.0
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            # Apply shadows boost
            if shadows != 0:
                # Calculate luminance
                luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
                shadow_mask = luminance < 128
                
                if np.any(shadow_mask):
                    shadow_factor = 1 + (shadows / 100.0) * (1 - luminance[shadow_mask] / 128.0)
                    shadow_factor = np.clip(shadow_factor, 1, 2)  # Limit boost to 2x
                    
                    for c in range(3):
                        img_array[shadow_mask, c] = np.clip(
                            img_array[shadow_mask, c] * shadow_factor, 0, 255
                        ).astype(np.uint8)
            
            # Apply highlights dampen
            if highlights != 0:
                # Calculate luminance
                luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
                highlight_mask = luminance > 200
                
                if np.any(highlight_mask):
                    highlight_factor = 1 - (highlights / 100.0) * ((luminance[highlight_mask] - 200) / 55.0)
                    highlight_factor = np.clip(highlight_factor, 0.5, 1)  # Limit dampen to 50%
                    
                    for c in range(3):
                        img_array[highlight_mask, c] = np.clip(
                            img_array[highlight_mask, c] * highlight_factor, 0, 255
                        ).astype(np.uint8)
            
            # Apply sharpness
            if sharpness != 0:
                # Convert back to PIL for sharpening
                temp_img = Image.fromarray(img_array)
                sharpness_factor = sharpness / 100.0
                
                # Apply unsharp mask
                if sharpness_factor > 0:
                    # Create a blurred version
                    blurred = temp_img.filter(ImageFilter.GaussianBlur(radius=1))
                    # Create the sharpened image
                    sharpened = ImageEnhance.Sharpness(temp_img).enhance(1 + sharpness_factor)
                    # Blend the sharpened image with the original
                    temp_img = Image.blend(temp_img, sharpened, sharpness_factor * 0.5)
                
                img_array = np.array(temp_img)
            
            # Apply vibrance (selective saturation boost)
            if vibrance != 0:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                vibrance_factor = vibrance / 100.0
                
                # Boost saturation more for less saturated pixels
                saturation_boost = 1 + vibrance_factor * (1 - hsv[:, :, 1] / 255.0)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
                
                img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            # Apply clarity (local contrast enhancement)
            if clarity != 0:
                clarity_factor = clarity / 100.0
                
                # Calculate luminance
                luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
                
                if clarity_factor > 0:
                    # Increase contrast in mid-tones
                    mid_tone_mask = (luminance > 64) & (luminance < 192)
                    if np.any(mid_tone_mask):
                        contrast_boost = 1 + clarity_factor * 0.5
                        for c in range(3):
                            img_array[mid_tone_mask, c] = np.clip(
                                img_array[mid_tone_mask, c] * contrast_boost, 0, 255
                            ).astype(np.uint8)
                else:
                    # Decrease contrast for negative clarity
                    soft_factor = 1 + clarity_factor * 0.3
                    for c in range(3):
                        img_array[:, :, c] = np.clip(
                            img_array[:, :, c] * soft_factor, 0, 255
                        ).astype(np.uint8)
            
            # Apply temperature (color temperature adjustment)
            if temperature != 0:
                temp_factor = temperature / 100.0
                
                if temp_factor > 0:
                    # Warmer (more red/yellow)
                    img_array[:, :, 0] = np.clip(img_array[:, :, 0] + temp_factor * 20, 0, 255).astype(np.uint8)
                    img_array[:, :, 1] = np.clip(img_array[:, :, 1] + temp_factor * 10, 0, 255).astype(np.uint8)
                    img_array[:, :, 2] = np.clip(img_array[:, :, 2] - temp_factor * 15, 0, 255).astype(np.uint8)
                else:
                    # Cooler (more blue)
                    img_array[:, :, 0] = np.clip(img_array[:, :, 0] + temp_factor * 15, 0, 255).astype(np.uint8)
                    img_array[:, :, 1] = np.clip(img_array[:, :, 1] + temp_factor * 5, 0, 255).astype(np.uint8)
                    img_array[:, :, 2] = np.clip(img_array[:, :, 2] - temp_factor * 20, 0, 255).astype(np.uint8)
            
            # Convert back to PIL Image
            img = Image.fromarray(img_array)
            
            # Save the edited image
            edited_filename = f"edited_{image.filename}"
            edited_path = os.path.join('static', 'uploads', edited_filename)
            img.save(edited_path, quality=95)
            
            # Update the image record
            image.filename = edited_filename
            db.session.commit()
            
            return jsonify({
                'success': True,
                'edited_image_url': f'/static/uploads/{edited_filename}',
                'message': 'Color edits applied successfully'
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Chromecast API endpoints
def manual_chromecast_scan():
    """Manual network scan for Chromecast devices using common ports and hostnames"""
    devices = []
    
    try:
        import socket
        import threading
        import time
        
        # Common Chromecast ports
        ports = [8009, 8008, 8007]
        
        # Get local network info
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        network_prefix = '.'.join(local_ip.split('.')[:-1])
        
        def scan_host(ip):
            for port in ports:
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(1)
                    result = sock.connect_ex((ip, port))
                    sock.close()
                    
                    if result == 0:
                        # Port is open, try to connect as Chromecast
                        try:
                            import pychromecast
                            # Handle different pychromecast versions
                            try:
                                # Try newer version constructor with timeout parameter
                                chromecast = pychromecast.Chromecast(ip, port, timeout=2)
                            except TypeError:
                                # Fallback to older version constructor
                                chromecast = pychromecast.Chromecast(ip, port)
                                chromecast.wait(timeout=2)
                            
                            # Create a mock device object
                            class MockDevice:
                                def __init__(self, ip, port, name):
                                    self.host = ip
                                    self.port = port
                                    self.name = name
                                    self.uuid = f"{ip}_{port}"
                                    self.model_name = "Chromecast"
                                    self.cast_type = "cast"
                            
                            device = MockDevice(ip, port, chromecast.name or f"Chromecast_{ip}")
                            devices.append(device)
                            chromecast.disconnect()
                            
                        except Exception as e:
                            print(f"Failed to connect to {ip}:{port} as Chromecast: {e}")
                            continue
                            
                except Exception as e:
                    continue
        
        # Scan common network ranges
        threads = []
        for i in range(1, 255):
            ip = f"{network_prefix}.{i}"
            thread = threading.Thread(target=scan_host, args=(ip,))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            
            # Limit concurrent threads
            if len(threads) >= 50:
                for t in threads:
                    t.join()
                threads = []
        
        # Wait for remaining threads
        for t in threads:
            t.join()
            
        return devices
        
    except Exception as e:
        print(f"Error in manual_chromecast_scan: {e}")
        return []

def mdns_chromecast_scan():
    """Try to discover Chromecast devices using mDNS/zeroconf"""
    devices = []
    
    try:
        # Try using zeroconf if available
        import zeroconf
        from zeroconf import ServiceBrowser, Zeroconf
        
        class ChromecastListener:
            def __init__(self):
                self.devices = []
            
            def add_service(self, zeroconf, type, name):
                info = zeroconf.get_service_info(type, name)
                if info and info.port:
                    try:
                        # Create a mock device object
                        class MockDevice:
                            def __init__(self, host, port, name):
                                self.host = host
                                self.port = port
                                self.name = name
                                self.uuid = f"{host}_{port}"
                                self.model_name = "Chromecast"
                                self.cast_type = "cast"
                        
                        device = MockDevice(info.parsed_addresses()[0], info.port, name)
                        self.devices.append(device)
                        print(f"Found device via mDNS: {device.host}:{device.port} - {device.name}")
                    except Exception as e:
                        print(f"Error processing mDNS service: {e}")
            
            def remove_service(self, zeroconf, type, name):
                pass
            
            def update_service(self, zeroconf, type, name):
                pass
        
        listener = ChromecastListener()
        zeroconf_instance = Zeroconf()
        browser = ServiceBrowser(zeroconf_instance, "_googlecast._tcp.local.", listener)
        
        # Wait for discovery
        import time
        time.sleep(5)
        
        zeroconf_instance.close()
        devices = listener.devices
        
    except ImportError:
        print("zeroconf not available, skipping mDNS discovery")
    except Exception as e:
        print(f"Error in mdns_chromecast_scan: {e}")
    
    return devices

def simple_network_scan():
    """Simple network scan for common Chromecast IPs"""
    devices = []
    
    try:
        import socket
        
        # Get local network info
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        network_prefix = '.'.join(local_ip.split('.')[:-1])
        
        print(f"Local network: {network_prefix}.0/24")
        
        # Try common Chromecast IPs and ranges
        common_ips = [
            f"{network_prefix}.1",   # Router
            f"{network_prefix}.2",   # Common device
            f"{network_prefix}.10",  # Common device
            f"{network_prefix}.20",  # Common device
            f"{network_prefix}.50",  # Common device
            f"{network_prefix}.100", # Common device
            f"{network_prefix}.200", # Common device
            f"{network_prefix}.254"  # Common device
        ]
        
        for ip in common_ips:
            try:
                print(f"Trying to connect to {ip}:8009...")
                import pychromecast
                # Handle different pychromecast versions
                try:
                    # Try newer version constructor with timeout parameter
                    chromecast = pychromecast.Chromecast(ip, 8009, timeout=3)
                except TypeError:
                    # Fallback to older version constructor
                    chromecast = pychromecast.Chromecast(ip, 8009)
                    chromecast.wait(timeout=3)
                
                # Create a mock device object
                class MockDevice:
                    def __init__(self, ip, port, name):
                        self.host = ip
                        self.port = port
                        self.name = name
                        self.uuid = f"{ip}_{port}"
                        self.model_name = "Chromecast"
                        self.cast_type = "cast"
                
                device_name = chromecast.name or f"Chromecast_{ip}"
                device = MockDevice(ip, 8009, device_name)
                devices.append(device)
                print(f"Found Chromecast at {ip}: {device_name}")
                chromecast.disconnect()
                
            except Exception as e:
                print(f"No Chromecast at {ip}: {e}")
                continue
        
        # Also try a few more IPs in the range
        for i in range(3, 20):
            ip = f"{network_prefix}.{i}"
            if ip not in common_ips:
                try:
                    print(f"Trying to connect to {ip}:8009...")
                    import pychromecast
                    # Handle different pychromecast versions
                    try:
                        # Try newer version constructor with timeout parameter
                        chromecast = pychromecast.Chromecast(ip, 8009, timeout=2)
                    except TypeError:
                        # Fallback to older version constructor
                        chromecast = pychromecast.Chromecast(ip, 8009)
                        chromecast.wait(timeout=2)
                    
                    # Create a mock device object
                    class MockDevice:
                        def __init__(self, ip, port, name):
                            self.host = ip
                            self.port = port
                            self.name = name
                            self.uuid = f"{ip}_{port}"
                            self.model_name = "Chromecast"
                            self.cast_type = "cast"
                    
                    device_name = chromecast.name or f"Chromecast_{ip}"
                    device = MockDevice(ip, 8009, device_name)
                    devices.append(device)
                    print(f"Found Chromecast at {ip}: {device_name}")
                    chromecast.disconnect()
                    
                except Exception as e:
                    continue
                    
    except Exception as e:
        print(f"Simple network scan error: {e}")
    
    return devices

def find_device_by_id(device_id):
    """Find a Chromecast device by ID using the same robust discovery methods"""
    try:
        # Try multiple discovery methods for better compatibility
        devices = []
        
        try:
            # Method 1: Try the newer CastBrowser with proper listener
            from pychromecast.discovery import CastBrowser
            from pychromecast.controllers.media import MediaController
            
            class SimpleCastListener:
                def __init__(self):
                    self.devices = []
                
                def add_cast(self, uuid, service):
                    print(f"CastBrowser found device: {service}")
                    self.devices.append(service)
                
                def remove_cast(self, uuid, service, cast_info):
                    pass
                
                def update_cast(self, uuid, service):
                    pass
            
            print("Starting CastBrowser discovery for device lookup...")
            listener = SimpleCastListener()
            browser = CastBrowser(listener)
            browser.start_discovery()
            
            # Wait for discovery
            import time
            time.sleep(5)
            browser.stop_discovery()
            
            devices = listener.devices
            print(f"CastBrowser found {len(devices)} devices for lookup")
            
        except Exception as e:
            print(f"CastBrowser discovery failed for lookup: {e}")
            try:
                # Method 2: Fallback to discover_chromecasts
                print("Trying discover_chromecasts fallback for lookup...")
                from pychromecast.discovery import discover_chromecasts
                devices = discover_chromecasts(timeout=5)
                print(f"discover_chromecasts found {len(devices)} devices for lookup")
            except Exception as e2:
                print(f"discover_chromecasts also failed for lookup: {e2}")
                devices = []
        
        # Method 3: Try manual network scanning if no devices found
        if not devices:
            try:
                print("Trying manual network scan for lookup...")
                devices = manual_chromecast_scan()
            except Exception as e3:
                print(f"Manual scan failed for lookup: {e3}")
                devices = []
        
        # Method 4: Try to find devices using mDNS if available
        if not devices:
            try:
                print("Trying mDNS discovery for lookup...")
                devices = mdns_chromecast_scan()
            except Exception as e4:
                print(f"mDNS scan failed for lookup: {e4}")
                devices = []
        
        # If no devices found, try a simple network scan for common Chromecast IPs
        if not devices:
            print("No devices found for lookup, trying simple network scan...")
            devices = simple_network_scan()
            print(f"Simple scan found {len(devices)} devices for lookup")
        
        print(f"Total devices found for lookup: {len(devices)}")
        
        # Find the target device by ID
        target_device = None
        for device in devices:
            try:
                # Handle both old and new device formats
                if hasattr(device, 'uuid'):
                    device_uuid = str(device.uuid)
                elif hasattr(device, 'host'):
                    # Generate a unique ID from host and port
                    device_uuid = f"{device.host}_{device.port}"
                else:
                    continue
                
                if device_uuid == device_id:
                    # Convert to our standard format
                    device_info = {
                        'id': device_uuid,
                        'host': str(device.host),
                        'port': int(device.port) if hasattr(device, 'port') else 8009,
                        'name': str(device.name) if hasattr(device, 'name') else 'Unknown Device',
                        'model_name': str(device.model_name) if hasattr(device, 'model_name') else 'Unknown',
                        'uuid': device_uuid,
                        'cast_type': str(device.cast_type) if hasattr(device, 'cast_type') else 'Unknown',
                        'is_connected': True
                    }
                    target_device = device_info
                    print(f"Found target device: {target_device}")
                    break
                    
            except Exception as e:
                print(f"Error processing device during lookup: {e}")
                continue
        
        return target_device
        
    except Exception as e:
        print(f"Error in find_device_by_id: {e}")
        return None

def get_image_mime_type(image_path):
    """Determine the correct MIME type for an image based on its extension"""
    import os
    extension = os.path.splitext(image_path)[1].lower()
    
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    
    return mime_types.get(extension, 'image/jpeg')  # Default to JPEG

def create_chromecast_connection(device_info, timeout=10):
    """Create a Chromecast connection with enhanced error handling"""
    try:
        # Handle different pychromecast versions
        try:
            # Try newer version constructor with timeout parameter
            chromecast = pychromecast.Chromecast(device_info['host'], device_info['port'], timeout=timeout)
        except TypeError:
            # Fallback to older version constructor
            chromecast = pychromecast.Chromecast(device_info['host'], device_info['port'])
            chromecast.wait(timeout=timeout)
        
        # Test if the connection is actually working
        if not chromecast.is_idle:
            print(f"Chromecast {chromecast.name} is not idle, current app: {chromecast.app_id}")
        
        return chromecast, None
        
    except pychromecast.error.UnsupportedNamespace:
        error_msg = f"Unsupported namespace on device {device_info['name']}"
        print(error_msg)
        return None, error_msg
    except pychromecast.error.ChromecastConnectionError:
        error_msg = f"Connection failed to {device_info['name']} at {device_info['host']}:{device_info['port']}"
        print(error_msg)
        return None, error_msg
    except pychromecast.error.NoChromecastFoundError:
        error_msg = f"No Chromecast found at {device_info['host']}:{device_info['port']}"
        print(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Unexpected error connecting to {device_info['name']}: {str(e)}"
        print(error_msg)
        return None, error_msg

@app.route('/api/chromecast/status/<device_id>')
def get_chromecast_status(device_id):
    """Get the current status of a Chromecast device"""
    try:
        target_device = find_device_by_id(device_id)
        
        if not target_device:
            return jsonify({
                'success': False,
                'error': 'Device not found'
            }), 404
        
        # Try to create a connection to get status
        try:
            chromecast, error = create_chromecast_connection(target_device, timeout=5)
            
            if chromecast is None:
                return jsonify({
                    'success': False,
                    'device': target_device,
                    'is_online': False,
                    'error': error
                })
            
            status = {
                'success': True,
                'device': target_device,
                'is_online': True,
                'device_name': chromecast.name,
                'app_id': chromecast.app_id,
                'is_idle': chromecast.is_idle,
                'cast_type': chromecast.cast_type,
                'model_name': chromecast.model_name,
                'volume_level': getattr(chromecast, 'volume_level', None),
                'volume_muted': getattr(chromecast, 'volume_muted', None)
            }
            
            chromecast.disconnect()
            return jsonify(status)
            
        except Exception as conn_error:
            return jsonify({
                'success': False,
                'device': target_device,
                'is_online': False,
                'error': f'Connection failed: {str(conn_error)}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chromecast/refresh', methods=['POST'])
def refresh_chromecast_devices():
    """Force refresh of Chromecast device list"""
    try:
        # Clear any cached device lists and rediscover
        devices = []
        
        # Try multiple discovery methods for better compatibility
        try:
            # Method 1: Try the newer CastBrowser with proper listener
            from pychromecast.discovery import CastBrowser
            from pychromecast.controllers.media import MediaController
            
            class SimpleCastListener:
                def __init__(self):
                    self.devices = []
                
                def add_cast(self, uuid, service):
                    print(f"CastBrowser found device during refresh: {service}")
                    self.devices.append(service)
                
                def remove_cast(self, uuid, service, cast_info):
                    pass
                
                def update_cast(self, uuid, service):
                    pass
            
            print("Starting CastBrowser discovery for refresh...")
            listener = SimpleCastListener()
            browser = CastBrowser(listener)
            browser.start_discovery()
            
            # Wait for discovery
            import time
            time.sleep(8)  # Longer wait for refresh
            browser.stop_discovery()
            
            devices = listener.devices
            print(f"CastBrowser found {len(devices)} devices during refresh")
            
        except Exception as e:
            print(f"CastBrowser discovery failed during refresh: {e}")
            try:
                # Method 2: Fallback to discover_chromecasts
                print("Trying discover_chromecasts fallback for refresh...")
                from pychromecast.discovery import discover_chromecasts
                devices = discover_chromecasts(timeout=8)
                print(f"discover_chromecasts found {len(devices)} devices during refresh")
            except Exception as e2:
                print(f"discover_chromecasts also failed during refresh: {e2}")
                devices = []
        
        # Method 3: Try manual network scanning if no devices found
        if not devices:
            try:
                print("Trying manual network scan for refresh...")
                devices = manual_chromecast_scan()
            except Exception as e3:
                print(f"Manual scan failed during refresh: {e3}")
                devices = []
        
        # Method 4: Try to find devices using mDNS if available
        if not devices:
            try:
                print("Trying mDNS discovery for refresh...")
                devices = mdns_chromecast_scan()
            except Exception as e4:
                print(f"mDNS scan failed during refresh: {e4}")
                devices = []
        
        # If no devices found, try a simple network scan for common Chromecast IPs
        if not devices:
            print("No devices found during refresh, trying simple network scan...")
            devices = simple_network_scan()
            print(f"Simple scan found {len(devices)} devices during refresh")
        
        device_list = []
        
        print(f"Processing {len(devices)} discovered devices during refresh...")
        for i, device in enumerate(devices):
            try:
                print(f"Processing device {i+1} during refresh: {device}")
                
                # Handle both old and new device formats
                if hasattr(device, 'uuid'):
                    device_uuid = str(device.uuid)
                elif hasattr(device, 'host'):
                    # Generate a unique ID from host and port
                    device_uuid = f"{device.host}_{device.port}"
                else:
                    continue
                
                device_info = {
                    'id': device_uuid,
                    'host': str(device.host),
                    'port': int(device.port) if hasattr(device, 'port') else 8009,
                    'name': str(device.name) if hasattr(device, 'name') else 'Unknown Device',
                    'model_name': str(device.model_name) if hasattr(device, 'model_name') else 'Unknown',
                    'uuid': device_uuid,
                    'cast_type': str(device.cast_type) if hasattr(device, 'cast_type') else 'Unknown',
                    'is_connected': True  # Assume devices are available if discovered
                }
                device_list.append(device_info)
                print(f"Added device during refresh: {device_info}")
            except AttributeError as attr_error:
                # Skip devices with missing attributes
                print(f"Skipping device due to missing attributes during refresh: {attr_error}")
                continue
            except Exception as e:
                print(f"Error processing device during refresh: {e}")
                continue
        
        print(f"Successfully processed {len(device_list)} devices during refresh")
        return jsonify({
            'success': True,
            'devices': device_list,
            'message': f'Found {len(device_list)} Chromecast devices'
        })
        
    except Exception as e:
        print(f"Error in refresh_chromecast_devices: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chromecast/devices')
def get_chromecast_devices():
    """Get list of available Chromecast devices on the network"""
    try:
        # Try multiple discovery methods for better compatibility
        devices = []
        
        try:
            # Method 1: Try the newer CastBrowser with proper listener
            from pychromecast.discovery import CastBrowser
            from pychromecast.controllers.media import MediaController
            
            class SimpleCastListener:
                def __init__(self):
                    self.devices = []
                
                def add_cast(self, uuid, service):
                    print(f"CastBrowser found device: {service}")
                    self.devices.append(service)
                
                def remove_cast(self, uuid, service, cast_info):
                    pass
                
                def update_cast(self, uuid, service):
                    pass
            
            print("Starting CastBrowser discovery...")
            listener = SimpleCastListener()
            browser = CastBrowser(listener)
            browser.start_discovery()
            
            # Wait for discovery
            import time
            time.sleep(5)
            browser.stop_discovery()
            
            devices = listener.devices
            print(f"CastBrowser found {len(devices)} devices")
            
        except Exception as e:
            print(f"CastBrowser discovery failed: {e}")
            try:
                # Method 2: Fallback to discover_chromecasts
                print("Trying discover_chromecasts fallback...")
                from pychromecast.discovery import discover_chromecasts
                devices = discover_chromecasts(timeout=5)
                print(f"discover_chromecasts found {len(devices)} devices")
            except Exception as e2:
                print(f"discover_chromecasts also failed: {e2}")
                devices = []
        
        # Method 3: Try manual network scanning if no devices found
        if not devices:
            try:
                print("Trying manual network scan...")
                devices = manual_chromecast_scan()
            except Exception as e3:
                print(f"Manual scan failed: {e3}")
                devices = []
        
        # Method 4: Try to find devices using mDNS if available
        if not devices:
            try:
                print("Trying mDNS discovery...")
                devices = mdns_chromecast_scan()
            except Exception as e4:
                print(f"mDNS scan failed: {e4}")
                devices = []
        
        print(f"Total devices found across all methods: {len(devices)}")
        
        # If no devices found, try a simple network scan for common Chromecast IPs
        if not devices:
            print("No devices found, trying simple network scan...")
            devices = simple_network_scan()
            print(f"Simple scan found {len(devices)} devices")
        
        device_list = []
        
        print(f"Processing {len(devices)} discovered devices...")
        for i, device in enumerate(devices):
            try:
                print(f"Processing device {i+1}: {device}")
                print(f"Device attributes: {dir(device)}")
                
                # Handle both old and new device formats
                if hasattr(device, 'uuid'):
                    device_uuid = str(device.uuid)
                    print(f"Using uuid: {device_uuid}")
                elif hasattr(device, 'host'):
                    # Generate a unique ID from host and port
                    device_uuid = f"{device.host}_{device.port}"
                    print(f"Generated uuid from host/port: {device_uuid}")
                else:
                    print(f"Device missing both uuid and host attributes, skipping")
                    continue
                
                device_info = {
                    'id': device_uuid,
                    'host': str(device.host),
                    'port': int(device.port) if hasattr(device, 'port') else 8009,
                    'name': str(device.name) if hasattr(device, 'name') else 'Unknown Device',
                    'model_name': str(device.model_name) if hasattr(device, 'model_name') else 'Unknown',
                    'uuid': device_uuid,
                    'cast_type': str(device.cast_type) if hasattr(device, 'cast_type') else 'Unknown',
                    'is_connected': True  # Assume devices are available if discovered
                }
                device_list.append(device_info)
                print(f"Added device: {device_info}")
                
            except AttributeError as attr_error:
                # Skip devices with missing attributes
                print(f"Skipping device due to missing attributes: {attr_error}")
                continue
            except Exception as e:
                print(f"Error processing device: {e}")
                continue
        
        print(f"Successfully processed {len(device_list)} devices")
        return jsonify({
            'success': True,
            'devices': device_list,
            'message': f'Found {len(device_list)} Chromecast devices'
        })
        
    except Exception as e:
        print(f"Error in get_chromecast_devices: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chromecast/test_connection', methods=['POST'])
def test_chromecast_connection():
    """Test connection to a Chromecast device"""
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        
        if not device_id:
            return jsonify({
                'success': False,
                'error': 'Device ID is required'
            }), 400
        
        # Find the device using our robust discovery method
        target_device = find_device_by_id(device_id)
        
        if not target_device:
            return jsonify({
                'success': False,
                'error': 'Device not found'
            }), 404
        
        # Try to create a connection to test if device is reachable
        try:
            chromecast, error = create_chromecast_connection(target_device, timeout=5)
            
            if chromecast is None:
                return jsonify({
                    'success': False,
                    'error': error,
                    'is_reachable': False
                })
            
            return jsonify({
                'success': True,
                'message': f'Successfully connected to {chromecast.name}',
                'device_name': chromecast.name,
                'is_reachable': True
            })
        except Exception as conn_error:
            return jsonify({
                'success': False,
                'error': f'Connection failed: {str(conn_error)}',
                'is_reachable': False
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chromecast/cast', methods=['POST'])
def cast_to_chromecast():
    """Cast a single image to Chromecast"""
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        image_url = data.get('image_url')
        image_id = data.get('image_id')
        
        if not device_id or not image_url:
            return jsonify({
                'success': False,
                'error': 'Device ID and image URL are required'
            }), 400
        
        # Find the device using our robust discovery method
        target_device = find_device_by_id(device_id)
        
        if not target_device:
            return jsonify({
                'success': False,
                'error': 'Device not found'
            }), 404
        
        # Create Chromecast object with enhanced error handling
        chromecast, error = create_chromecast_connection(target_device)
        
        if chromecast is None:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        # Get the media controller and play the image
        mc = chromecast.media_controller
        
        # Convert relative URL to absolute URL if needed
        if image_url.startswith('/'):
            # Get the base URL from the request
            base_url = request.url_root.rstrip('/')
            full_image_url = base_url + image_url
        else:
            full_image_url = image_url
        
        # Determine the correct MIME type
        mime_type = get_image_mime_type(image_url)
        
        # Play the media with proper MIME type
        mc.play_media(
            full_image_url,
            mime_type
        )
        
        return jsonify({
            'success': True,
            'message': f'Casting image to {chromecast.name}',
            'device_name': chromecast.name,
            'mime_type': mime_type
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chromecast/slideshow', methods=['POST'])
def start_chromecast_slideshow():
    """Start a slideshow on Chromecast with multiple images"""
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        speed = data.get('speed', 5)
        image_ids = data.get('image_ids', [])
        
        if not device_id or not image_ids:
            return jsonify({
                'success': False,
                'error': 'Device ID and image IDs are required'
            }), 400
        
        # Find the device using our robust discovery method
        target_device = find_device_by_id(device_id)
        
        if not target_device:
            return jsonify({
                'success': False,
                'error': 'Device not found'
            }), 404
        
        # Get image URLs from the database
        from models import Image
        images = Image.query.filter(Image.id.in_(image_ids)).all()
        
        if not images:
            return jsonify({
                'success': False,
                'error': 'No images found'
            }), 404
        
        # Create Chromecast object with enhanced error handling
        chromecast, error = create_chromecast_connection(target_device)
        
        if chromecast is None:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        # Get the media controller
        mc = chromecast.media_controller
        
        # Start with the first image
        first_image = images[0]
        image_url = first_image.image_url
        
        # Convert relative URL to absolute URL if needed
        if image_url.startswith('/'):
            base_url = request.url_root.rstrip('/')
            full_image_url = base_url + image_url
        else:
            full_image_url = image_url
        
        # Determine the correct MIME type
        mime_type = get_image_mime_type(image_url)
        
        # Play the media with proper MIME type
        mc.play_media(
            full_image_url,
            mime_type
        )
        
        return jsonify({
            'success': True,
            'message': f'Started slideshow on {chromecast.name}',
            'device_name': chromecast.name,
            'total_images': len(images),
            'speed': speed,
            'mime_type': mime_type
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chromecast/stop', methods=['POST'])
def stop_chromecast():
    """Stop playback on Chromecast"""
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        
        if not device_id:
            return jsonify({
                'success': False,
                'error': 'Device ID is required'
            }), 400
        
        # Find the device using our robust discovery method
        target_device = find_device_by_id(device_id)
        
        if not target_device:
            return jsonify({
                'success': False,
                'error': 'Device not found'
            }), 404
        
        # Create Chromecast object with enhanced error handling
        chromecast, error = create_chromecast_connection(target_device)
        
        if chromecast is None:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        # Stop playback
        chromecast.media_controller.stop()
        
        return jsonify({
            'success': True,
            'message': f'Stopped playback on {chromecast.name}',
            'device_name': chromecast.name
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chromecast/volume', methods=['POST'])
def set_chromecast_volume():
    """Set volume on Chromecast device"""
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        volume = data.get('volume')  # 0.0 to 1.0
        
        if not device_id or volume is None:
            return jsonify({
                'success': False,
                'error': 'Device ID and volume are required'
            }), 400
        
        if not 0.0 <= volume <= 1.0:
            return jsonify({
                'success': False,
                'error': 'Volume must be between 0.0 and 1.0'
            }), 400
        
        # Find the device using our robust discovery method
        target_device = find_device_by_id(device_id)
        
        if not target_device:
            return jsonify({
                'success': False,
                'error': 'Device not found'
            }), 404
        
        # Create Chromecast object with enhanced error handling
        chromecast, error = create_chromecast_connection(target_device)
        
        if chromecast is None:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        # Set volume
        chromecast.set_volume(volume)
        
        return jsonify({
            'success': True,
            'message': f'Volume set to {int(volume * 100)}% on {chromecast.name}',
            'device_name': chromecast.name,
            'volume': volume
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/chromecast/mute', methods=['POST'])
def toggle_chromecast_mute():
    """Toggle mute state on Chromecast device"""
    try:
        data = request.get_json()
        device_id = data.get('device_id')
        
        if not device_id:
            return jsonify({
                'success': False,
                'error': 'Device ID is required'
            }), 400
        
        # Find the device using our robust discovery method
        target_device = find_device_by_id(device_id)
        
        if not target_device:
            return jsonify({
                'success': False,
                'error': 'Device not found'
            }), 404
        
        # Create Chromecast object with enhanced error handling
        chromecast, error = create_chromecast_connection(target_device)
        
        if chromecast is None:
            return jsonify({
                'success': False,
                'error': error
            }), 500
        
        # Toggle mute
        current_mute_state = chromecast.is_volume_muted
        chromecast.set_volume_muted(not current_mute_state)
        
        return jsonify({
            'success': True,
            'message': f'{"Muted" if not current_mute_state else "Unmuted"} {chromecast.name}',
            'device_name': chromecast.name,
            'muted': not current_mute_state
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/rebuild_face_db')
def rebuild_face_db():
    """Show the rebuild face database page"""
    from face_recognition_module import get_face_db
    
    # Get current database statistics
    face_db = get_face_db()
    stats = face_db.get_database_stats()
    
    return render_template('rebuild_face_db.html', stats=stats)

@app.route('/api/rebuild_face_db', methods=['POST'])
def api_rebuild_face_db():
    """API endpoint to rebuild the face recognition database (threaded version)"""
    from face_recognition_module import get_face_db
    from models import Image, Person
    import json
    import os
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    import time
    
    def send_progress(data):
        """Send progress data as server-sent event"""
        return f"data: {json.dumps(data)}\n\n"
    
    def process_image_batch(batch_data):
        """Process a batch of images in a separate thread"""
        batch_results = []
        
        # Each thread needs its own face_db instance for thread safety
        with app.app_context():
            thread_face_db = get_face_db()
            
            for image_data in batch_data:
                try:
                    image_id, image_filename, image_path, people_names = image_data
                    
                    if image_path and os.path.exists(image_path):
                        # Extract face embedding
                        face_embedding = thread_face_db.extract_face_embedding(image_path)
                        
                        if face_embedding is not None:
                            batch_results.append({
                                'success': True,
                                'image_id': image_id,
                                'image_filename': image_filename,
                                'image_path': image_path,
                                'people_names': people_names,
                                'embedding': face_embedding
                            })
                        else:
                            batch_results.append({
                                'success': False,
                                'image_id': image_id,
                                'image_filename': image_filename,
                                'error': 'Failed to extract face embedding'
                            })
                    else:
                        batch_results.append({
                            'success': False,
                            'image_id': image_id,
                            'image_filename': image_filename,
                            'error': 'Image file not found'
                        })
                        
                except Exception as e:
                    batch_results.append({
                        'success': False,
                        'image_id': image_data[0] if len(image_data) > 0 else 'unknown',
                        'image_filename': image_data[1] if len(image_data) > 1 else 'unknown',
                        'error': str(e)
                    })
        
        return batch_results
    
    def stream_response():
        """Stream the rebuild progress with threading"""
        with app.app_context():
            try:
                face_db = get_face_db()
                
                # Get all images with people assigned
                images_with_people = Image.query.filter(Image.people.any()).all()
                total_images = len(images_with_people)
                
                yield send_progress({
                    'type': 'progress',
                    'processed': 0,
                    'total': total_images,
                    'status': f'Found {total_images} images with named people',
                    'faces_added': 0
                })
                
                yield send_progress({
                    'type': 'log',
                    'message': f'Starting THREADED rebuild of face recognition database with {total_images} images',
                    'level': 'info'
                })
                
                # Backup existing database before clearing
                backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_dir = 'face_database_backups'
                os.makedirs(backup_dir, exist_ok=True)
                
                # Backup the current database files
                backup_files = []
                if os.path.exists('face_database/embeddings.npy'):
                    backup_path = os.path.join(backup_dir, f'embeddings_backup_{backup_timestamp}.npy')
                    shutil.copy2('face_database/embeddings.npy', backup_path)
                    backup_files.append('embeddings.npy')
                
                if os.path.exists('face_database/faces.pkl'):
                    backup_path = os.path.join(backup_dir, f'faces_backup_{backup_timestamp}.pkl')
                    shutil.copy2('face_database/faces.pkl', backup_path)
                    backup_files.append('faces.pkl')
                
                if os.path.exists('face_database/metadata.json'):
                    backup_path = os.path.join(backup_dir, f'metadata_backup_{backup_timestamp}.json')
                    shutil.copy2('face_database/metadata.json', backup_path)
                    backup_files.append('metadata.json')
                
                yield send_progress({
                    'type': 'log',
                    'message': f'Backed up existing database to {backup_dir}/ (files: {", ".join(backup_files)})',
                    'level': 'info'
                })
                
                # Clear existing database
                face_db.faces_db = {
                    'people': {},
                    'embeddings': np.array([]),
                    'person_ids': [],
                    'metadata': {
                        'created_at': datetime.now().isoformat(),
                        'model_name': face_db.model_name,
                        'total_faces': 0,
                        'total_people': 0,
                        'last_updated': datetime.now().isoformat()
                    }
                }
                
                yield send_progress({
                    'type': 'log',
                    'message': 'Cleared existing face recognition database',
                    'level': 'info'
                })
                
                # Prepare image data for threading
                image_batch_data = []
                for image in images_with_people:
                    # Get image path
                    if image.original_path and os.path.exists(image.original_path.replace('/static/', 'static/')):
                        image_path = image.original_path.replace('/static/', 'static/')
                    else:
                        # Search for the actual file with hash prefix
                        base_filename = image.filename
                        uploads_dir = os.path.join(os.getcwd(), 'static', 'uploads')
                        matching_files = []
                        if os.path.exists(uploads_dir):
                            for file in os.listdir(uploads_dir):
                                if base_filename in file:
                                    matching_files.append(file)
                        
                        if matching_files:
                            cropped_files = [f for f in matching_files if f.startswith('cropped_')]
                            if cropped_files:
                                image_path = os.path.join(uploads_dir, cropped_files[0])
                            else:
                                image_path = os.path.join(uploads_dir, matching_files[0])
                        else:
                            image_path = None
                    
                    # Get people names for this image
                    people_names = [person.name.strip() for person in image.people if person.name.strip()]
                    
                    if image_path and people_names:
                        image_batch_data.append((image.id, image.filename, image_path, people_names))
                
                yield send_progress({
                    'type': 'log',
                    'message': f'Prepared {len(image_batch_data)} images for threaded processing',
                    'level': 'info'
                })
                
                # Create batches for threading (batch size of 5 for optimal GPU usage)
                batch_size = 5
                batches = [image_batch_data[i:i + batch_size] for i in range(0, len(image_batch_data), batch_size)]
                
                # Determine optimal thread count (max 4 to avoid overwhelming GPU)
                max_workers = min(4, len(batches), 4)  # Cap at 4 threads for GPU efficiency
                
                yield send_progress({
                    'type': 'log',
                    'message': f'Using {max_workers} worker threads with {len(batches)} batches of {batch_size} images each',
                    'level': 'info'
                })
                
                processed_count = 0
                faces_added = 0
                people_updated = set()
                failed_count = 0
                
                # Thread-safe locks
                progress_lock = threading.Lock()
                
                # Process batches in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all batches for processing
                    future_to_batch = {executor.submit(process_image_batch, batch): batch for batch in batches}
                    
                    # Process completed batches as they finish
                    for future in as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        try:
                            batch_results = future.result()
                            
                            # Process results from this batch
                            for result in batch_results:
                                with progress_lock:
                                    processed_count += 1
                                    
                                    if result['success']:
                                        # Add to face recognition database
                                        for person_name in result['people_names']:
                                            person = Person.query.filter_by(name=person_name).first()
                                            if person:
                                                # Add to face recognition database
                                                if person.id not in face_db.faces_db['people']:
                                                    face_db.faces_db['people'][person.id] = {
                                                        'name': person.name,
                                                        'embeddings': [],
                                                        'image_paths': [],
                                                        'is_confirmed': person.is_confirmed
                                                    }
                                                
                                                # Add embedding to person's collection
                                                face_db.faces_db['people'][person.id]['embeddings'].append(result['embedding'])
                                                face_db.faces_db['people'][person.id]['image_paths'].append(result['image_path'])
                                                
                                                # Add to global embeddings array
                                                if len(face_db.faces_db['embeddings']) == 0:
                                                    face_db.faces_db['embeddings'] = result['embedding'].reshape(1, -1)
                                                else:
                                                    face_db.faces_db['embeddings'] = np.vstack([face_db.faces_db['embeddings'], result['embedding']])
                                                
                                                face_db.faces_db['person_ids'].append(person.id)
                                                faces_added += 1
                                                people_updated.add(person.id)
                                        
                                        yield send_progress({
                                            'type': 'log',
                                            'message': f'✓ Processed {result["image_filename"]} - added {len(result["people_names"])} faces',
                                            'level': 'info'
                                        })
                                    else:
                                        failed_count += 1
                                        yield send_progress({
                                            'type': 'log',
                                            'message': f'✗ Failed {result["image_filename"]}: {result["error"]}',
                                            'level': 'error'
                                        })
                                    
                                    # Update progress every 20 images
                                    if processed_count % 20 == 0 or processed_count == len(image_batch_data):
                                        yield send_progress({
                                            'type': 'progress',
                                            'processed': processed_count,
                                            'total': len(image_batch_data),
                                            'status': f'Processed {processed_count}/{len(image_batch_data)} images ({max_workers} threads)',
                                            'faces_added': faces_added
                                        })
                        
                        except Exception as e:
                            yield send_progress({
                                'type': 'log',
                                'message': f'Batch processing error: {str(e)}',
                                'level': 'error'
                            })
                
                # Update metadata
                face_db.faces_db['metadata'].update({
                    'last_updated': datetime.now().isoformat(),
                    'total_faces': len(face_db.faces_db['embeddings']),
                    'total_people': len(face_db.faces_db['people']),
                    'processed_images': processed_count,
                    'failed_images': failed_count
                })
                
                # Save database
                face_db._save_face_database()
                
                yield send_progress({
                    'type': 'log',
                    'message': f'Saved face recognition database with {faces_added} faces from {len(people_updated)} people (failed: {failed_count})',
                    'level': 'success'
                })
                
                # Send completion
                yield send_progress({
                    'type': 'complete',
                    'processed': processed_count,
                    'total': len(image_batch_data),
                    'faces_added': faces_added,
                    'people_updated': len(people_updated)
                })
                
            except Exception as e:
                yield send_progress({
                    'type': 'error',
                    'message': f'Threaded rebuild failed: {str(e)}'
                })
    
    return Response(stream_response(), mimetype='text/plain')

@app.route('/api/backups/list', methods=['GET'])
def api_list_backups():
    """List available database backups"""
    backup_dir = 'face_database_backups'
    backups = []
    
    if os.path.exists(backup_dir):
        for file in os.listdir(backup_dir):
            if file.endswith('.npy') and 'embeddings_backup_' in file:
                timestamp = file.replace('embeddings_backup_', '').replace('.npy', '')
                backup_info = {
                    'timestamp': timestamp,
                    'date': datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S'),
                    'files': []
                }
                
                # Check for all related files
                for ext in ['.npy', '.pkl', '.json']:
                    backup_file = f"{'embeddings' if ext == '.npy' else 'faces' if ext == '.pkl' else 'metadata'}_backup_{timestamp}{ext}"
                    if os.path.exists(os.path.join(backup_dir, backup_file)):
                        backup_info['files'].append(backup_file)
                
                if backup_info['files']:
                    backups.append(backup_info)
    
    # Sort by timestamp (newest first)
    backups.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify({'backups': backups})

@app.route('/api/backups/restore', methods=['POST'])
def api_restore_backup():
    """Restore database from backup"""
    data = request.get_json()
    timestamp = data.get('timestamp')
    
    if not timestamp:
        return jsonify({'success': False, 'error': 'No timestamp provided'}), 400
    
    backup_dir = 'face_database_backups'
    restored_files = []
    
    try:
        # Restore each file
        for ext, source_name in [('.npy', 'embeddings'), ('.pkl', 'faces'), ('.json', 'metadata')]:
            backup_file = f"{source_name}_backup_{timestamp}{ext}"
            backup_path = os.path.join(backup_dir, backup_file)
            restore_path = f"face_database/{source_name}{ext}"
            
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, restore_path)
                restored_files.append(backup_file)
        
        if restored_files:
            return jsonify({
                'success': True, 
                'message': f'Restored {len(restored_files)} files from backup {timestamp}',
                'restored_files': restored_files
            })
        else:
            return jsonify({'success': False, 'error': 'No backup files found'}), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': f'Restore failed: {str(e)}'}), 500

@app.route('/api/namer/delete_image', methods=['POST'])
def api_namer_delete_image():
    from models import Image, db
    import os, json
    data = request.get_json() or {}
    image_id = data.get('image_id')
    
    if not image_id:
        return jsonify({'success': False, 'error': 'Missing image_id'}), 400
    
    try:
        image = Image.query.get(image_id)
        if not image:
            return jsonify({'success': False, 'error': 'Image not found'}), 404
        
        # Delete the image file
        if image.original_path and os.path.exists(image.original_path.replace('/static/', 'static/')):
            try:
                os.remove(image.original_path.replace('/static/', 'static/'))
            except:
                pass  # File might not exist
        
        # Also try to delete from uploads directory
        uploads_path = os.path.join('static', 'uploads', image.filename)
        if os.path.exists(uploads_path):
            try:
                os.remove(uploads_path)
            except:
                pass
        
        # Delete from database
        db.session.delete(image)
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== VIDEO NAMER API ROUTES =====

@app.route('/api/video-namer/skip', methods=['POST'])
def api_video_namer_skip():
    """Skip a video by adding 'skipped' tag to it."""
    from models import Video, db
    import json
    
    try:
        data = request.get_json() or {}
        video_id = data.get('video_id')
        
        if not video_id:
            return jsonify({'success': False, 'error': 'No video ID provided'}), 400
        
        video = Video.query.get(video_id)
        if not video:
            return jsonify({'success': False, 'error': 'Video not found'}), 404
        
        # Add 'skipped' tag to video
        tags = []
        if video.tags:
            try:
                tags = json.loads(video.tags) if isinstance(video.tags, str) else video.tags
            except (json.JSONDecodeError, TypeError):
                tags = []
        
        if not isinstance(tags, list):
            tags = []
        
        if 'skipped' not in tags:
            tags.append('skipped')
        
        video.tags = json.dumps(tags)
        db.session.commit()
        
        print(f"✓ Video {video_id} marked as skipped")
        return jsonify({'success': True, 'message': 'Video skipped successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/video-namer/next_video', methods=['POST'])
def api_video_namer_next_video():
    """Get the next video to name, excluding skipped ones."""
    from models import Video
    import os, json
    
    data = request.get_json() or {}
    skipped_ids = data.get('skipped_ids', [])  # Keep for backward compatibility
    
    def file_exists(vid):
        if vid.file_path:
            file_path = vid.file_path.replace('/static/', 'static/')
            if os.path.exists(file_path):
                return True
        return False
    
    # Get videos without people associated, excluding skipped ones (by tags)
    videos_to_name = Video.query.filter(
        ~Video.people.any()  # No people associated
    ).order_by(Video.id.desc()).all()
    
    # Filter for file existence and skipped tags
    filtered_videos = []
    for vid in videos_to_name:
        if file_exists(vid):
            # Check if video has 'skipped' tag
            tags = []
            if vid.tags:
                try:
                    tags = json.loads(vid.tags) if isinstance(vid.tags, str) else vid.tags
                except (json.JSONDecodeError, TypeError):
                    tags = []
            
            if not isinstance(tags, list):
                tags = []
            
            # Skip if has 'skipped' tag OR is in old skipped_ids list
            if 'skipped' not in tags and vid.id not in skipped_ids:
                filtered_videos.append(vid)
    
    print(f"Found {len(filtered_videos)} videos to name (excluding skipped)")
    
    # Return first available video
    if filtered_videos:
        print(f"Returning next video: {filtered_videos[0].id}")
        return jsonify({'success': True, 'video': filtered_videos[0].to_dict()})
    
    print("No more videos to name")
    return jsonify({'success': True, 'video': None})

@app.route('/api/video-namer/face_suggestions', methods=['POST'])
@time_function("api.video_namer.face_suggestions")
def api_video_namer_face_suggestions():
    """Get face recognition suggestions for a specific face in a video thumbnail"""
    from face_recognition_module import get_face_db
    try:
        with time_operation("api.video_namer.face_suggestions.parse_request"):
            data = request.get_json()
            video_id = data.get('video_id')
            face_index = data.get('face_index', 0)
            
            if not video_id:
                return jsonify({'success': False, 'error': 'Missing video_id'}), 400
        
        with time_operation("api.video_namer.face_suggestions.query_video"):
            video = Video.query.get(video_id)
            if not video:
                return jsonify({'success': False, 'error': 'Video not found'}), 404
        
        # Use the video thumbnail for face recognition
        thumbnail_path = video.thumbnail_path.replace('/static/', 'static/') if video.thumbnail_path else None
        
        if not thumbnail_path or not os.path.exists(thumbnail_path):
            return jsonify({'success': False, 'error': 'Video thumbnail not found'}), 404
        
        # Get face recognition
        face_db = get_face_db()
        
        # Get suggestions directly from the thumbnail
        with time_operation("api.video_namer.face_suggestions.get_suggestions", face_index=face_index):
            similar_faces = face_db.find_similar_faces(thumbnail_path, threshold=0.0, top_k=20)
        
        # Format suggestions - ensure unique names
        with time_operation("api.video_namer.face_suggestions.format_suggestions"):
            seen_names = set()
            suggestions = []
            for face in similar_faces:
                if face['person_name'] and face['person_name'] not in seen_names:
                    seen_names.add(face['person_name'])
                    suggestions.append({
                        'person_id': face['person_id'],
                        'name': face['person_name'],
                        'confidence': face['similarity'],
                        'is_confirmed': face['is_confirmed']
                    })
                if len(suggestions) == 5:
                    break
        
        return jsonify({'success': True, 'suggestions': suggestions})
        
    except Exception as e:
        print(f"Error getting face suggestions for video {video_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/video-namer/save_multiple', methods=['POST'])
def api_video_namer_save_multiple():
    """Save multiple names for a video."""
    from models import Video, Person, db
    import json
    
    data = request.get_json() or {}
    video_id = data.get('video_id')
    names = data.get('names', [])
    
    if not video_id or not names:
        return jsonify({'success': False, 'error': 'Missing video_id or names'}), 400
    
    try:
        video = Video.query.get(video_id)
        if not video:
            return jsonify({'success': False, 'error': 'Video not found'}), 404
        
        # Process each name
        for name in names:
            if not name.strip():
                continue
                
            # Find or create person
            person = Person.query.filter_by(name=name.strip()).first()
            if not person:
                person = Person(name=name.strip())
                db.session.add(person)
                db.session.flush()  # Get the ID
            
            # Associate person with video if not already associated
            if person not in video.people:
                video.people.append(person)
        
        # Update video tags
        tags = json.loads(video.tags) if video.tags else []
        if 'named' not in tags:
            tags.append('named')
        video.tags = json.dumps(tags)
        
        db.session.commit()
        
        # Add faces to face recognition database
        try:
            if video.thumbnail_path and os.path.exists(video.thumbnail_path):
                from face_recognition_module import get_face_db
                face_db = get_face_db()
                
                # Detect faces in the thumbnail
                detected_faces = face_db.detect_faces(video.thumbnail_path)
                
                if detected_faces:
                    # Process each face individually (same as recognition logic)
                    for i, face in enumerate(detected_faces):
                        if i < len(names) and names[i].strip():
                            person_name = names[i].strip()
                            person = Person.query.filter_by(name=person_name).first()
                            if person:
                                try:
                                    # Extract individual face region from thumbnail (same as recognition)
                                    from PIL import Image as PILImage
                                    
                                    # Load the thumbnail
                                    thumbnail_path = video.thumbnail_path
                                    if thumbnail_path.startswith('/static/'):
                                        thumbnail_path = thumbnail_path[1:]
                                    
                                    im = PILImage.open(thumbnail_path)
                                    
                                    # Add padding around face (same as recognition)
                                    padding_x = int(face['width'] * 0.5)
                                    padding_y = int(face['height'] * 0.5)
                                    
                                    # Calculate crop box with padding
                                    x1 = max(0, face['x'] - padding_x)
                                    y1 = max(0, face['y'] - padding_y)
                                    x2 = min(im.width, face['x'] + face['width'] + padding_x)
                                    y2 = min(im.height, face['y'] + face['height'] + padding_y)
                                    
                                    # Crop the individual face
                                    face_crop = im.crop((x1, y1, x2, y2))
                                    
                                    # Ensure minimum size for face recognition (112x112)
                                    min_size = 112
                                    if face_crop.width < min_size or face_crop.height < min_size:
                                        ratio = max(min_size / face_crop.width, min_size / face_crop.height)
                                        new_width = int(face_crop.width * ratio)
                                        new_height = int(face_crop.height * ratio)
                                        face_crop = face_crop.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
                                    
                                    # Save individual face as temporary file
                                    temp_crop_path = os.path.join(tempfile.gettempdir(), f'video_face_crop_{video.id}_{i}_{uuid4().hex}.jpg')
                                    face_crop.save(temp_crop_path)
                                    
                                    # Add THIS specific face to recognition database
                                    success = face_db.add_face_to_database(
                                        image_path=temp_crop_path,  # Use individual face crop
                                        person_id=person.id,
                                        person_name=person.name,
                                        is_confirmed=person.is_confirmed
                                    )
                                    
                                    # Clean up temp file
                                    try:
                                        os.remove(temp_crop_path)
                                    except Exception:
                                        pass
                                    
                                    if success:
                                        print(f"✓ Added video face {i+1} for {person_name} to recognition database")
                                    else:
                                        print(f"✗ Failed to add video face {i+1} for {person_name}")
                                        
                                except Exception as face_error:
                                    print(f"Error processing video face {i+1} for {person_name}: {face_error}")
                
                # Reload face database to make new faces immediately available
                reload_face_db()
        except Exception as e:
            print(f"Error adding video thumbnail faces to recognition database: {e}")
        
        return jsonify({'success': True, 'redirect_to': '/video-namer'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/video-namer/rotate', methods=['POST'])
def api_video_namer_rotate():
    """Rotate a video and regenerate thumbnail."""
    from models import Video, db
    import cv2
    import os
    from pathlib import Path
    
    data = request.get_json() or {}
    video_id = data.get('video_id')
    degrees = data.get('degrees', 90)
    
    if not video_id:
        return jsonify({'success': False, 'error': 'Missing video_id'}), 400
    
    try:
        video = Video.query.get(video_id)
        if not video:
            return jsonify({'success': False, 'error': 'Video not found'}), 404
        
        # Check if video file exists
        actual_video_path = video.file_path
        if actual_video_path.startswith('/static/'):
            actual_video_path = actual_video_path[1:]  # Remove leading slash
        
        if not os.path.exists(actual_video_path):
            return jsonify({'success': False, 'error': 'Video file not found'}), 404
        
        # Rotate the video
        cap = cv2.VideoCapture(actual_video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Could not open video file'}), 500
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output filename
        video_path = Path(actual_video_path)
        output_path = video_path.parent / f"{video_path.stem}_rotated{video_path.suffix}"
        
        # Setup video writer with browser-compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # avc1 codec for maximum browser compatibility
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (height, width) if degrees in [90, -90] else (width, height))
        
        # Process each frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Rotate frame
            if degrees == 90:
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif degrees == -90:
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif degrees == 180:
                rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
            else:
                rotated_frame = frame
            
            out.write(rotated_frame)
            frame_count += 1
            
            # Generate thumbnail from first frame
            if frame_count == 1:
                thumbnail_path = video_path.parent / f"{video_path.stem}_rotated_thumb.jpg"
                cv2.imwrite(str(thumbnail_path), rotated_frame)
        
        cap.release()
        out.release()
        
        # Update video record
        video.file_path = f"/static/videos/{output_path.name}"
        video.thumbnail_path = f"/static/videos/{thumbnail_path.name}"
        db.session.commit()
        
        return jsonify({
            'success': True,
            'thumbnail_path': f'/static/videos/{thumbnail_path.name}',
            'video_path': f"/static/videos/{output_path.name}"
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate_thumbnail/<int:video_id>', methods=['POST'])
def api_generate_thumbnail(video_id):
    """Generate a thumbnail for a video that doesn't have one."""
    try:
        video = Video.query.get(video_id)
        if not video:
            return jsonify({'success': False, 'error': 'Video not found'}), 404
        
        # Check if thumbnail already exists
        if video.thumbnail_path and os.path.exists(video.thumbnail_path.replace('/static/', 'static/')):
            return jsonify({'success': True, 'thumbnail_path': video.thumbnail_path})
        
        # Get video file path
        video_path = video.file_path.replace('/static/', 'static/')
        
        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': 'Video file not found'}), 404
        
        # Generate thumbnail using OpenCV
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Could not open video file'}), 500
        
        # Seek to 1 second
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'success': False, 'error': 'Could not extract frame from video'}), 500
        
        # Generate thumbnail path
        video_filename = Path(video_path).stem
        thumbnail_filename = f"{video_filename}_thumb.jpg"
        thumbnail_path = f"static/thumbnails/{thumbnail_filename}"
        
        # Ensure thumbnails directory exists
        os.makedirs("static/thumbnails", exist_ok=True)
        
        # Save thumbnail
        cv2.imwrite(thumbnail_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Update video record
        video.thumbnail_path = f"/static/thumbnails/{thumbnail_filename}"
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'thumbnail_path': f"/static/thumbnails/{thumbnail_filename}"
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/video-namer/delete_video', methods=['POST'])
def api_video_namer_delete_video():
    """Delete a video from the namer."""
    from models import Video, db
    import os, json
    
    data = request.get_json() or {}
    video_id = data.get('video_id')
    
    if not video_id:
        return jsonify({'success': False, 'error': 'Missing video_id'}), 400
    
    try:
        video = Video.query.get(video_id)
        if not video:
            return jsonify({'success': False, 'error': 'Video not found'}), 404
        
        # Delete the video file
        if video.file_path and os.path.exists(video.file_path.replace('/static/', 'static/')):
            try:
                os.remove(video.file_path.replace('/static/', 'static/'))
            except:
                pass  # File might not exist
        
        # Delete thumbnail if exists
        if video.thumbnail_path and os.path.exists(video.thumbnail_path.replace('/static/', 'static/')):
            try:
                os.remove(video.thumbnail_path.replace('/static/', 'static/'))
            except:
                pass  # File might not exist
        
        # Remove from database
        db.session.delete(video)
        db.session.commit()
        
        return jsonify({'success': True, 'redirect_to': '/video-namer'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

# ===== CHATBOT ROUTES =====

@app.route('/chat')
def chat_page():
    """Main chat page showing all chat sessions"""
    from models import ChatSession, Image
    sessions = ChatSession.query.order_by(ChatSession.updated_at.desc()).all()
    
    # Add image data to each session
    sessions_with_images = []
    for session in sessions:
        session_data = session.to_dict()
        if session.image_id:
            image = Image.query.get(session.image_id)
            if image:
                session_data['image'] = image.to_dict()
        sessions_with_images.append(session_data)
    
    return render_template('chat.html', sessions=sessions_with_images)

@app.route('/chat/<int:session_id>')
def chat_session(session_id):
    """Individual chat session page"""
    from models import ChatSession, ChatMessage, Image
    session = ChatSession.query.get_or_404(session_id)
    messages = session.messages.order_by(ChatMessage.created_at).all()
    
    # Get the associated image
    image = Image.query.get(session.image_id) if session.image_id else None
    
    return render_template('chat_session.html', 
                         session=session.to_dict(), 
                         messages=[m.to_dict() for m in messages],
                         image=image.to_dict() if image else None)

@app.route('/api/chat/create_session', methods=['POST'])
def create_chat_session():
    """Create a new chat session for an image"""
    from models import ChatSession, Image, db
    import os
    
    data = request.get_json()
    image_id = data.get('image_id')
    personality_description = data.get('personality_description', '').strip()
    
    if not image_id:
        return jsonify({'success': False, 'error': 'Image ID required'}), 400
    
    if not personality_description:
        return jsonify({'success': False, 'error': 'Personality description required'}), 400
    
    try:
        # Get the image
        image = Image.query.get_or_404(image_id)
        
        # Get image file path
        image_path = None
        if image.original_path and os.path.exists(image.original_path.replace('/static/', 'static/')):
            image_path = image.original_path.replace('/static/', 'static/')
        else:
            uploads_path = os.path.join('static', 'uploads', image.filename)
            if os.path.exists(uploads_path):
                image_path = uploads_path
        
        if not image_path:
            return jsonify({'success': False, 'error': 'Image file not found'}), 404
        
        print(f"[Chat] Creating session for image {image_id}, path: {image_path}")
        
        # Chat functionality disabled - Ollama not available
        return jsonify({'success': False, 'error': 'Chat functionality is disabled. Ollama chat is not available.'}), 503
        
        # Generate system prompt using Llama (disabled)
        system_prompt = f"Default system prompt for {personality_description}"
        
        # Create chat session
        session = ChatSession(
            image_id=image_id,
            personality_description=personality_description,
            system_prompt=system_prompt
        )
        db.session.add(session)
        db.session.commit()
        
        print(f"[Chat] Chat session {session.id} created successfully")
        
        # Add image URL to session data
        session_data = session.to_dict()
        session_data['image_url'] = image.to_dict()['image_url']
        
        return jsonify({
            'success': True, 
            'session': session_data,
            'message': 'Chat session created successfully!'
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"[Chat] Error creating session: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chat/send_message', methods=['POST'])
def send_chat_message():
    """Send a message in a chat session"""
    from models import ChatSession, ChatMessage, db
    
    data = request.get_json()
    session_id = data.get('session_id')
    message_content = data.get('message', '').strip()
    
    if not session_id:
        return jsonify({'success': False, 'error': 'Session ID required'}), 400
    
    if not message_content:
        return jsonify({'success': False, 'error': 'Message content required'}), 400
    
    try:
        # Get the session
        session = ChatSession.query.get_or_404(session_id)
        
        # Save user message
        user_message = ChatMessage(
            session_id=session_id,
            role='user',
            content=message_content
        )
        db.session.add(user_message)
        db.session.commit()
        
        # Get conversation history
        history = session.messages.order_by(ChatMessage.created_at).all()
        conversation_history = []
        for msg in history[:-1]:  # Exclude the message we just added
            conversation_history.append({
                'role': msg.role,
                'content': msg.content
            })
        
        # Chat functionality disabled - Ollama not available
        return jsonify({
            'success': False, 
            'error': 'Chat functionality is disabled. Ollama chat is not available.'
        }), 503
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/sessions')
def get_chat_sessions():
    """Get all chat sessions"""
    from models import ChatSession
    sessions = ChatSession.query.order_by(ChatSession.updated_at.desc()).all()
    return jsonify({'sessions': [s.to_dict() for s in sessions]})

@app.route('/api/chat/session/<int:session_id>')
def get_chat_session(session_id):
    """Get a specific chat session with messages"""
    from models import ChatSession, ChatMessage
    session = ChatSession.query.get_or_404(session_id)
    messages = session.messages.order_by(ChatMessage.created_at).all()
    return jsonify({
        'session': session.to_dict(),
        'messages': [m.to_dict() for m in messages]
    })

@app.route('/api/chat/delete_session/<int:session_id>', methods=['DELETE'])
def delete_chat_session(session_id):
    """Delete a chat session"""
    from models import ChatSession, db
    
    try:
        session = ChatSession.query.get_or_404(session_id)
        db.session.delete(session)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Session deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/chat/update_system_prompt/<int:session_id>', methods=['POST'])
def update_system_prompt(session_id):
    """Update the system prompt for a chat session"""
    from models import ChatSession, db
    
    data = request.get_json()
    system_prompt = data.get('system_prompt', '').strip()
    
    if not system_prompt:
        return jsonify({'success': False, 'error': 'System prompt cannot be empty'}), 400
    
    try:
        session = ChatSession.query.get_or_404(session_id)
        session.system_prompt = system_prompt
        session.updated_at = datetime.utcnow()
        db.session.commit()
        
        print(f"[Chat] Updated system prompt for session {session_id}")
        
        return jsonify({
            'success': True,
            'message': 'System prompt updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"[Chat] Error updating system prompt: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/chat/test_ollama')
def test_ollama():
    """Test Ollama connection and model availability (disabled)"""
    return jsonify({
        'success': False,
        'message': 'Chat functionality is disabled. Ollama chat is not available.',
        'error': 'Ollama chat has been removed from this application.'
    }), 503

# ===== VIDEO MANAGEMENT ROUTES =====

@app.route('/video')
def video_tab():
    """Video tab with grid display and upload functionality"""
    from models import Video, Person
    from sqlalchemy.orm import joinedload
    import json
    
    # Get pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)  # Limit to 50 videos per page
    per_page = min(per_page, 100)  # Cap at 100 videos per page
    
    # Get sort parameter from query string
    sort_option = request.args.get('sort', 'random')
    
    # Create base query with eager loading to prevent N+1 queries
    base_query = Video.query.options(joinedload(Video.people))
    
    # Handle random sorting differently - load all videos first
    if sort_option == 'random':
        # For random sorting, load ALL videos and shuffle them
        all_videos = base_query.all()
        import random
        
        # Use random seed if provided to ensure consistent randomization
        random_seed = request.args.get('random_seed')
        if random_seed:
            random.seed(int(random_seed))
        
        random.shuffle(all_videos)
        
        # Apply pagination manually to the shuffled list
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        videos = all_videos[start_idx:end_idx]
        
        # Create a mock pagination object
        total_videos = len(all_videos)
        total_pages = (total_videos + per_page - 1) // per_page
        
        videos_pagination = type('MockPagination', (), {
            'items': videos,
            'page': page,
            'pages': total_pages,
            'per_page': per_page,
            'total': total_videos,
            'has_next': page < total_pages,
            'has_prev': page > 1,
            'next_num': page + 1 if page < total_pages else None,
            'prev_num': page - 1 if page > 1 else None
        })()
    else:
        # Apply sorting for non-random options
        if sort_option == 'named-date':
            # Use a more efficient approach: load all videos and sort in Python
            # This avoids complex subqueries
            videos_query = base_query.order_by(Video.created_at.desc())
        elif sort_option == 'date-newest':
            videos_query = base_query.order_by(Video.created_at.desc())
        elif sort_option == 'date-oldest':
            videos_query = base_query.order_by(Video.created_at.asc())
        elif sort_option == 'filename-a-z':
            videos_query = base_query.order_by(Video.original_filename.asc())
        elif sort_option == 'filename-z-a':
            videos_query = base_query.order_by(Video.original_filename.desc())
        elif sort_option == 'size-largest':
            # Sort by file_size descending, with NULL values last
            videos_query = base_query.order_by(nullslast(Video.file_size.desc()), Video.id.desc())
        elif sort_option == 'size-smallest':
            # Sort by file_size ascending, with NULL values last
            videos_query = base_query.order_by(nullslast(Video.file_size.asc()), Video.id.asc())
        elif sort_option == 'duration-longest':
            # Sort by duration descending, with NULL values last
            # Use COALESCE to convert NULL to -1 so they sort last (negative values last in descending)
            videos_query = base_query.order_by(func.coalesce(Video.duration, -1).desc(), Video.id.desc())
        elif sort_option == 'duration-shortest':
            # Sort by duration ascending, with NULL values last
            # Use COALESCE to convert NULL to a very large number so they sort last in ascending
            videos_query = base_query.order_by(func.coalesce(Video.duration, 999999999).asc(), Video.id.asc())
        else:
            # Default to date-newest for better performance
            videos_query = base_query.order_by(Video.created_at.desc())
        
        # Apply pagination
        videos_pagination = videos_query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        videos = videos_pagination.items
        
        # For 'named-date' sorting, sort in Python after loading
        if sort_option == 'named-date':
            videos.sort(key=lambda v: (len(v.people) == 0, v.created_at), reverse=True)
    
    # Parse JSON tags efficiently - batch process
    for video in videos:
        if video.tags:
            try:
                video.parsed_tags = json.loads(video.tags)
            except (json.JSONDecodeError, TypeError):
                video.parsed_tags = []
        else:
            video.parsed_tags = []
    
    return render_template('video.html', 
                         videos=videos, 
                         current_sort=sort_option,
                         pagination=videos_pagination,
                         current_page=page)

@app.route('/api/videos/upload', methods=['POST'])
def upload_videos():
    """Upload multiple video files"""
    from models import Video, db
    import uuid
    import os
    
    if 'videos' not in request.files:
        return jsonify({'success': False, 'error': 'No videos provided'}), 400
    
    files = request.files.getlist('videos')
    if not files or all(file.filename == '' for file in files):
        return jsonify({'success': False, 'error': 'No files selected'}), 400
    
    uploaded_videos = []
    errors = []
    
    # Create videos directory if it doesn't exist
    videos_dir = os.path.join('static', 'videos')
    os.makedirs(videos_dir, exist_ok=True)
    
    for file in files:
        if file and file.filename:
            try:
                # Generate unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                file_path = os.path.join(videos_dir, unique_filename)
                
                # Save file
                file.save(file_path)
                
                # Get file size
                file_size = os.path.getsize(file_path)
                
                # Create video record
                video = Video(
                    filename=unique_filename,
                    original_filename=filename,
                    file_path=f'/static/videos/{unique_filename}',
                    file_size=file_size,
                    mime_type=file.content_type or 'video/mp4'
                )
                
                db.session.add(video)
                uploaded_videos.append(video)
                
            except Exception as e:
                errors.append(f"Error uploading {file.filename}: {str(e)}")
        else:
            errors.append(f"Invalid file: {file.filename if file else 'Unknown'}")
    
    try:
        db.session.commit()
        
        # Return uploaded videos data
        videos_data = [video.to_dict() for video in uploaded_videos]
        
        return jsonify({
            'success': True,
            'videos': videos_data,
            'uploaded_count': len(uploaded_videos),
            'errors': errors
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/videos')
def get_videos():
    """Get videos with pagination and filtering"""
    from models import Video, db
    from sqlalchemy.orm import joinedload
    
    # Get query parameters
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    per_page = min(per_page, 100)  # Cap at 100 videos per page
    sort_option = request.args.get('sort', 'random')
    
    # Create base query with eager loading
    base_query = Video.query.options(joinedload(Video.people))
    
    # Handle random sorting differently - load all videos first
    if sort_option == 'random':
        # For random sorting, load ALL videos and shuffle them
        all_videos = base_query.all()
        import random
        
        # Use random seed if provided to ensure consistent randomization
        random_seed = request.args.get('random_seed')
        if random_seed:
            random.seed(int(random_seed))
        
        random.shuffle(all_videos)
        
        # Apply pagination manually to the shuffled list
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        videos = all_videos[start_idx:end_idx]
        
        # Create a mock pagination object
        total_videos = len(all_videos)
        total_pages = (total_videos + per_page - 1) // per_page
        
        videos_pagination = type('MockPagination', (), {
            'items': videos,
            'page': page,
            'pages': total_pages,
            'per_page': per_page,
            'total': total_videos,
            'has_next': page < total_pages,
            'has_prev': page > 1,
            'next_num': page + 1 if page < total_pages else None,
            'prev_num': page - 1 if page > 1 else None
        })()
    else:
        # Apply sorting for non-random options
        if sort_option == 'date-newest':
            videos_query = base_query.order_by(Video.created_at.desc())
        elif sort_option == 'date-oldest':
            videos_query = base_query.order_by(Video.created_at.asc())
        elif sort_option == 'filename-a-z':
            videos_query = base_query.order_by(Video.original_filename.asc())
        elif sort_option == 'filename-z-a':
            videos_query = base_query.order_by(Video.original_filename.desc())
        elif sort_option == 'size-largest':
            # Sort by file_size descending, with NULL values last
            videos_query = base_query.order_by(nullslast(Video.file_size.desc()), Video.id.desc())
        elif sort_option == 'size-smallest':
            # Sort by file_size ascending, with NULL values last
            videos_query = base_query.order_by(nullslast(Video.file_size.asc()), Video.id.asc())
        elif sort_option == 'duration-longest':
            # Sort by duration descending, with NULL values last
            # Use COALESCE to convert NULL to -1 so they sort last (negative values last in descending)
            videos_query = base_query.order_by(func.coalesce(Video.duration, -1).desc(), Video.id.desc())
        elif sort_option == 'duration-shortest':
            # Sort by duration ascending, with NULL values last
            # Use COALESCE to convert NULL to a very large number so they sort last in ascending
            videos_query = base_query.order_by(func.coalesce(Video.duration, 999999999).asc(), Video.id.asc())
        else:
            videos_query = base_query.order_by(Video.created_at.desc())
        
        # Apply pagination
        videos_pagination = videos_query.paginate(
            page=page, per_page=per_page, error_out=False
        )
        videos = videos_pagination.items
    
    # Generate missing thumbnails for videos
    import cv2
    from pathlib import Path
    
    for video in videos:
        # Generate thumbnail if missing
        if not video.thumbnail_path or not os.path.exists(video.thumbnail_path.replace('/static/', 'static/')):
            try:
                video_path = video.file_path.replace('/static/', 'static/')
                
                if os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        # Seek to 1 second
                        cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret and frame is not None:
                            # Generate thumbnail path
                            video_filename = Path(video_path).stem
                            thumbnail_filename = f"{video_filename}_thumb.jpg"
                            thumbnail_dir = "static/thumbnails"
                            os.makedirs(thumbnail_dir, exist_ok=True)
                            thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
                            
                            # Save thumbnail
                            cv2.imwrite(thumbnail_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            
                            # Update video record in database
                            video.thumbnail_path = f"/static/thumbnails/{thumbnail_filename}"
                            db.session.commit()
                            print(f"Generated and saved thumbnail for video {video.id}: {video.thumbnail_path}")
            except Exception as e:
                print(f"Error generating thumbnail for video {video.id}: {e}")
                # Continue with other videos even if one fails
                continue
    
    return jsonify({
        'success': True,
        'videos': [video.to_dict() for video in videos],
        'pagination': {
            'page': videos_pagination.page,
            'pages': videos_pagination.pages,
            'per_page': videos_pagination.per_page,
            'total': videos_pagination.total,
            'has_next': videos_pagination.has_next,
            'has_prev': videos_pagination.has_prev,
            'next_num': videos_pagination.next_num,
            'prev_num': videos_pagination.prev_num
        }
    })

@app.route('/api/videos/<int:video_id>')
def get_video(video_id):
    """Get a specific video"""
    from models import Video
    video = Video.query.get_or_404(video_id)
    return jsonify({'success': True, 'video': video.to_dict()})

@app.route('/api/videos/<int:video_id>/delete', methods=['DELETE'])
def delete_video(video_id):
    """Delete a video"""
    from models import Video, db
    import os
    
    video = Video.query.get_or_404(video_id)
    
    try:
        # Delete file if it exists
        file_path = video.file_path.replace('/static/', 'static/')
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from database
        db.session.delete(video)
        db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/videos/<int:video_id>/toggle_favorite', methods=['POST'])
def toggle_video_favorite(video_id):
    """Toggle favorite status of a video"""
    from models import Video, db
    
    video = Video.query.get_or_404(video_id)
    video.is_favorite = not video.is_favorite
    
    try:
        db.session.commit()
        return jsonify({'success': True, 'is_favorite': video.is_favorite})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/person_videos/<int:person_id>')
def get_person_videos(person_id):
    """Get all videos for a specific person"""
    from models import Person, Video
    import json
    
    person = Person.query.get_or_404(person_id)
    
    # Get videos for this person, sorted by creation date
    videos = Video.query.filter(Video.people.contains(person)).order_by(Video.created_at.desc()).all()
    
    # Parse JSON tags for each video
    video_data = []
    for video in videos:
        video_dict = {
            'id': video.id,
            'file_path': video.file_path,
            'thumbnail_path': video.thumbnail_path,
            'original_filename': video.original_filename,
            'mime_type': video.mime_type,
            'created_at': video.created_at.isoformat() if video.created_at else None,
            'is_favorite': video.is_favorite
        }
        
        # Parse tags
        if video.tags:
            try:
                video_dict['parsed_tags'] = json.loads(video.tags)
            except (json.JSONDecodeError, TypeError):
                video_dict['parsed_tags'] = []
        else:
            video_dict['parsed_tags'] = []
        
        video_data.append(video_dict)
    
    return jsonify({
        'success': True,
        'videos': video_data,
        'person_name': person.name
    })

if __name__ == '__main__':
    import argparse
    import socket
    import atexit
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visage - Face Recognition Photo Management')
    parser.add_argument('--host', choices=['local', 'network'], default='local',
                       help='Choose host mode: local (127.0.0.1) or network (0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to run on (default: 5000)')
    parser.add_argument('--hostname', type=str, default='visage.local',
                       help='mDNS hostname (default: visage.local)')
    args = parser.parse_args()
    
    # Determine host based on choice
    mdns_service = None
    if args.host == 'local':
        host = '127.0.0.1'
        print("🌐 Running in LOCAL mode - accessible only on this computer")
        print(f"   URL: http://localhost:{args.port}")
    else:
        host = '0.0.0.0'
        print("🌐 Running in NETWORK mode - accessible on local network")
        
        # Get local IP address
        local_ip = None
        try:
            # Connect to a remote server to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"   Local URL: http://localhost:{args.port}")
            print(f"   Network URL: http://{local_ip}:{args.port}")
            print(f"   Other devices can access: http://{local_ip}:{args.port}")
        except Exception as e:
            print(f"   Local URL: http://localhost:{args.port}")
            print(f"   Network URL: http://0.0.0.0:{args.port}")
            print("   (Could not determine local IP address)")
        
        # Start mDNS/Bonjour service for .local hostname
        try:
            from zeroconf import ServiceInfo, Zeroconf
            
            if local_ip:
                service_name = f"_{args.hostname.split('.')[0]}._http._tcp.local."
                server_name = args.hostname
                
                info = ServiceInfo(
                    "_http._tcp.local.",
                    service_name,
                    addresses=[socket.inet_aton(local_ip)],
                    port=args.port,
                    server=server_name + ".",
                )
                
                zeroconf = Zeroconf()
                zeroconf.register_service(info)
                mdns_service = (zeroconf, info)
                
                print(f"   🏷️  mDNS/Bonjour hostname: http://{args.hostname}:{args.port}")
                print(f"   (Accessible from iPad/Apple devices at this address)")
            else:
                print("   ⚠️  Could not determine local IP - mDNS service not started")
        except ImportError:
            print("   ⚠️  zeroconf not installed - install with: pip install zeroconf")
            print("   (mDNS hostname will not be available)")
        except Exception as e:
            print(f"   ⚠️  Could not start mDNS service: {e}")
    
    # Cleanup function for mDNS
    def cleanup_mdns():
        if mdns_service:
            try:
                zeroconf, info = mdns_service
                zeroconf.unregister_service(info)
                zeroconf.close()
                print("\n✓ mDNS service stopped")
            except Exception:
                pass
    
    atexit.register(cleanup_mdns)
    
    print(f"   Port: {args.port}")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    with app.app_context():
        db.create_all()
        # Initialize face recognition model on startup
        # _ = get_face_db()
    
    try:
        app.run(debug=True, host=host, port=args.port)
    except KeyboardInterrupt:
        cleanup_mdns() 