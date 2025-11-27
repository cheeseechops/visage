# Initialize ONNX Runtime FIRST before ANYTHING else
# Add CUDA 12.2 DLL directory so ONNX Runtime can find the DLLs
# (This is done in app.py, but we ensure it here too as a safeguard)
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
    try:
        onnx_capi_dir = os.path.join(os.path.dirname(os.__file__), "..", "..", ".venv", "Lib", "site-packages", "onnxruntime", "capi")
        onnx_capi_dir = os.path.abspath(onnx_capi_dir)
        if os.path.exists(onnx_capi_dir) and os.path.exists(dll_path):
            target_dll = os.path.join(onnx_capi_dir, required_dll)
            if not os.path.exists(target_dll):
                shutil.copy2(dll_path, target_dll)
    except Exception:
        pass  # If copy fails, rely on PATH/add_dll_directory
    
    os.environ["CUDA_PATH"] = cuda_path

import onnxruntime as ort
ort.set_default_logger_severity(3)  # Suppress warnings

# Optimize ONNX Runtime for GPU performance
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())  # Use all CPU cores
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())  # Use all CPU cores for MKL
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())  # Use all CPU cores for OpenBLAS
os.environ['VECLIB_MAXIMUM_THREADS'] = str(os.cpu_count())  # Use all CPU cores for Accelerate
os.environ['NUMEXPR_NUM_THREADS'] = str(os.cpu_count())  # Use all CPU cores for NumExpr

# GPU optimization environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Non-blocking CUDA operations
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'  # RTX 4050 architecture

import os
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle
from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from models import db, Person, Image

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# Import performance logging
from utils.performance_logger import time_operation, time_function

class FaceRecognitionDB:
    def __init__(self, app: Flask, model_name: str = 'buffalo_l'):  # Changed from buffalo_m to buffalo_l for better accuracy
        """
        Initialize the face recognition database
        
        Args:
            app: Flask application instance
            model_name: InsightFace model name (buffalo_l, buffalo_m, buffalo_s)
        """
        self.app = app
        self.model_name = model_name
        self.app_insightface = None
        self.face_db_path = Path("face_database")
        self.face_db_path.mkdir(exist_ok=True)
        
        # Database files
        self.faces_file = self.face_db_path / "faces.pkl"
        self.metadata_file = self.face_db_path / "metadata.json"
        self.embeddings_file = self.face_db_path / "embeddings.npy"
        
        # Load or initialize face database
        self.faces_db = self._load_face_database()
        
    @time_function("face_db.load_database")
    def _load_face_database(self) -> Dict:
        """Load existing face database or create new one"""
        if self.faces_file.exists():
            try:
                with open(self.faces_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading face database: {e}")
        
        return {
            'people': {},  # person_id -> {name, embeddings, image_paths}
            'embeddings': np.array([]),
            'person_ids': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'model_name': self.model_name,
                'total_faces': 0,
                'total_people': 0
            }
        }
    
    @time_function("face_db.save_database")
    def _save_face_database(self):
        """Save face database to disk"""
        try:
            # Save faces dictionary
            with open(self.faces_file, 'wb') as f:
                pickle.dump(self.faces_db, f)
            
            # Save embeddings as numpy array
            if len(self.faces_db['embeddings']) > 0:
                np.save(self.embeddings_file, self.faces_db['embeddings'])
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.faces_db['metadata'], f, indent=2)
                
            print(f"Face database saved successfully")
        except Exception as e:
            print(f"Error saving face database: {e}")
    
    @time_function("face_db.initialize_insightface")
    def initialize_insightface(self):
        """Initialize InsightFace model"""
        try:
            # Check if ONNX Runtime GPU is available
            import onnxruntime as ort
            has_cuda_provider = 'CUDAExecutionProvider' in ort.get_available_providers()
            
            self.app_insightface = FaceAnalysis(name=self.model_name)
            
            # Try to use GPU if available, otherwise use CPU with optimizations
            import torch
            if torch.cuda.is_available() and has_cuda_provider:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"GPU available: {gpu_name}")
                print(f"ONNX Runtime CUDA provider available: {has_cuda_provider}")
                self.app_insightface.prepare(ctx_id=0, det_size=(640, 640))  # GPU
                print(f"InsightFace model '{self.model_name}' initialized on GPU successfully")
            else:
                if not torch.cuda.is_available():
                    print("GPU not available (PyTorch), using CPU with optimizations")
                elif not has_cuda_provider:
                    print("ONNX Runtime GPU provider not available (CPU-only ONNX Runtime installed), using CPU")
                    print("  Install onnxruntime-gpu for GPU acceleration: pip uninstall onnxruntime && pip install onnxruntime-gpu")
                # Use smaller detection size for faster CPU processing
                self.app_insightface.prepare(ctx_id=-1, det_size=(320, 320))  # CPU with smaller size
                print(f"InsightFace model '{self.model_name}' initialized on CPU successfully")
                
        except Exception as e:
            print(f"Error initializing InsightFace: {e}")
            # Fallback to CPU
            try:
                self.app_insightface.prepare(ctx_id=-1, det_size=(320, 320))
                print(f"InsightFace model '{self.model_name}' initialized on CPU (fallback)")
            except Exception as e2:
                print(f"Error in fallback initialization: {e2}")
                raise
    
    @time_function("face_db.extract_face_embedding")
    def extract_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Face embedding vector or None if no face detected
        """
        if self.app_insightface is None:
            self.initialize_insightface()
        
        try:
            # Read image
            with time_operation("face_db.read_image", image_path=image_path):
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Could not read image: {image_path}")
                    return None
            
            # Convert BGR to RGB
            with time_operation("face_db.convert_bgr_to_rgb"):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            with time_operation("face_db.detect_faces_insightface", image_path=image_path):
                faces = self.app_insightface.get(img_rgb)
            
            if len(faces) == 0:
                print(f"No faces detected in: {image_path}")
                return None
            
            if len(faces) > 1:
                print(f"Multiple faces detected in: {image_path}, using the largest face")
                # Use the face with largest bounding box
                largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
                return largest_face.embedding
            
            # Return embedding of the single face
            return faces[0].embedding
            
        except Exception as e:
            print(f"Error extracting face embedding from {image_path}: {e}")
            return None
    
    @time_function("face_db.get_single_person_images")
    def get_single_person_images(self) -> List[Dict]:
        """
        Get all images that have exactly one person assigned
        
        Returns:
            List of image dictionaries with person info
        """
        with self.app.app_context():
            with time_operation("face_db.query_all_images"):
                images = Image.query.all()
            
            single_person_images = []
            
            with time_operation("face_db.filter_single_person_images"):
                for img in images:
                    if len(img.people) == 1:
                        person = img.people[0]
                        single_person_images.append({
                            'image_id': img.id,
                            'person_id': person.id,
                            'person_name': person.name,
                            'image_path': img.cropped_path or f'static/uploads/{img.filename}',
                            'filename': img.filename,
                            'is_confirmed': person.is_confirmed
                        })
            
            return single_person_images
    
    def build_face_database(self, min_confidence: float = 0.5):
        """
        Build face recognition database from single-person images
        
        Args:
            min_confidence: Minimum confidence threshold for face detection
        """
        print("Building face recognition database...")
        
        # Get single-person images
        single_person_images = self.get_single_person_images()
        print(f"Found {len(single_person_images)} single-person images")
        
        # Initialize InsightFace if not already done
        if self.app_insightface is None:
            self.initialize_insightface()
        
        # Process each image
        processed_count = 0
        failed_count = 0
        
        for img_data in single_person_images:
            person_id = img_data['person_id']
            person_name = img_data['person_name']
            image_path = img_data['image_path']
            
            # Convert relative path to absolute
            if not os.path.isabs(image_path):
                image_path = os.path.join(os.getcwd(), image_path)
            
            # Check if image file exists
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                failed_count += 1
                continue
            
            # Extract face embedding
            embedding = self.extract_face_embedding(image_path)
            
            if embedding is not None:
                # Add to database
                if person_id not in self.faces_db['people']:
                    self.faces_db['people'][person_id] = {
                        'name': person_name,
                        'embeddings': [],
                        'image_paths': [],
                        'is_confirmed': img_data['is_confirmed']
                    }
                
                self.faces_db['people'][person_id]['embeddings'].append(embedding)
                self.faces_db['people'][person_id]['image_paths'].append(image_path)
                
                # Add to global embeddings array
                if len(self.faces_db['embeddings']) == 0:
                    self.faces_db['embeddings'] = embedding.reshape(1, -1)
                else:
                    self.faces_db['embeddings'] = np.vstack([self.faces_db['embeddings'], embedding])
                
                self.faces_db['person_ids'].append(person_id)
                processed_count += 1
                
                print(f"✓ Processed {person_name} - {img_data['filename']}")
            else:
                failed_count += 1
                print(f"✗ Failed to process {person_name} - {img_data['filename']}")
        
        # Update metadata
        self.faces_db['metadata'].update({
            'last_updated': datetime.now().isoformat(),
            'total_faces': len(self.faces_db['embeddings']),
            'total_people': len(self.faces_db['people']),
            'processed_images': processed_count,
            'failed_images': failed_count
        })
        
        # Save database
        self._save_face_database()
        
        print(f"\nFace database build completed!")
        print(f"✓ Processed: {processed_count} images")
        print(f"✗ Failed: {failed_count} images")
        print(f"📊 Total faces: {len(self.faces_db['embeddings'])}")
        print(f"👥 Total people: {len(self.faces_db['people'])}")
    
    @time_function("face_db.find_similar_faces")
    def find_similar_faces(self, image_path: str, threshold: float = 0.6, top_k: int = 5) -> List[Dict]:
        """
        Find similar faces in the database
        
        Args:
            image_path: Path to query image
            threshold: Similarity threshold
            top_k: Number of top matches to return
            
        Returns:
            List of similar faces with similarity scores
        """
        if self.app_insightface is None:
            self.initialize_insightface()
        
        # Extract embedding from query image
        query_embedding = self.extract_face_embedding(image_path)
        if query_embedding is None:
            return []
        
        # Calculate similarities
        with time_operation("face_db.calculate_similarities", 
                          embeddings_count=len(self.faces_db['embeddings'])):
            similarities = []
            for i, stored_embedding in enumerate(self.faces_db['embeddings']):
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                similarities.append((similarity, i))
        
        # Sort by similarity and filter by threshold
        with time_operation("face_db.sort_and_filter_matches", 
                          similarities_count=len(similarities), top_k=top_k):
            similarities.sort(reverse=True)
            matches = []
            
            for similarity, idx in similarities[:top_k]:
                if similarity >= threshold:
                    person_id = self.faces_db['person_ids'][idx]
                    person_data = self.faces_db['people'][person_id]
                    matches.append({
                        'person_id': person_id,
                        'person_name': person_data['name'],
                        'similarity': float(similarity),
                        'is_confirmed': person_data['is_confirmed']
                    })
        
        return matches
    
    def get_face_suggestions_from_path(self, image_path: str, threshold: float = 0.6, top_k: int = 5) -> List[Dict]:
        """
        Get face recognition suggestions for an image by file path
        
        Args:
            image_path: Path to the image file
            threshold: Similarity threshold
            top_k: Number of top matches to return
            
        Returns:
            List of suggestions with name and confidence
        """
        # Find similar faces
        similar_faces = self.find_similar_faces(image_path, threshold, top_k)
        
        # Convert to suggestions format
        suggestions = []
        for face in similar_faces:
            suggestions.append({
                'name': face['person_name'],
                'confidence': face['similarity'],
                'person_id': face['person_id'],
                'is_confirmed': face['is_confirmed']
            })
        
        return suggestions
    
    def add_face_to_database(self, image_path: str, person_id: int, person_name: str, is_confirmed: bool = True) -> bool:
        """
        Add a single face to the face recognition database
        
        Args:
            image_path: Path to the image file
            person_id: ID of the person
            person_name: Name of the person
            is_confirmed: Whether the person is confirmed
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            # Extract face embedding
            embedding = self.extract_face_embedding(image_path)
            if embedding is None:
                print(f"No face detected in: {image_path}")
                return False
            
            # Add to database
            if person_id not in self.faces_db['people']:
                self.faces_db['people'][person_id] = {
                    'name': person_name,
                    'embeddings': [],
                    'image_paths': [],
                    'is_confirmed': is_confirmed
                }
            
            self.faces_db['people'][person_id]['embeddings'].append(embedding)
            self.faces_db['people'][person_id]['image_paths'].append(image_path)
            
            # Add to global embeddings array
            if len(self.faces_db['embeddings']) == 0:
                self.faces_db['embeddings'] = embedding.reshape(1, -1)
            else:
                self.faces_db['embeddings'] = np.vstack([self.faces_db['embeddings'], embedding])
            
            self.faces_db['person_ids'].append(person_id)
            
            # Update metadata
            self.faces_db['metadata'].update({
                'last_updated': datetime.now().isoformat(),
                'total_faces': len(self.faces_db['embeddings']),
                'total_people': len(self.faces_db['people'])
            })
            
            # Save database
            self._save_face_database()
            
            print(f"✓ Added face for {person_name} to recognition database")
            return True
            
        except Exception as e:
            print(f"Error adding face to database: {e}")
            return False
    
    def reload_database(self):
        """
        Reload the face database from disk to get the latest changes
        """
        try:
            print("🔄 Reloading face recognition database...")
            self.faces_db = self._load_face_database()
            print(f"✓ Face database reloaded: {self.faces_db['metadata']['total_faces']} faces, {self.faces_db['metadata']['total_people']} people")
            return True
        except Exception as e:
            print(f"Error reloading face database: {e}")
            return False
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the face database"""
        stats = {
            'total_people': len(self.faces_db['people']),
            'total_faces': len(self.faces_db['embeddings']),
            'confirmed_people': sum(1 for p in self.faces_db['people'].values() if p['is_confirmed']),
            'unconfirmed_people': sum(1 for p in self.faces_db['people'].values() if not p['is_confirmed']),
            'people_with_multiple_faces': sum(1 for p in self.faces_db['people'].values() if len(p['embeddings']) > 1),
            'model_name': self.faces_db['metadata'].get('model_name', 'unknown'),
            'last_updated': self.faces_db['metadata'].get('last_updated', 'unknown')
        }
        
        # Add per-person statistics
        people_stats = []
        for person_id, person_data in self.faces_db['people'].items():
            people_stats.append({
                'person_id': person_id,
                'name': person_data['name'],
                'face_count': len(person_data['embeddings']),
                'is_confirmed': person_data['is_confirmed']
            })
        
        people_stats.sort(key=lambda x: x['face_count'], reverse=True)
        stats['top_people'] = people_stats[:10]
        
        return stats
    
    @time_function("face_db.detect_faces")
    def detect_faces(self, image_path: str) -> List[Dict]:
        """
        Detect faces in an image and return bounding box coordinates
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of face bounding boxes with coordinates
        """
        if self.app_insightface is None:
            self.initialize_insightface()
        
        try:
            # Read image
            with time_operation("face_db.read_image_for_detection", image_path=image_path):
                img = cv2.imread(image_path)
                if img is None:
                    return []
            
            # Convert BGR to RGB
            with time_operation("face_db.convert_bgr_to_rgb_for_detection"):
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            with time_operation("face_db.detect_faces_insightface", image_path=image_path):
                faces = self.app_insightface.get(img_rgb)
                
                if faces is None:
                    return []
            
            # Extract bounding boxes
            with time_operation("face_db.extract_bounding_boxes", faces_count=len(faces)):
                face_boxes = []
                for face in faces:
                    bbox = face.bbox.astype(int)
                    face_boxes.append({
                        'x': int(bbox[0]),
                        'y': int(bbox[1]),
                        'width': int(bbox[2] - bbox[0]),
                        'height': int(bbox[3] - bbox[1]),
                        'confidence': float(face.det_score)
                    })
            
            return face_boxes
            
        except Exception as e:
            print(f"Error detecting faces in {image_path}: {e}")
            return []

# Flask CLI commands for easy usage
def register_face_recognition_commands(app: Flask):
    """Register Flask CLI commands for face recognition operations"""
    
    @app.cli.command('build-face-db')
    def build_face_database():
        """Build face recognition database from single-person images"""
        with app.app_context():
            face_db = FaceRecognitionDB(app)
            face_db.build_face_database()
    
    @app.cli.command('face-db-stats')
    def face_database_stats():
        """Show face database statistics"""
        with app.app_context():
            face_db = FaceRecognitionDB(app)
            stats = face_db.get_database_stats()
            
            print("\n=== Face Recognition Database Statistics ===")
            print(f"Total People: {stats['total_people']}")
            print(f"Total Faces: {stats['total_faces']}")
            print(f"Confirmed People: {stats['confirmed_people']}")
            print(f"Unconfirmed People: {stats['unconfirmed_people']}")
            print(f"People with Multiple Faces: {stats['people_with_multiple_faces']}")
            print(f"Model: {stats['model_name']}")
            print(f"Last Updated: {stats['last_updated']}")
            
            print("\n=== Top 10 People by Face Count ===")
            for person in stats['top_people']:
                status = "✓" if person['is_confirmed'] else "?"
                print(f"{status} {person['name']}: {person['face_count']} faces") 

# --- Singleton accessor for FaceRecognitionDB ---
_face_db_instance = None

def get_face_db():
    global _face_db_instance
    from flask import current_app
    if _face_db_instance is None:
        _face_db_instance = FaceRecognitionDB(current_app)
    return _face_db_instance 