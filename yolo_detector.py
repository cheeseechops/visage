import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import tempfile
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential, ModuleList, ModuleDict
import torch.nn as nn
import requests

# Configure torch.load to allow ultralytics models (PyTorch 2.6+ compatibility)
# Add common PyTorch classes that ultralytics models might contain
safe_classes = [
    DetectionModel,
    Sequential,
    ModuleList,
    ModuleDict,
    nn.Module,
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.LeakyReLU,
    nn.SiLU,
    nn.Upsample,
    nn.MaxPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Dropout,
    nn.Linear,
    nn.Identity,
]
torch.serialization.add_safe_globals(safe_classes)

DEEPFASHION2_MODEL_URL = "https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt"
DEEPFASHION2_MODEL_PATH = "deepfashion2_yolov8s-seg.pt"

class YOLODetector:
    clothing_model_path = DEEPFASHION2_MODEL_PATH
    clothing_model_url = DEEPFASHION2_MODEL_URL

    @classmethod
    def set_clothing_model(cls, model_path, model_url=None):
        cls.clothing_model_path = model_path
        if model_url:
            cls.clothing_model_url = model_url
        # Remove the file if exists to force re-download
        if os.path.exists(model_path):
            os.remove(model_path)
        # Next instance will re-download and reload
        global _yolo_detector
        _yolo_detector = None

    @classmethod
    def force_reload_clothing_model(cls):
        # Remove the file to force re-download
        if os.path.exists(cls.clothing_model_path):
            os.remove(cls.clothing_model_path)
        global _yolo_detector
        _yolo_detector = None

    def __init__(self, model_path='yolov8n.pt', clothing_model_path=None):
        """Initialize YOLO detector with person and clothing detection models"""
        # Check for GPU availability with detailed diagnostics
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
            device = 0  # Use GPU (device index 0)
        else:
            print("GPU not available, using CPU")
            print(f"  PyTorch version: {torch.__version__}")
            print(f"  CUDA available: {torch.cuda.is_available()}")
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"  CUDA version in PyTorch: {torch.version.cuda}")
            device = 'cpu'
        
        # Temporarily monkey-patch torch.load to allow loading YOLO models (PyTorch 2.6+ compatibility)
        # This is safe since we're loading trusted ultralytics models
        original_load = torch.load
        def patched_load(*args, **kwargs):
            # Force weights_only=False for YOLO model loading
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        try:
            # Initialize YOLO model
            self.model = YOLO(model_path)
            # Explicitly move model to GPU/CPU device
            if device != 'cpu':
                # For GPU, ensure all model parameters are on the device
                self.model.to(device)
                print(f"YOLO person detection model loaded on GPU (device {device})")
            else:
                self.model.to(device)
                print("YOLO person detection model loaded on CPU")
        finally:
            torch.load = original_load
        
        # Use class variable for clothing model path
        if clothing_model_path is None:
            clothing_model_path = YOLODetector.clothing_model_path
        clothing_model_url = YOLODetector.clothing_model_url
        # Download clothing model if not present
        if not os.path.exists(clothing_model_path):
            print(f"Downloading clothing YOLOv8 model to {clothing_model_path}...")
            r = requests.get(clothing_model_url, stream=True)
            with open(clothing_model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        
        # Temporarily monkey-patch torch.load to allow loading YOLO models (PyTorch 2.6+ compatibility)
        original_load = torch.load
        def patched_load(*args, **kwargs):
            # Force weights_only=False for YOLO model loading
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        try:
            # Initialize YOLO clothing model
            self.clothing_model = YOLO(clothing_model_path)
            # Explicitly move model to GPU/CPU device
            if device != 'cpu':
                # For GPU, ensure all model parameters are on the device
                self.clothing_model.to(device)
                print(f"YOLO clothing detection model loaded on GPU (device {device})")
            else:
                self.clothing_model.to(device)
                print("YOLO clothing detection model loaded on CPU")
        finally:
            torch.load = original_load
        self.person_class_id = 0  # COCO dataset person class
        self.device = device  # Store device for use in detection methods
        
    def detect_persons(self, image_path):
        """
        Detect persons in an image and return bounding boxes
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries with person detection info:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'area': int
            }
        """
        try:
            # Run YOLO detection with device specification
            # Use the device from initialization for consistency
            results = self.model(image_path, verbose=False, device=self.device)
            
            persons = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Check if it's a person (class 0 in COCO)
                        if int(box.cls) == self.person_class_id:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            
                            # Calculate area
                            area = int((x2 - x1) * (y2 - y1))
                            
                            persons.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'area': area
                            })
            
            # Sort by confidence (highest first)
            persons.sort(key=lambda x: x['confidence'], reverse=True)
            
            return persons
            
        except Exception as e:
            print(f"Error detecting persons in {image_path}: {e}")
            return []
    
    def crop_person(self, image_path, bbox, output_path=None):
        """
        Crop a person from an image using bounding box
        
        Args:
            image_path: Path to the original image
            bbox: Bounding box [x1, y1, x2, y2]
            output_path: Path to save the cropped image (optional)
            
        Returns:
            PIL Image object of the cropped person
        """
        try:
            # Open image
            image = Image.open(image_path)
            
            # Crop the person
            x1, y1, x2, y2 = bbox
            cropped = image.crop((x1, y1, x2, y2))
            
            # Save if output path provided
            if output_path:
                cropped.save(output_path)
            
            return cropped
            
        except Exception as e:
            print(f"Error cropping person from {image_path}: {e}")
            return None
    
    def process_image(self, image_path, output_dir=None, confidence_threshold=0.5):
        """
        Process an image to detect and crop all persons
        
        Args:
            image_path: Path to the image file
            output_dir: Directory to save cropped images (optional)
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Dictionary with processing results:
            {
                'persons_found': int,
                'crops': List of crop info,
                'original_path': str,
                'cropped_paths': List of str
            }
        """
        # Detect persons
        persons = self.detect_persons(image_path)
        
        # Filter by confidence threshold
        persons = [p for p in persons if p['confidence'] >= confidence_threshold]
        
        crops = []
        cropped_paths = []
        
        if output_dir and persons:
            os.makedirs(output_dir, exist_ok=True)
            
            # Get base filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            for i, person in enumerate(persons):
                # Create output filename
                crop_filename = f"cropped_{base_name}_person_{i+1}.jpg"
                crop_path = os.path.join(output_dir, crop_filename)
                
                # Crop the person
                cropped_image = self.crop_person(image_path, person['bbox'], crop_path)
                
                if cropped_image:
                    crops.append({
                        'bbox': person['bbox'],
                        'confidence': person['confidence'],
                        'area': person['area'],
                        'crop_path': crop_path
                    })
                    cropped_paths.append(crop_path)
        
        return {
            'persons_found': len(persons),
            'crops': crops,
            'original_path': image_path,
            'cropped_paths': cropped_paths
        }

    def detect_clothing(self, image_path, confidence_threshold=0.3):
        """
        Detect clothing items in an image using DeepFashion2-trained YOLOv8 model.
        Returns a list of dicts: {class_name, bbox, confidence}
        """
        try:
            # Run YOLO detection with device specification
            # Use the device from initialization for consistency
            results = self.clothing_model(image_path, verbose=False, device=self.device)
            clothing_items = []
            for result in results:
                boxes = result.boxes
                names = result.names  # class id to name mapping
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls)
                        class_name = names[class_id] if names and class_id in names else str(class_id)
                        confidence = float(box.conf[0].cpu().numpy())
                        if confidence < confidence_threshold:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        clothing_items.append({
                            'class_name': class_name,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence
                        })
            return clothing_items
        except Exception as e:
            print(f"Error detecting clothing in {image_path}: {e}")
            return []

# Global instance for reuse
_yolo_detector = None

def get_yolo_detector(force_reload=False):
    """Get or create global YOLO detector instance
    
    Args:
        force_reload: If True, force recreation of the detector (useful after GPU setup changes)
    """
    global _yolo_detector
    if _yolo_detector is None or force_reload:
        if force_reload:
            print("Forcing YOLO detector reload...")
        _yolo_detector = YOLODetector()
    return _yolo_detector 