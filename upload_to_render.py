#!/usr/bin/env python3
"""
Script to upload database and images to Render deployment
Usage: python upload_to_render.py
"""

import requests
import os
from pathlib import Path

# Your Render app URL
RENDER_URL = "https://visage-z9vz.onrender.com"  # Update this to your URL

def upload_database():
    """Upload database file to Render"""
    print("Uploading database...")
    
    # Try instance folder first, then root
    db_paths = ["instance/visage.db", "visage.db"]
    db_path = None
    
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("❌ Database file not found. Tried:", db_paths)
        return False
    
    print(f"📁 Found database: {db_path}")
    
    try:
        with open(db_path, 'rb') as f:
            files = {'file': ('visage.db', f, 'application/x-sqlite3')}
            response = requests.post(f"{RENDER_URL}/api/admin/upload-db", files=files, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Database uploaded successfully!")
                print(f"   Found {result.get('tables', [])} tables")
                return True
            else:
                print(f"❌ Upload failed: {response.status_code}")
                print(f"   {response.text}")
                return False
    except Exception as e:
        print(f"❌ Error uploading database: {e}")
        return False

def upload_images_batch(upload_dir, max_files=50):
    """Upload images in batches"""
    print(f"\nUploading images from {upload_dir}...")
    
    if not os.path.exists(upload_dir):
        print(f"❌ Directory not found: {upload_dir}")
        return False
    
    # Get image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(upload_dir).glob(f"*{ext}"))
        image_files.extend(Path(upload_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in {upload_dir}")
        return False
    
    print(f"📸 Found {len(image_files)} images (uploading first {max_files})")
    
    uploaded = 0
    failed = 0
    
    for img_file in image_files[:max_files]:
        try:
            with open(img_file, 'rb') as f:
                files = {'images': (img_file.name, f, 'image/jpeg')}
                response = requests.post(f"{RENDER_URL}/import", files=files, timeout=60)
                
                if response.status_code in [200, 302]:
                    uploaded += 1
                    if uploaded % 10 == 0:
                        print(f"   ✅ Uploaded {uploaded} images...")
                else:
                    failed += 1
                    print(f"   ❌ Failed to upload {img_file.name}")
        except Exception as e:
            failed += 1
            print(f"   ❌ Error uploading {img_file.name}: {e}")
    
    print(f"\n✅ Upload complete: {uploaded} uploaded, {failed} failed")
    return uploaded > 0

def main():
    print("=" * 60)
    print("Upload Data to Render")
    print("=" * 60)
    
    # Upload database
    db_success = upload_database()
    
    # Upload images
    print("\n" + "=" * 60)
    print("Image Upload")
    print("=" * 60)
    print("Choose image directory to upload:")
    print("1. static/uploads")
    print("2. static/thumbnails")
    print("3. face_crops")
    print("4. Skip image upload")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        upload_images_batch("static/uploads")
    elif choice == "2":
        upload_images_batch("static/thumbnails")
    elif choice == "3":
        upload_images_batch("face_crops")
    else:
        print("Skipping image upload")
    
    print("\n" + "=" * 60)
    print("Upload Complete!")
    print("=" * 60)
    if db_success:
        print("✅ Database uploaded - your data should now be on Render")
    print(f"🌐 Visit your app: {RENDER_URL}")

if __name__ == "__main__":
    main()

