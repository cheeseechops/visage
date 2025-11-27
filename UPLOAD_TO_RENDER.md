# How to Upload Database and Images to Render

## Option 1: Render Disk (Recommended for Persistence)

### Step 1: Add a Disk in Render
1. Go to your Render dashboard
2. Click **"New +"** → **"Disk"**
3. Configure:
   - **Name**: `visage-data`
   - **Size**: Choose based on your data (start with 10GB, can increase later)
   - **Mount Path**: `/opt/render/project/src/data`
4. Click **"Create Disk"**

### Step 2: Update App to Use Disk
Update `app.py` to use the disk path:

```python
# In app.py, update database path
import os

# Use disk if available, otherwise use local
if os.path.exists('/opt/render/project/src/data'):
    DB_PATH = '/opt/render/project/src/data/visage.db'
    UPLOAD_FOLDER = '/opt/render/project/src/data/static/uploads'
    THUMBNAILS_FOLDER = '/opt/render/project/src/data/static/thumbnails'
else:
    DB_PATH = 'visage.db'
    UPLOAD_FOLDER = 'static/uploads'
    THUMBNAILS_FOLDER = 'static/thumbnails'

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
```

### Step 3: Upload Files via Render Shell
1. In Render dashboard → Your service → **"Shell"** tab
2. Create directories:
   ```bash
   mkdir -p /opt/render/project/src/data/static/uploads
   mkdir -p /opt/render/project/src/data/static/thumbnails
   mkdir -p /opt/render/project/src/data/static/videos
   ```
3. Use `scp` or `rsync` from your local machine to upload files

## Option 2: Upload via Web Interface (Easiest)

### For Database:
1. Go to your Render app URL
2. Use the import/upload feature in your app (if available)
3. Or create a temporary admin route to upload the database

### For Images:
1. Use your app's upload interface at `/import` or `/upload`
2. Upload images in batches

## Option 3: Use Render Shell + SCP (Manual)

### Step 1: Get Render Shell Access
1. In Render dashboard → Your service → **"Shell"** tab
2. This opens a terminal on your Render instance

### Step 2: Upload from Your PC
From your local machine, use `scp`:

```powershell
# Upload database
scp instance/visage.db render@your-service.onrender.com:/opt/render/project/src/visage.db

# Upload images (zip first for efficiency)
cd static
tar -czf ../uploads.tar.gz uploads/
scp ../uploads.tar.gz render@your-service.onrender.com:/opt/render/project/src/
```

**Note**: Render doesn't provide direct SSH/SCP access on free tier. You'll need to use Option 4.

## Option 4: Create Upload Script (Best for Large Files)

Create a Python script that uploads files via your app's API or a special admin route.

### Create `upload_to_render.py`:

```python
import requests
import os
import sqlite3
from pathlib import Path

RENDER_URL = "https://visage-z9vz.onrender.com"  # Your Render URL

def upload_database():
    """Upload database file"""
    db_path = "instance/visage.db"
    if not os.path.exists(db_path):
        db_path = "visage.db"
    
    if os.path.exists(db_path):
        # Read database
        with open(db_path, 'rb') as f:
            files = {'file': ('visage.db', f, 'application/x-sqlite3')}
            response = requests.post(f"{RENDER_URL}/api/admin/upload-db", files=files)
            print(f"Database upload: {response.status_code}")
    else:
        print("Database file not found")

def upload_images():
    """Upload images in batches"""
    upload_dir = Path("static/uploads")
    if not upload_dir.exists():
        print("Uploads directory not found")
        return
    
    image_files = list(upload_dir.glob("*.jpg")) + list(upload_dir.glob("*.png"))
    
    for img_file in image_files[:100]:  # Upload first 100 as test
        with open(img_file, 'rb') as f:
            files = {'file': (img_file.name, f, 'image/jpeg')}
            response = requests.post(f"{RENDER_URL}/import", files=files)
            print(f"Uploaded {img_file.name}: {response.status_code}")

if __name__ == "__main__":
    upload_database()
    upload_images()
```

## Option 5: Use Render's Environment Variables + External Storage

Since Render's filesystem is ephemeral, consider:
- **Cloudflare R2** (free tier: 10GB)
- **AWS S3** (pay-as-you-go)
- **OneDrive API** (as we discussed earlier)

Then update your app to read from external storage instead of local files.

## Quick Start: Manual Upload via Web

**Easiest method for now:**

1. **Database**: 
   - Export your database to SQL or CSV
   - Create an import route in your app
   - Import via web interface

2. **Images**:
   - Go to your Render app: `https://visage-z9vz.onrender.com/import`
   - Upload images using the web interface
   - Or zip images and upload via the import workflow

## Recommended Approach

For **immediate use**: Use your app's web interface to upload images.

For **persistent storage**: Add a Render Disk ($0.25/GB/month) and update paths to use it.

For **long-term**: Migrate to external storage (Cloudflare R2 or OneDrive API).

---

**Which method would you like to use?** I can help set up any of these options.

