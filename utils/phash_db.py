import os
import sqlite3
from PIL import Image
import imagehash
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

PHASH_DB_PATH = 'phash_cache.db'

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)

def ensure_db():
    conn = sqlite3.connect(PHASH_DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS phashes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE,
        phash TEXT
    )''')
    conn.commit()
    return conn

def compute_phash(image_path):
    try:
        if OPENCV_AVAILABLE:
            import cv2
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                return str(imagehash.phash(pil_img))
        # Fallback to PIL
        with Image.open(image_path) as img:
            return str(imagehash.phash(img))
    except Exception as e:
        print(f"Error computing pHash for {image_path}: {e}")
        return None

def build_phash_db(root_dirs, force_recompute=False, progress_callback=None, num_workers=8, batch_size=100):
    conn = ensure_db()
    c = conn.cursor()
    all_files = []
    for root_dir in root_dirs:
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if is_image_file(fname):
                    all_files.append(os.path.join(dirpath, fname))
    # Filter out already hashed if not force_recompute
    if not force_recompute:
        c.execute('SELECT path FROM phashes')
        existing = set(row[0] for row in c.fetchall())
        files_to_hash = [f for f in all_files if os.path.relpath(f) not in existing]
    else:
        files_to_hash = all_files
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_path = {executor.submit(compute_phash, f): f for f in files_to_hash}
        for idx, future in enumerate(tqdm(as_completed(future_to_path), total=len(future_to_path), desc='Hashing images (parallel)')):
            fpath = future_to_path[future]
            phash = future.result()
            if phash:
                results.append((os.path.relpath(fpath), phash))
            if progress_callback:
                progress_callback(idx + 1, len(files_to_hash))
            # Batch insert
            if len(results) >= batch_size:
                c.executemany('INSERT OR REPLACE INTO phashes (path, phash) VALUES (?, ?)', results)
                conn.commit()
                results = []
    # Final batch
    if results:
        c.executemany('INSERT OR REPLACE INTO phashes (path, phash) VALUES (?, ?)', results)
        conn.commit()
    conn.close()

def get_phash(path):
    conn = ensure_db()
    c = conn.cursor()
    c.execute('SELECT phash FROM phashes WHERE path=?', (os.path.relpath(path),))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def find_similar_phash(phash, max_distance=6):
    conn = ensure_db()
    c = conn.cursor()
    results = []
    for row in c.execute('SELECT path, phash FROM phashes'):
        other_path, other_phash = row
        if other_phash and phash:
            dist = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(other_phash)
            if dist <= max_distance:
                results.append((other_path, dist))
    conn.close()
    return results

def get_all_phashes():
    conn = ensure_db()
    c = conn.cursor()
    c.execute('SELECT path, phash FROM phashes')
    rows = c.fetchall()
    conn.close()
    return rows 