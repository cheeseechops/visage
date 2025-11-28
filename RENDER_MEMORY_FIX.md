# Fix Memory Issues on Render Free Tier

## The Problem
YOLO (Ultralytics) loads PyTorch with CUDA support, which includes huge CUDA libraries (~2GB) even when using CPU. This causes out-of-memory errors on Render's 512MB free tier.

## Solution: Disable YOLO on Render

### Option 1: Environment Variable (Recommended)

In Render dashboard → Your service → Environment Variables:
- Add: `DISABLE_YOLO=true`

This will:
- ✅ Prevent YOLO from loading
- ✅ Save ~200-300MB of memory
- ✅ App will still work (YOLO features will return 503 errors)
- ✅ Face recognition still works (lighter, uses ONNX)

### Option 2: Comment Out ultralytics

In `requirements-railway.txt`, comment out:
```txt
# ultralytics==8.0.196  # Disabled - too heavy for free tier
```

Then YOLO won't be installed at all.

## What Still Works

- ✅ Face recognition (InsightFace) - lighter, uses ONNX
- ✅ Image browsing and management
- ✅ People management
- ✅ All core features

## What Won't Work

- ❌ YOLO person detection (auto-cropper)
- ❌ YOLO clothing detection
- ❌ Automatic person detection features

## Alternative: Upgrade Render Plan

If you need YOLO:
- **Starter Plan** ($7/month): 512MB → 2GB RAM
- Enough for YOLO + face recognition

---

**Quick Fix**: Add `DISABLE_YOLO=true` in Render environment variables.

