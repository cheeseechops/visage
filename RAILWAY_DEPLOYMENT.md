# Railway.app Deployment Guide for Visage

This guide will help you deploy your Visage Flask application to Railway.app for 24/7 access from your phone.

## Prerequisites

1. **GitHub Account** - Your code needs to be in a GitHub repository
2. **Railway Account** - Sign up at [railway.app](https://railway.app) (free tier available)

## Important Considerations

⚠️ **GPU/CUDA**: Your app uses GPU-accelerated ONNX Runtime. Railway doesn't provide GPU instances, so:
- The app will run slower (CPU-only mode)
- Face recognition will still work but be slower
- Consider this when processing large batches

⚠️ **Storage**: Railway's filesystem is ephemeral. Files uploaded will be lost on redeploy unless you:
- Use Railway's Volume plugin for persistent storage
- Or use external storage (S3, Cloudflare R2, etc.)

⚠️ **Database**: SQLite works but consider PostgreSQL for production (Railway offers managed Postgres)

## Step-by-Step Deployment

### Step 1: Prepare Your Code

1. **Push to GitHub** (if not already):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/visage.git
   git push -u origin main
   ```

2. **Verify files are in place**:
   - ✅ `Procfile` - Tells Railway how to run your app
   - ✅ `runtime.txt` - Specifies Python version
   - ✅ `requirements-railway.txt` - CPU-only dependencies
   - ✅ `railway.json` - Railway configuration

### Step 2: Deploy to Railway

1. **Sign up/Login**:
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub (recommended)

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your visage repository

3. **Configure the Service**:
   - Railway will auto-detect Python
   - Set the **Root Directory** to `/` (root of your repo)
   - Railway will use the `Procfile` automatically

4. **Set Environment Variables** (optional):
   - Go to your service → Variables tab
   - Add if needed:
     ```
     FLASK_ENV=production
     PORT=5000  # Railway sets this automatically
     ```

5. **Configure Build**:
   - Railway should auto-detect Python
   - If needed, set build command: `pip install -r requirements-railway.txt`

### Step 3: Add Persistent Storage (Recommended)

1. **Add Volume Plugin**:
   - In your Railway project, click "New" → "Volume"
   - Mount it to `/app/static` or `/app/instance`
   - This preserves uploaded files and database

2. **Update Paths** (if needed):
   - Your app uses relative paths which should work
   - Consider using environment variables for paths

### Step 4: Get Your Public URL

1. **Generate Domain**:
   - Go to your service → Settings → Generate Domain
   - Railway provides: `your-app-name.up.railway.app`
   - This is your public HTTPS URL!

2. **Custom Domain** (optional):
   - Add your own domain in Settings → Custom Domain

### Step 5: Access from Your Phone

1. **Open the Railway URL** on your phone's browser
2. **Bookmark it** for easy access
3. The app should work just like on your computer!

## Troubleshooting

### Build Fails

- **Check logs**: Railway dashboard → Deployments → View logs
- **Common issues**:
  - Missing dependencies → Check `requirements-railway.txt`
  - Python version mismatch → Check `runtime.txt`
  - Build timeout → Large dependencies may need optimization

### App Crashes

- **Check logs**: Service → Logs tab
- **Common issues**:
  - Port binding error → Railway sets `$PORT` automatically
  - Database errors → Ensure SQLite file is writable
  - Memory issues → Reduce workers in `Procfile`

### Slow Performance

- **Expected**: CPU-only mode is slower than GPU
- **Solutions**:
  - Reduce image processing batch sizes
  - Use smaller models if possible
  - Consider upgrading Railway plan for more resources

### Files Disappear After Redeploy

- **Solution**: Use Railway Volume plugin for persistent storage
- Mount volumes to:
  - `/app/static/uploads`
  - `/app/static/thumbnails`
  - `/app/instance` (for database)

## Alternative: Home PC with Cloudflare Tunnel

If you prefer to keep it on your PC (free, but requires PC to stay on):

1. **Install Cloudflare Tunnel**:
   - Download from: https://github.com/cloudflare/cloudflared/releases
   - Extract to a folder

2. **Run your app**:
   ```bash
   python app.py --host network --port 5000
   ```

3. **Run Cloudflare Tunnel**:
   ```bash
   cloudflared tunnel --url http://localhost:5000
   ```

4. **Access from phone**: Use the Cloudflare URL provided

## Cost Estimate

**Railway Free Tier**:
- $5/month credit
- 500 hours of usage
- Enough for 24/7 for ~20 days/month, or lighter usage all month

**Railway Hobby Plan** ($5/month):
- Unlimited usage
- Better for 24/7 access

## Next Steps

1. ✅ Deploy to Railway
2. ✅ Test from your phone
3. ✅ Set up persistent storage (Volume)
4. ✅ Consider PostgreSQL for production
5. ✅ Set up custom domain (optional)

## Support

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- Check Railway logs for errors

---

**Recommended**: Start with Railway's free tier to test, then upgrade if needed. The setup is straightforward and Railway handles HTTPS, scaling, and deployments automatically.

