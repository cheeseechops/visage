# Quick Start: Deploy Visage to Railway (5 minutes)

## 🚀 Fastest Way to Get 24/7 Access

### 1. Push to GitHub
```bash
git add .
git commit -m "Ready for Railway deployment"
git push
```

### 2. Deploy on Railway
1. Go to [railway.app](https://railway.app) → Sign up with GitHub
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your visage repository
4. Railway auto-detects and deploys!

### 3. Get Your URL
- Railway provides: `your-app-name.up.railway.app`
- Open this URL on your phone
- ✅ Done! Your app is now accessible 24/7

## 📱 Access from Phone

Just bookmark the Railway URL and open it anytime!

## ⚙️ Important Notes

- **First deploy takes 5-10 minutes** (installing dependencies)
- **CPU-only mode** (no GPU on Railway - face recognition will be slower)
- **Files reset on redeploy** unless you add a Volume (see full guide)

## 🔧 If Something Goes Wrong

1. Check Railway logs: Service → Logs tab
2. Common fix: Make sure `requirements-railway.txt` is being used
3. See `RAILWAY_DEPLOYMENT.md` for detailed troubleshooting

## 💰 Cost

- **Free tier**: $5 credit/month (enough for testing)
- **Hobby plan**: $5/month (unlimited, recommended for 24/7)

---

**That's it!** Your app will be live and accessible from anywhere. 🎉

