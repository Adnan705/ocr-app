# 🚀 Deploy to Railway — Step-by-Step Guide

This guide gets your OCR web app live on Railway's free tier in under 10 minutes.

---

## Prerequisites

- A free [Railway account](https://railway.app) (sign up with GitHub)
- [Git](https://git-scm.com) installed locally
- [Railway CLI](https://docs.railway.app/develop/cli) (optional but faster)

---

## Option A — Deploy via GitHub (recommended)

### 1. Push your project to GitHub

```bash
cd ocr_deploy/          # your project folder

git init
git add .
git commit -m "Initial OCR app"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/ocr-app.git
git push -u origin main
```

### 2. Create a new Railway project

1. Go to [railway.app](https://railway.app) → **New Project**
2. Click **Deploy from GitHub repo**
3. Authorise Railway to access your GitHub
4. Select your `ocr-app` repository
5. Railway detects the `Dockerfile` automatically — click **Deploy**

### 3. Wait for the build

The first build takes **5–10 minutes** because it:
- Installs system packages (Tesseract, OpenCV)
- Installs Python dependencies
- Pre-downloads EasyOCR models (~100 MB)

Subsequent deploys are much faster due to Docker layer caching.

### 4. Get your live URL

Once deployed:
1. Click your service in the Railway dashboard
2. Go to **Settings → Domains**
3. Click **Generate Domain** → you'll get a URL like:
   `https://ocr-app-production.up.railway.app`

Your app is now live! 🎉

---

## Option B — Deploy via Railway CLI

```bash
# Install Railway CLI
npm install -g @railway/cli     # or: brew install railway

# Login
railway login

# From your project folder:
cd ocr_deploy/
railway init          # creates a new Railway project
railway up            # builds and deploys

# Get your URL
railway domain
```

---

## Environment Variables (optional)

Set these in Railway → your service → **Variables** tab:

| Variable       | Default | Purpose                            |
|----------------|---------|------------------------------------|
| `PORT`         | `8080`  | Port the app listens on (auto-set) |
| `FLASK_DEBUG`  | `0`     | Set to `1` for debug logs          |

Railway sets `PORT` automatically — you don't need to touch it.

---

## Free Tier Limits

Railway's free tier ("Hobby" plan) includes:
- **$5 / month** of free credits
- ~500 hours of runtime per month
- 1 GB RAM, shared CPU
- 1 GB disk

The OCR app uses ~300–400 MB RAM at idle, so you'll comfortably stay within limits for personal / demo use.

---

## Updating Your App

Any `git push` to `main` triggers an automatic re-deploy:

```bash
# Make your changes, then:
git add .
git commit -m "Update UI"
git push origin main
# Railway auto-deploys within ~2 minutes
```

---

## Troubleshooting

**Build fails with "no space left on device"**
> The EasyOCR model pre-download (~300 MB) can exceed Railway's build cache.
> In `Dockerfile`, comment out the EasyOCR pre-download line — models will
> download on first request instead (first OCR call will be slow).

**App crashes with "out of memory"**
> EasyOCR + PyTorch need ~400 MB RAM. Upgrade to Railway's Hobby plan
> ($5/month) which gives dedicated resources, or switch to Tesseract-only
> by setting `ocr_backend=tesseract` in requests.

**"Application failed to respond" on Railway**
> Check the deployment logs. Common causes:
> - Build didn't complete (wait for green checkmark)
> - Port mismatch (Railway injects `$PORT` — already handled in `app.py`)

**Tesseract not found**
> The Dockerfile installs `tesseract-ocr`. If you see errors, check the
> build logs for apt-get failures.

---

## File Structure After Deployment

```
ocr_deploy/
├── app.py              ← Flask server
├── preprocessing.py
├── detection.py
├── recognition.py
├── utils.py
├── static/
│   └── index.html      ← Web UI
├── requirements.txt
├── Dockerfile          ← Railway uses this
├── railway.json        ← Railway config
└── .gitignore
```
