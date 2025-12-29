# 🚀 Simple Render Deployment Guide (No External Database)

## Overview

This guide shows you how to deploy your Explainable DBMS project on Render **without setting up an external database**. We'll use **SQLite** which is built-in and requires zero configuration.

> **Note**: SQLite data will be stored on Render's persistent disk, so it will survive redeployments.

---

## ⚡ Quick Deployment (5 Steps)

### Step 1: Update render.yaml

Your `render.yaml` needs to be updated to remove MySQL environment variables and use SQLite.

**Updated render.yaml:**

```yaml
services:
  # Backend API Service
  - type: web
    name: explainable-dbms-backend
    env: python
    region: oregon
    plan: free
    buildCommand: pip install -r requirements-backend.txt
    startCommand: python start.py
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: GEMINI_API_KEY
        sync: false
      - key: DB_TYPE
        value: sqlite
    disk:
      name: data
      mountPath: /opt/render/project/src
      sizeGB: 1

  # Frontend Static Site
  - type: web
    name: explainable-dbms-frontend
    env: static
    buildCommand: cd src/explainable_dbms/xai_dbms_frontend && npm install && npx vite build
    staticPublishPath: src/explainable_dbms/xai_dbms_frontend/dist
    routes:
      - type: rewrite
        source: /*
        destination: /index.html
```

**Key Changes:**
- ✅ Removed all `MYSQL_*` environment variables
- ✅ Added `DB_TYPE: sqlite`
- ✅ Persistent disk will store SQLite database file

---

### Step 2: Get Your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click **"Create API Key"**
3. Copy the API key (starts with `AIza...`)
4. Keep it handy for Step 4

---

### Step 3: Push to GitHub

Make sure all your code is committed and pushed:

```bash
# Check status
git status

# Add all changes
git add .

# Commit
git commit -m "Ready for Render deployment with SQLite"

# Push to GitHub
git push origin main
```

---

### Step 4: Deploy on Render

#### 4.1 Create Render Account

1. Go to [https://dashboard.render.com/register](https://dashboard.render.com/register)
2. Click **"Sign up with GitHub"** (recommended)
3. Authorize Render to access your repositories

#### 4.2 Create Blueprint

1. In Render Dashboard, click **"New +"** → **"Blueprint"**
2. Click **"Connect a repository"**
3. Find and select: `surajmeruva0786/explainable_dbms`
4. Click **"Connect"**

#### 4.3 Render Detects render.yaml

Render will automatically detect your `render.yaml` and show:

```
✅ explainable-dbms-backend (Web Service)
✅ explainable-dbms-frontend (Static Site)
```

#### 4.4 Set Environment Variables

**IMPORTANT**: Before clicking "Apply", you need to set your Gemini API key.

1. Click on **"explainable-dbms-backend"** in the blueprint preview
2. Scroll to **"Environment Variables"**
3. Find `GEMINI_API_KEY` (it will show as "sync: false")
4. Click **"Edit"** and paste your API key from Step 2
5. Click **"Save"**

**That's the only environment variable you need to set!**

#### 4.5 Apply Blueprint

1. Click **"Apply"** button at the bottom
2. Render will start creating your services

---

### Step 5: Monitor Deployment

#### Backend Deployment

1. Click on **"explainable-dbms-backend"** service
2. Watch the **"Logs"** tab
3. You should see:
   ```
   ==> Installing dependencies from requirements-backend.txt
   ==> Successfully installed all packages
   ==> 🚀 Starting FastAPI server on 0.0.0.0:10000
   ```
4. First deployment takes **5-10 minutes**

#### Frontend Deployment

1. Click on **"explainable-dbms-frontend"** service
2. Watch the **"Logs"** tab
3. You should see:
   ```
   ==> Running build command: cd src/explainable_dbms/xai_dbms_frontend && npm install && npx vite build
   ==> Build complete
   ==> Site is live
   ```
4. Takes **3-5 minutes**

---

### Step 6: Get Your URLs

After successful deployment, you'll have two URLs:

**Backend API:**
```
https://explainable-dbms-backend.onrender.com
```

**Frontend:**
```
https://explainable-dbms-frontend.onrender.com
```

**Test Backend:**
Visit: `https://explainable-dbms-backend.onrender.com/docs`

You should see the FastAPI interactive documentation.

---

### Step 7: Update Frontend to Connect to Backend

Your frontend needs to know where the backend is deployed.

#### Option A: Update Environment Variable in Render

1. Go to **"explainable-dbms-frontend"** service
2. Click **"Environment"** tab
3. Click **"Add Environment Variable"**
4. Add:
   - **Key**: `VITE_API_URL`
   - **Value**: `https://explainable-dbms-backend.onrender.com`
5. Click **"Save Changes"**
6. Render will automatically redeploy the frontend

#### Option B: Update Code and Push

1. Create/update `src/explainable_dbms/xai_dbms_frontend/.env`:
   ```env
   VITE_API_URL=https://explainable-dbms-backend.onrender.com
   ```

2. Commit and push:
   ```bash
   git add .
   git commit -m "Update frontend API URL for production"
   git push origin main
   ```

3. Render will automatically redeploy

---

## ✅ Verify Deployment

### Test Backend

1. **Visit API Docs:**
   ```
   https://explainable-dbms-backend.onrender.com/docs
   ```

2. **Test with curl:**
   ```bash
   curl https://explainable-dbms-backend.onrender.com/docs
   ```

### Test Frontend

1. **Visit Frontend:**
   ```
   https://explainable-dbms-frontend.onrender.com
   ```

2. **Test Full Workflow:**
   - Upload a CSV file
   - Click "Analyze"
   - View SHAP/LIME visualizations
   - Ask a question in the query interface

---

## 🐛 Troubleshooting

### Issue 1: Backend Build Fails

**Error**: `No module named 'fastapi'`

**Solution**: Ensure `requirements-backend.txt` includes:
```
fastapi
uvicorn[standard]
python-multipart
```

---

### Issue 2: Frontend Can't Connect to Backend

**Symptoms**: Upload works but analysis fails with network error

**Solution**:
1. Check browser console for CORS errors
2. Verify `VITE_API_URL` is set correctly
3. Ensure backend is running (check backend logs)

---

### Issue 3: Cold Starts (Free Tier)

**Problem**: First request after 15 minutes takes 30-60 seconds

**This is normal on Render's free tier!**

**Solutions**:
- Upgrade to paid plan ($7/month) for always-on service
- Use [UptimeRobot](https://uptimerobot.com/) to ping every 14 minutes
- Accept cold starts for demo/development

---

### Issue 4: Database File Not Found

**Error**: `sqlite3.OperationalError: unable to open database file`

**Solution**: Ensure persistent disk is mounted correctly:
- Mount path: `/opt/render/project/src`
- This is already configured in `render.yaml`

---

### Issue 5: Artifacts Not Persisting

**Problem**: Generated plots disappear after redeploy

**Solution**: Persistent disk should handle this. Verify:
1. Disk is attached to service
2. Mount path is `/opt/render/project/src`
3. Artifacts are saved to `artifacts/` directory

---

## 📊 What Gets Stored in SQLite

Your SQLite database (`explainable_dbms.db`) stores:

- ✅ **Predictions**: Model outputs with probabilities
- ✅ **SHAP Values**: Feature importance for each prediction
- ✅ **LIME Values**: Local explanations
- ✅ **Metadata**: Timestamps, model names, instance IDs

**Location on Render:**
```
/opt/render/project/src/explainable_dbms.db
```

This file persists across deployments thanks to the persistent disk.

---

## 💰 Cost Breakdown (Free Tier)

| Service | Cost | What You Get |
|---------|------|--------------|
| Backend Web Service | **$0** | 512 MB RAM, spins down after 15 min |
| Frontend Static Site | **$0** | 100 GB bandwidth/month |
| Persistent Disk (1 GB) | **$0** | Stores SQLite DB and artifacts |
| **Total** | **$0/month** | Perfect for demos! |

---

## 🔄 Continuous Deployment

Every time you push to GitHub, Render automatically redeploys:

```bash
# Make changes to your code
git add .
git commit -m "Add new feature"
git push origin main

# Render automatically:
# 1. Detects the push
# 2. Runs build command
# 3. Deploys new version
# 4. Zero-downtime deployment
```

**Disable Auto-Deploy:**
- Go to service settings
- Toggle **"Auto-Deploy"** to OFF
- Deploy manually via "Manual Deploy" button

---

## 🎯 Complete Deployment Checklist

- [ ] Updated `render.yaml` (removed MySQL variables)
- [ ] Obtained Gemini API key
- [ ] Committed and pushed all code to GitHub
- [ ] Created Render account
- [ ] Connected GitHub repository
- [ ] Created Blueprint from `render.yaml`
- [ ] Set `GEMINI_API_KEY` environment variable
- [ ] Applied Blueprint
- [ ] Backend deployed successfully
- [ ] Frontend deployed successfully
- [ ] Updated frontend `VITE_API_URL`
- [ ] Tested backend API docs
- [ ] Tested frontend upload and analysis
- [ ] Verified artifacts are generated

---

## 📝 Environment Variables Summary

You only need **ONE** environment variable:

| Variable | Value | Required |
|----------|-------|----------|
| `GEMINI_API_KEY` | Your API key from Google AI Studio | ✅ Yes |

**That's it!** No database credentials needed.

---

## 🚀 Quick Reference Commands

```bash
# 1. Prepare for deployment
git add .
git commit -m "Ready for Render deployment"
git push origin main

# 2. Go to Render
# https://dashboard.render.com

# 3. New + → Blueprint

# 4. Connect GitHub → Select repository

# 5. Set GEMINI_API_KEY

# 6. Click Apply

# 7. Wait 5-10 minutes

# 8. Visit your URLs!
```

---

## 🎉 You're Done!

Your Explainable DBMS is now live on Render with:

✅ **Backend API**: `https://explainable-dbms-backend.onrender.com`  
✅ **Frontend**: `https://explainable-dbms-frontend.onrender.com`  
✅ **SQLite Database**: Automatically managed  
✅ **Persistent Storage**: Artifacts saved across deployments  
✅ **Auto-Deploy**: Updates automatically on git push  

**Total Setup Time**: ~15 minutes  
**Total Cost**: $0/month  

---

## 📞 Need Help?

- **Render Docs**: [https://render.com/docs](https://render.com/docs)
- **Render Community**: [https://community.render.com](https://community.render.com)
- **GitHub Issues**: [Your Repository Issues](https://github.com/surajmeruva0786/explainable_dbms/issues)

---

**Happy Deploying! 🚀**
