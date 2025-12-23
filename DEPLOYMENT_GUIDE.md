# Quick Deployment Guide - Explainable DBMS

Complete guide for deploying both frontend and backend to Render.

## 📦 What You're Deploying

- **Frontend**: React + Vite static site
- **Backend**: FastAPI web service
- **Database**: MySQL or PostgreSQL (your choice)

## 🚀 Quick Start (5 Steps)

### 1. Push Your Code
```bash
git add .
git commit -m "Add deployment configuration"
git push origin main
```

### 2. Create Database (Choose One)

**Option A: Render PostgreSQL** (Recommended)
- Dashboard → New + → PostgreSQL
- Name: `explainable-dbms-db`
- Plan: Free
- Copy **Internal Database URL**

**Option B: External MySQL**
- Use PlanetScale, AWS RDS, or Railway
- Get connection credentials

### 3. Deploy Backend

**Via render.yaml** (Automatic):
1. Dashboard → New + → Web Service
2. Connect repository
3. Render detects `render.yaml`
4. Click "Apply"
5. Add environment variables (see below)

**Manual Configuration**:
- Name: `explainable-dbms-backend`
- Build: `pip install -r requirements-backend.txt`
- Start: `python start.py`
- Add environment variables

**Required Environment Variables**:
```
PYTHON_VERSION=3.11.0
GEMINI_API_KEY=your_gemini_api_key

# If using MySQL:
MYSQL_HOST=your_host
MYSQL_PORT=3306
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=explainable_dbms

# If using PostgreSQL:
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

### 4. Deploy Frontend

**Via render.yaml** (Automatic):
- Already configured in `render.yaml`
- Will deploy automatically with backend

**Manual Configuration**:
- Type: Static Site
- Root Directory: `src/explainable_dbms/xai_dbms_frontend`
- Build: `npm install && npm run build`
- Publish: `src/explainable_dbms/xai_dbms_frontend/build`

### 5. Connect Frontend to Backend

Update frontend API URL:
```javascript
// In frontend .env or config
VITE_API_URL=https://your-backend.onrender.com
```

Rebuild frontend to apply changes.

## 📋 Files Created

✅ `requirements-backend.txt` - Backend dependencies  
✅ `start.py` - Production server script  
✅ `render.yaml` - Infrastructure as code  
✅ `BACKEND_DEPLOYMENT.md` - Detailed backend guide  
✅ `RENDER_DEPLOYMENT.md` - Detailed frontend guide  

## 🔗 Your Deployed URLs

After deployment, you'll get:
- **Backend API**: `https://explainable-dbms-backend.onrender.com`
- **Frontend**: `https://explainable-dbms-frontend.onrender.com`
- **API Docs**: `https://explainable-dbms-backend.onrender.com/docs`

## ⚠️ Important Notes

### Free Tier Limitations
- Backend **spins down after 15 min** of inactivity
- First request after spin-down: 30-60 sec delay
- 750 hours/month runtime
- 512 MB RAM

### Database Choice
- **PostgreSQL**: Easier on Render, free tier available
- **MySQL**: Requires external provider (PlanetScale, etc.)
- **SQLite**: NOT recommended (ephemeral filesystem)

### Persistent Storage
- Add 1 GB disk for artifacts (free)
- Mount path: `/opt/render/project/src/artifacts`
- Prevents data loss on redeploys

## 🐛 Common Issues

**Build fails**: Check `requirements-backend.txt` has all dependencies  
**Database error**: Verify environment variables are set correctly  
**CORS error**: Already configured in `app.py`, check frontend URL  
**Cold starts**: Normal on free tier, upgrade to paid ($7/mo) for always-on  

## 📚 Detailed Guides

- **Backend**: See [BACKEND_DEPLOYMENT.md](./BACKEND_DEPLOYMENT.md)
- **Frontend**: See [RENDER_DEPLOYMENT.md](./RENDER_DEPLOYMENT.md)

## 🎯 Next Steps

1. ✅ Deploy backend with database
2. ✅ Deploy frontend
3. ✅ Test API endpoints at `/docs`
4. ✅ Update frontend to use backend URL
5. ✅ Test full application flow
6. ✅ Set up monitoring and alerts

---

**Need help?** Check the detailed guides or [Render Docs](https://render.com/docs)
