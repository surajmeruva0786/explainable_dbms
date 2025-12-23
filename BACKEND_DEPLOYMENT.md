# Deploying Explainable DBMS Backend to Render

This guide walks you through deploying the FastAPI backend of the Explainable DBMS project as a web service on Render.

## 📋 Overview

The backend is a **FastAPI application** that provides:
- REST API endpoints for ML analysis
- LLM-powered code generation
- SHAP/LIME visualization generation
- Natural language query handling
- File upload and artifact serving

## 🗄️ Database Options

Your backend requires a MySQL database. You have **three options**:

### Option 1: Render PostgreSQL (Recommended for Render)

> [!TIP]
> Render's free PostgreSQL database integrates seamlessly and requires minimal configuration changes.

**Steps**:
1. Create a PostgreSQL database on Render (free tier available)
2. Update your SQLAlchemy code to use PostgreSQL instead of MySQL
3. Change `requirements-backend.txt`: replace `mysql-connector-python` with `psycopg2-binary`

### Option 2: External MySQL Database

Use a managed MySQL service like:
- **PlanetScale** (free tier available)
- **AWS RDS MySQL** (free tier for 12 months)
- **Railway** (MySQL support)
- **Aiven** (free tier available)

### Option 3: SQLite (Development/Demo Only)

> [!CAUTION]
> SQLite is NOT recommended for production as Render's filesystem is ephemeral. Data will be lost on redeploys.

For demo purposes, you can modify your code to use SQLite, but understand the limitations.

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

1. **Ensure all new files are committed**:
   ```bash
   git add requirements-backend.txt start.py render.yaml
   git commit -m "Add backend deployment configuration"
   git push origin main
   ```

2. **Verify your files**:
   - ✅ `requirements-backend.txt` - Backend dependencies
   - ✅ `start.py` - Production server script
   - ✅ `render.yaml` - Infrastructure configuration

### Step 2: Set Up Database (Choose One Option)

#### Option A: Create Render PostgreSQL Database

1. **In Render Dashboard**:
   - Click **"New +"** → **"PostgreSQL"**
   - Name: `explainable-dbms-db`
   - Select **Free** plan
   - Click **"Create Database"**

2. **Get connection details**:
   - Render provides: `Internal Database URL` and `External Database URL`
   - Copy the **Internal Database URL** (faster, free internal networking)

3. **Update your code** to use PostgreSQL:
   - Modify `src/explainable_dbms/config.py` or wherever you configure SQLAlchemy
   - Change from MySQL to PostgreSQL connection string

#### Option B: Use External MySQL

1. **Create MySQL database** on your chosen provider
2. **Get connection credentials**:
   - Host
   - Port (usually 3306)
   - Username
   - Password
   - Database name
3. **Keep these ready** for environment variable configuration

### Step 3: Create Backend Web Service

1. **Log in to Render**:
   - Go to [https://dashboard.render.com](https://dashboard.render.com)

2. **Create New Web Service**:
   - Click **"New +"** → **"Web Service"**
   - Connect your repository: `surajmeruva0786/explainable_dbms`

3. **Configure the service**:

| Setting | Value |
|---------|-------|
| **Name** | `explainable-dbms-backend` |
| **Region** | Oregon (US West) or closest to you |
| **Branch** | `main` |
| **Root Directory** | Leave empty (project root) |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements-backend.txt` |
| **Start Command** | `python start.py` |

### Step 4: Configure Environment Variables

Click **"Advanced"** and add these environment variables:

#### Required Variables:

| Key | Value | Notes |
|-----|-------|-------|
| `PYTHON_VERSION` | `3.11.0` | Specify Python version |
| `GEMINI_API_KEY` | `your_api_key_here` | Get from [Google AI Studio](https://makersuite.google.com/app/apikey) |

#### Database Variables (if using MySQL):

| Key | Value | Example |
|-----|-------|---------|
| `MYSQL_HOST` | Your database host | `mysql.example.com` |
| `MYSQL_PORT` | `3306` | Default MySQL port |
| `MYSQL_USER` | Your database username | `admin` |
| `MYSQL_PASSWORD` | Your database password | `secure_password` |
| `MYSQL_DATABASE` | Your database name | `explainable_dbms` |

#### Database Variables (if using Render PostgreSQL):

| Key | Value |
|-----|-------|
| `DATABASE_URL` | Copy from your Render PostgreSQL database (Internal URL) |

> [!IMPORTANT]
> Never commit API keys or passwords to your repository. Always use environment variables.

### Step 5: Configure Persistent Disk (Optional)

For storing generated artifacts persistently:

1. In **Advanced Settings**, scroll to **"Disks"**
2. Click **"Add Disk"**:
   - **Name**: `artifacts`
   - **Mount Path**: `/opt/render/project/src/artifacts`
   - **Size**: 1 GB (free tier)

> [!NOTE]
> Without a persistent disk, generated plots and analysis results will be lost on redeploys. The disk ensures artifacts persist across deployments.

### Step 6: Deploy

1. Click **"Create Web Service"**
2. Render will:
   - Clone your repository
   - Install dependencies from `requirements-backend.txt`
   - Run `python start.py`
   - Expose your API on a public URL

3. **Monitor deployment**:
   - Watch build logs in real-time
   - First deployment takes 5-10 minutes
   - Look for: `🚀 Starting FastAPI server on 0.0.0.0:10000`

### Step 7: Get Your API URL

Once deployed, Render provides a URL like:
```
https://explainable-dbms-backend.onrender.com
```

**Test your API**:
```bash
curl https://explainable-dbms-backend.onrender.com/docs
```

This should show the FastAPI interactive documentation.

## 🔗 Connect Frontend to Backend

Update your frontend to use the deployed backend API:

1. **In your frontend code**, update API base URL:
   ```javascript
   // In your frontend config or API client
   const API_BASE_URL = 'https://explainable-dbms-backend.onrender.com';
   ```

2. **Set as environment variable** (recommended):
   - In `src/explainable_dbms/xai_dbms_frontend/.env`:
     ```
     VITE_API_URL=https://explainable-dbms-backend.onrender.com
     ```
   - Use in code: `import.meta.env.VITE_API_URL`

3. **Redeploy frontend** to apply changes

## 📊 Using render.yaml (Infrastructure as Code)

The `render.yaml` file in your repository defines both services. Render can auto-detect this:

1. **Connect repository** to Render
2. Render detects `render.yaml`
3. Click **"Apply"** to create both services automatically

**Benefits**:
- Version-controlled infrastructure
- Consistent deployments
- Easy to replicate environments

## 🐛 Troubleshooting

### Build Fails: "No module named 'fastapi'"

**Solution**: Ensure `requirements-backend.txt` includes `fastapi` and `uvicorn[standard]`

### Database Connection Error

**Solution**: 
- Verify all database environment variables are set correctly
- Check database host is accessible from Render
- For Render PostgreSQL, use **Internal Database URL**
- Test connection string format

### "Address already in use" Error

**Solution**: This shouldn't happen on Render. If it does, check that `start.py` uses `os.getenv("PORT")` correctly.

### Artifacts Not Persisting

**Solution**: 
- Add a persistent disk as described in Step 5
- Ensure mount path matches where your app saves artifacts
- Check disk is attached in service settings

### CORS Errors from Frontend

**Solution**: Your `app.py` already has CORS middleware configured. If issues persist:
```python
# In app.py, update CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend.onrender.com",
        "http://localhost:3000"  # for local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Free Tier Limitations

**Render Free Tier**:
- ⚠️ **Spins down after 15 minutes of inactivity**
- First request after spin-down takes 30-60 seconds (cold start)
- 750 hours/month of runtime
- 512 MB RAM

**Solutions**:
- Upgrade to paid plan ($7/month) for always-on service
- Use a service like [UptimeRobot](https://uptimerobot.com/) to ping your API every 14 minutes
- Accept cold starts for demo/development

## 🔐 Security Best Practices

1. **Environment Variables**: Never hardcode secrets
2. **HTTPS**: Render provides automatic SSL certificates
3. **API Keys**: Rotate your Gemini API key regularly
4. **Database**: Use strong passwords and restrict access
5. **CORS**: Limit allowed origins in production

## 📈 Monitoring & Logs

**View Logs**:
- Go to your service in Render dashboard
- Click **"Logs"** tab
- Real-time streaming logs
- Filter by severity

**Monitor Performance**:
- **Metrics** tab shows CPU, memory usage
- Set up alerts for errors
- Track response times

## 💰 Cost Optimization

**Free Tier Strategy**:
- Use Render's free PostgreSQL (500 MB)
- Free web service (512 MB RAM)
- 1 GB free persistent disk
- Total: **$0/month**

**Paid Tier** ($7/month):
- Always-on service (no cold starts)
- 512 MB RAM (can upgrade)
- Better for production use

## 🔄 Continuous Deployment

Render automatically redeploys when you push to your branch:

1. Make changes to backend code
2. Commit and push:
   ```bash
   git add .
   git commit -m "Update backend logic"
   git push origin main
   ```
3. Render detects push and redeploys automatically

**Disable auto-deploy**: In service settings → **"Auto-Deploy"** → Off

## 🧪 Testing Your Deployed API

### Test File Upload:
```bash
curl -X POST https://your-backend.onrender.com/api/upload \
  -F "file=@dataset.csv"
```

### Test Analysis:
```bash
curl -X POST https://your-backend.onrender.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"filename": "dataset.csv", "target_column": "price"}'
```

### Test Query:
```bash
curl -X POST https://your-backend.onrender.com/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the model accuracy?", "analysis_id": "your-analysis-id"}'
```

### Interactive API Docs:
Visit: `https://your-backend.onrender.com/docs`

## 📝 Database Migration (MySQL to PostgreSQL)

If switching from MySQL to PostgreSQL:

1. **Update requirements**:
   ```txt
   # Replace mysql-connector-python with:
   psycopg2-binary
   ```

2. **Update connection string** in your code:
   ```python
   # MySQL format:
   # mysql+mysqlconnector://user:pass@host:3306/dbname
   
   # PostgreSQL format:
   # postgresql://user:pass@host:5432/dbname
   ```

3. **Update SQLAlchemy dialect**:
   - MySQL: `mysql+mysqlconnector://`
   - PostgreSQL: `postgresql://` or `postgresql+psycopg2://`

4. **Test locally** before deploying

## 🌐 Custom Domain (Optional)

1. Go to service settings → **"Custom Domains"**
2. Add your domain: `api.yourdomain.com`
3. Update DNS records as instructed
4. Render provisions SSL automatically

## 📚 Next Steps

After deploying your backend:

1. ✅ **Test all endpoints** using `/docs` interface
2. ✅ **Update frontend** to use new backend URL
3. ✅ **Set up monitoring** and alerts
4. ✅ **Configure database backups** (if using Render PostgreSQL)
5. ✅ **Document API** for team members
6. ✅ **Set up staging environment** for testing

## 🔗 Useful Resources

- [Render Python Docs](https://render.com/docs/deploy-fastapi)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Render Environment Variables](https://render.com/docs/environment-variables)
- [Render Persistent Disks](https://render.com/docs/disks)

---

**Need help?** Check the [Render Community Forum](https://community.render.com/) or open an issue on GitHub.

## 📋 Quick Reference

### Service Configuration Summary

```yaml
Type: Web Service
Runtime: Python 3.11
Build: pip install -r requirements-backend.txt
Start: python start.py
Port: Auto-assigned by Render (via $PORT)
Health Check: /docs endpoint
```

### Required Files Checklist

- ✅ `requirements-backend.txt` - Python dependencies
- ✅ `start.py` - Production server script  
- ✅ `render.yaml` - Infrastructure config (optional but recommended)
- ✅ `.env.example` - Template for environment variables
- ✅ `src/explainable_dbms/app.py` - FastAPI application

### Environment Variables Checklist

- ✅ `PYTHON_VERSION`
- ✅ `GEMINI_API_KEY`
- ✅ Database credentials (MySQL or PostgreSQL)
- ✅ `PORT` (auto-set by Render)

---

**You're all set!** 🎉 Your FastAPI backend should now be running on Render.
