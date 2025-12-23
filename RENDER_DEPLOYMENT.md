# Deploying Explainable DBMS Frontend to Render as Static Site

This guide walks you through deploying the React frontend of the Explainable DBMS project as a static site on Render.

## 📋 Prerequisites

- GitHub account with your repository pushed
- Render account (free tier available at [render.com](https://render.com))
- Your project's frontend built successfully locally

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

1. **Ensure your code is pushed to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Render deployment"
   git push origin main
   ```

2. **Verify build configuration** - Your `vite.config.ts` is already configured correctly with:
   - Build output directory: `build`
   - Target: `esnext`

### Step 2: Create a Render Static Site

1. **Log in to Render**:
   - Go to [https://dashboard.render.com](https://dashboard.render.com)
   - Sign in with your GitHub account

2. **Create New Static Site**:
   - Click **"New +"** button in the top right
   - Select **"Static Site"**

3. **Connect Your Repository**:
   - Select your GitHub repository: `surajmeruva0786/explainable_dbms`
   - Click **"Connect"**

### Step 3: Configure Build Settings

Configure the following settings in Render:

| Setting | Value |
|---------|-------|
| **Name** | `explainable-dbms-frontend` (or your preferred name) |
| **Branch** | `main` (or your default branch) |
| **Root Directory** | `src/explainable_dbms/xai_dbms_frontend` |
| **Build Command** | `npm install && npx vite build` |
| **Publish Directory** | `src/explainable_dbms/xai_dbms_frontend/build` |

> [!IMPORTANT]
> The **Root Directory** setting is crucial because your frontend is nested within the project structure.

### Step 4: Advanced Settings (Optional)

If you need to configure environment variables for your frontend:

1. Click **"Advanced"** to expand advanced settings
2. Add environment variables if needed (e.g., API endpoints):
   ```
   VITE_API_URL=https://your-backend-api.com
   ```

> [!NOTE]
> For Vite projects, environment variables must be prefixed with `VITE_` to be exposed to the client-side code.

### Step 5: Deploy

1. Click **"Create Static Site"**
2. Render will automatically:
   - Clone your repository
   - Navigate to the root directory
   - Run `npm install && npm run build`
   - Deploy the contents of the `build` folder

3. **Monitor the deployment**:
   - Watch the build logs in real-time
   - First deployment typically takes 2-5 minutes

### Step 6: Access Your Deployed Site

Once deployment completes:

1. Render will provide a URL like: `https://explainable-dbms-frontend.onrender.com`
2. Click the URL to view your live site
3. You can customize this URL in the site settings

## 🔧 Configuration Files

### Option A: Using `render.yaml` (Recommended)

Create a `render.yaml` file in your **project root** for infrastructure-as-code:

```yaml
services:
  - type: web
    name: explainable-dbms-frontend
    env: static
    buildCommand: cd src/explainable_dbms/xai_dbms_frontend && npm install && npm run build
    staticPublishPath: src/explainable_dbms/xai_dbms_frontend/build
    routes:
      - type: rewrite
        source: /*
        destination: /index.html
```

> [!TIP]
> The `routes` configuration ensures that client-side routing works correctly for single-page applications.

### Option B: Manual Configuration

If you prefer not to use `render.yaml`, configure everything through the Render dashboard as described in Step 3.

## 🌐 Custom Domain (Optional)

To use a custom domain:

1. Go to your static site settings in Render
2. Click **"Custom Domains"**
3. Add your domain (e.g., `app.yourdomain.com`)
4. Update your DNS records as instructed by Render
5. Render automatically provisions SSL certificates

## 🔄 Continuous Deployment

Render automatically redeploys your site when you push to your connected branch:

1. Make changes to your frontend code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update frontend"
   git push origin main
   ```
3. Render detects the push and automatically rebuilds/redeploys

## 🐛 Troubleshooting

### Build Fails with "Module not found"

**Solution**: Ensure all dependencies are in `package.json`:
```bash
cd src/explainable_dbms/xai_dbms_frontend
npm install
```

### 404 Errors on Page Refresh

**Solution**: Add the rewrite rule in `render.yaml` (see Configuration Files section) or configure redirects in Render dashboard:
- Go to **Redirects/Rewrites**
- Add: `/*` → `/index.html` (200 rewrite)

### Build Times Out

**Solution**: 
- Render free tier has a 15-minute build timeout
- Optimize your build by removing unused dependencies
- Consider upgrading to a paid plan for longer timeouts

### Environment Variables Not Working

**Solution**: 
- Ensure variables are prefixed with `VITE_`
- Rebuild the site after adding environment variables
- Check that you're accessing them with `import.meta.env.VITE_VARIABLE_NAME`

## 📊 Monitoring Your Deployment

Render provides:
- **Build logs**: View real-time build output
- **Deploy history**: Track all deployments
- **Analytics**: Monitor traffic and performance (paid plans)

## 💰 Pricing

- **Free Tier**: 
  - 100 GB bandwidth/month
  - Automatic SSL
  - Continuous deployment
  - Perfect for personal projects and demos

- **Paid Plans**: Start at $7/month for increased bandwidth and features

## 🔐 Security Best Practices

1. **Never commit sensitive data**: Use environment variables for API keys
2. **Enable HTTPS**: Render provides automatic SSL certificates
3. **Review build logs**: Ensure no secrets are exposed during build
4. **Use `.gitignore`**: Keep `.env` files out of version control

## 📝 Next Steps

After deploying your frontend:

1. **Deploy your backend**: Consider deploying the FastAPI backend separately as a web service
2. **Update API endpoints**: Configure your frontend to point to the deployed backend
3. **Test thoroughly**: Verify all features work in production
4. **Set up monitoring**: Use Render's monitoring or integrate third-party tools

## 🔗 Useful Links

- [Render Documentation](https://render.com/docs/static-sites)
- [Vite Deployment Guide](https://vitejs.dev/guide/static-deploy.html)
- [Render Community Forum](https://community.render.com/)

---

**Need help?** Check the [Render documentation](https://render.com/docs) or open an issue on the GitHub repository.
