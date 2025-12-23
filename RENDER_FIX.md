# Render Deployment Fix - Vite Permission Error

## Problem
Build fails with error: `sh: 1: vite: Permission denied`

## Root Cause
When using `npm run build`, Render's build environment sometimes has issues executing binaries from `node_modules/.bin/`. This is especially common with Vite and other build tools in `devDependencies`.

## Solution
Use `npx vite build` instead of `npm run build` in the build command.

### What Changed

**Before:**
```yaml
buildCommand: cd src/explainable_dbms/xai_dbms_frontend && npm install && npm run build
```

**After:**
```yaml
buildCommand: cd src/explainable_dbms/xai_dbms_frontend && npm install && npx vite build
```

## Files Updated
- ✅ `render.yaml` - Updated build command
- ✅ `RENDER_DEPLOYMENT.md` - Updated documentation

## Next Steps

1. **Commit and push the fix:**
   ```bash
   git add render.yaml RENDER_DEPLOYMENT.md
   git commit -m "Fix: Use npx vite build to resolve permission error"
   git push origin main
   ```

2. **Trigger redeploy on Render:**
   - Go to your Render dashboard
   - Find the failed deployment
   - Click **"Manual Deploy"** → **"Clear build cache & deploy"**
   - Or wait for auto-deploy to trigger from your push

3. **Monitor the build:**
   - Watch the logs
   - Look for successful build completion
   - Verify the site deploys correctly

## Why This Works

- **`npx`** automatically finds and executes binaries from `node_modules/.bin/`
- It handles permissions correctly in CI/CD environments
- More reliable than relying on npm scripts in deployment contexts

## Alternative Solutions (if npx doesn't work)

### Option 1: Move vite to dependencies
```json
"dependencies": {
  "vite": "6.3.5"
}
```

### Option 2: Use explicit path
```bash
buildCommand: cd src/explainable_dbms/xai_dbms_frontend && npm install && ./node_modules/.bin/vite build
```

### Option 3: Use npm ci instead of npm install
```bash
buildCommand: cd src/explainable_dbms/xai_dbms_frontend && npm ci && npx vite build
```

## Verification

After successful deployment, verify:
- ✅ Build completes without errors
- ✅ Static files are generated in `build/` directory
- ✅ Site is accessible at your Render URL
- ✅ All routes work correctly (check the rewrite rules)

---

**Status**: ✅ Fixed - Ready to redeploy
