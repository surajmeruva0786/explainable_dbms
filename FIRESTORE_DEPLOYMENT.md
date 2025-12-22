# Deploying Firestore Security Rules

This guide explains how to deploy the Firestore security rules to fix the 403 Permission Denied error.

## Prerequisites

- Node.js and npm installed
- Access to the Firebase project (xaidbms)

## Option 1: Using Firebase CLI (Recommended)

### Step 1: Install Firebase CLI

```bash
npm install -g firebase-tools
```

### Step 2: Login to Firebase

```bash
firebase login
```

This will open a browser window for authentication.

### Step 3: Initialize Firebase in Your Project

Navigate to your project directory and run:

```bash
cd e:\github_projects\explainable_dbms
firebase init firestore
```

When prompted:
- Select your Firebase project: **xaidbms**
- Accept the default firestore.rules file location
- Accept the default firestore.indexes.json file location

### Step 4: Deploy the Rules

```bash
firebase deploy --only firestore:rules
```

You should see output like:
```
✔  Deploy complete!
```

## Option 2: Using Firebase Console (Manual)

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project: **xaidbms**
3. Click on **Firestore Database** in the left sidebar
4. Click on the **Rules** tab
5. Replace the existing rules with the contents of `firestore.rules`
6. Click **Publish**

## Verifying the Deployment

After deploying the rules:

1. Restart your application
2. Run an analysis
3. Check the terminal - you should no longer see the 403 error
4. Verify logs appear in Firebase Console under Firestore Database

## Security Note

⚠️ **Important**: The current rules allow public write access to logging collections. This is necessary for the application to function but could be abused. Consider implementing:

- Rate limiting
- Authentication
- IP whitelisting

For production deployments.

## Troubleshooting

### "Permission denied" when deploying

Make sure you're logged in with an account that has access to the Firebase project.

### Rules not taking effect

- Wait a few seconds after deployment
- Clear your browser cache
- Restart the application

### Firebase CLI not found

Make sure npm's global bin directory is in your PATH.
