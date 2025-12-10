# Render Deployment Guide (Free 24/7 + Custom Domain)

Follow these steps exactly to get your F1 ERS AI app online for free with your own domain and 24/7 uptime.

## 1. Push to GitHub
You need your code on GitHub for Render to access it.

1.  Create a new repository on GitHub (e.g., `f1-ers-ai`).
2.  Open your terminal in `d:\F1-ERS-AI` and run:
    ```powershell
    git init
    # If you have not configured git before:
    # git config --global user.email "you@example.com"
    # git config --global user.name "Your Name"
    git add .
    git commit -m "Initial Render deploy"
    git branch -M main
    git remote add origin https://github.com/<YOUR-USERNAME>/f1-ers-ai.git
    git push -u origin main
    ```

## 2. Deploy on Render
1.  Go to [dashboard.render.com](https://dashboard.render.com/) and log in (use GitHub).
2.  Click **New +** -> **Web Service**.
3.  Select "Build and deploy from a Git repository".
4.  Connect your GitHub account and select your `f1-ers-ai` repo.
5.  **Configuration**:
    *   **Name**: `f1-ers-app` (or whatever you like)
    *   **Region**: Closest to you (e.g., Singapore, Oregon).
    *   **Branch**: `main`
    *   **Runtime**: **Docker** (it should detect this automatically because of `Dockerfile`).
    *   **Instance Type**: **Free**.
6.  Click **Create Web Service**.
7.  Wait for it to build (might take 5-10 mins). Once it says "Live", your app is running!

## 3. Add Custom Domain (Free)
1.  In your Render Dashboard, go to your service.
2.  Click **Settings** -> **Custom Domains** -> **Add Custom Domain**.
3.  Enter your domain (e.g., `www.yourdomain.com`).
4.  Render will give you a `CNAME` record (e.g., `f1-ers-app.onrender.com`) and an `ANAME/ALIAS` record.
5.  Go to your Domain Registrar (GoDaddy, Namecheap, etc.) and add the CNAME record for `www`.
6.  Wait 15-30 mins for SSL to provision.

## 4. Make it 24/7 (Fix "Sleep")
Render's free tier sleeps after 15 mins of inactivity. We will use a free "pinger" to prevent this.

1.  Go to [UptimeRobot.com](https://uptimerobot.com/) and Sign Up (Free).
2.  Click **Add New Monitor**.
3.  **Monitor Type**: `HTTP(s)`.
4.  **Friendly Name**: `F1 App`.
5.  **URL (or IP)**: Paste your Render app URL (e.g., `https://f1-ers-app.onrender.com`).
6.  **Monitoring Interval**: **5 minutes**. (Important! Must be less than 15).
7.  Click **Create Monitor**.

**Done!** UseUptimeRobot will visit your site every 5 minutes, tricking Render into thinking it's always busy, so it never goes to sleep.
