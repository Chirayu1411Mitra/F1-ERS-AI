# Deployment Guide: F1 ERS AI Strategy

This project is configured for **Hugging Face Spaces** using the Docker SDK. This provides a free, 24/7 hosting environment with sufficient resources (16GB RAM) for the AI models.

## Prerequisites

1.  A [Hugging Face account](https://huggingface.co/join).
2.  The files in this directory (prepared by Antigravity).

## Step 1: Create a Space

1.  Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2.  **Space Name**: Enter a name (e.g., `f1-ers-strategy`).
3.  **License**: Select `MIT` or `OpenRAIL`.
4.  **SDK**: Select **Docker**. (This is important! Do not select Streamlit or Gradio).
5.  **Space Hardware**: Select **Free** (CPU Basic · 2 vCPU · 16 GB).
6.  **Visibility**: Choose **Public** (required for free unauthenticated access) or Private.
7.  Click **Create Space**.

## Step 2: Upload Files

You can upload files directly via the browser or using Git.

### Option A: Browser Upload (Easiest)
1.  On your new Space page, click the **Files** tab.
2.  Click **Add file** -> **Upload files**.
3.  Drag and drop **ALL** files and folders from your local `d:\F1-ERS-AI` folder into the upload box.
    *   **Crucial**: Include `Dockerfile`, `requirements.txt`, `app.py`, `templates/`, `static/`, and the `models/` folder (especially `models/ers_policy.pkl`).
4.  In "Commit message", type "Initial deploy" and click **Commit changes**.

### Option B: Git (For advanced users)
1.  Clone the repository shown on your Space page.
    ```bash
    git clone https://huggingface.co/spaces/<your-username>/<space-name>
    ```
2.  Copy all your local project files into that cloned folder.
3.  Push:
    ```bash
    git add .
    git commit -m "Initial deploy"
    git push
    ```

## Step 3: Wait for Build

1.  Click the **App** tab on your Space.
2.  You will see a "Building" status. This takes 2-5 minutes as it installs dependencies and sets up the container.
3.  Once it says "Running", your app is live!

## Step 4: Connect Custom Domain

1.  Go to your Space's **Settings** tab.
2.  Scroll down to **Domain**.
3.  Enter your custom domain (e.g., `www.your-f1-site.com`).
4.  Copy the **Target** value provided (e.g., `spaces.huggingface.tech`).
5.  Log in to your Domain Registrar (GoDaddy, Namecheap, etc.).
6.  Add a **CNAME Record**:
    *   **Type**: `CNAME`
    *   **Name (Host)**: `www` (or `@` if supported/depending on your provider)
    *   **Value (Target)**: Paste the target from Hugging Face.
    *   **TTL**: Automatic or 3600.
7.  Wait for DNS propagation (usually 5-30 mins). Hugging Face will automatically provision a valid HTTPS certificate.

## Troubleshooting

*   **Runtime Error**: Check the **Logs** tab in your Space to see Python/Flask errors.
*   **"Space not found"**: Ensure your Space visibility is Public if accessing without being logged in.
