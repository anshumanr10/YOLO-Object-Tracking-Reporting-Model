# Exposing your local FastAPI backend for Lovable (or any remote frontend)

When the frontend is hosted on Lovable and the backend runs on your machine, the browser needs a **public URL** that reaches your local server. Use a tunnel.

## 1. Start the backend so it accepts connections

From the repo root with venv activated:

```bash
uvicorn fastapi_server.app.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` lets the tunnel (and the rest of the network) reach the app.

## 2. Expose it with a tunnel

### Option A: ngrok

1. Install: [ngrok.com/download](https://ngrok.com/download)
2. Run: `ngrok http 8000`
3. Copy the **HTTPS** URL (e.g. `https://abc123.ngrok-free.app`).

### Option B: Cloudflare Tunnel (no account for quick tunnels)

1. Install [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation)
2. Run: `cloudflared tunnel --url http://localhost:8000`
3. Copy the `*.trycloudflare.com` URL.

## 3. Point the frontend at that URL

- **Testbench:** Open the testbench, paste the tunnel URL into **API base**, click **Set**.
- **Stream page:** It uses the same API base (from localStorage). Set it on the testbench once, or it will use the current site origin (wrong when frontend is on Lovable).

So after deploying to Lovable, open the testbench once, set API base to your tunnel URL (e.g. `https://abc123.ngrok-free.app`), then both the testbench and stream page will call your local backend.

## 4. Saving ngrok bandwidth

- **Same network:** If the device opening the frontend is on the **same LAN** as the machine running the backend (e.g. same WiFi), set **API base** to the backend’s local IP (e.g. `http://192.168.1.100:8000`). The stream then goes **directly** to the browser and does **not** use the ngrok tunnel, so it doesn’t count against your ngrok bandwidth.
- **When using ngrok for the stream:** On the stream page, check **Low bandwidth** so the stream uses half resolution and lower JPEG quality (`?scale=0.5&jpeg_quality=50`). That reduces data through the tunnel.

## 5. MJPEG stream and ngrok (same-origin)

The stream request sends a custom header so ngrok skips its warning page. When the **frontend is on a different origin** (e.g. localhost or Lovable), the browser sends a CORS **preflight (OPTIONS)** first; that preflight does **not** include our header, so ngrok can return HTML and the stream fails.

**Fix:** Open the app **through the tunnel** so frontend and API are same-origin (no preflight):

- Open: `https://YOUR-NGROK-URL/testbench/stream.html` (or `/testbench/` for the testbench).
- Leave **API base** blank or set to the same ngrok URL so requests stay same-origin.
- Then Start stream; the GET will send the skip header and the stream will work.

If you see "Connecting…" then an error or no video, or "Got HTML instead of stream", you’re likely on a different origin — use the ngrok URL for the page as above.

## 6. CORS

The FastAPI app already uses `allow_origins=["*"]`, so a frontend on Lovable can call your backend as long as it’s reachable at the tunnel URL.

## 7. Security note

The tunnel makes your local backend reachable from the internet. Keep it for development/demos; use proper hosting and auth for production.
