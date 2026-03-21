/**
 * /tracking-rtsp page only: POST URL, options fetch, WebRTC.
 * Kept separate from stream.js so this page still works if stream.js fails to load.
 */
function trackingRtspPage(config) {
  var RTSP_LOG = "[tracking-rtsp]";
  function logRtsp() {
    if (typeof console !== "undefined" && console.log) {
      var args = [RTSP_LOG].concat([].slice.call(arguments));
      console.log.apply(console, args);
    }
  }

  const status = document.getElementById("status");
  const video = document.getElementById("video");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const urlInput = document.getElementById("rtspUrlInput");
  const connectBtn = document.getElementById("rtspConnectBtn");
  const modelSelect = document.getElementById("modelSelect");
  const fpsInput = document.getElementById("fpsInput");
  const confInput = document.getElementById("confInput");
  const classesSelect = document.getElementById("classesSelect");

  logRtsp("init: trackingRtspPage() running");
  if (!status || !connectBtn || !startBtn) {
    logRtsp("FATAL: missing required DOM nodes (#status, #rtspConnectBtn, or #startBtn)");
    if (status) status.textContent = "Page error: missing controls (check console).";
    return;
  }

  const loadModelConfig = config.loadModel;
  const rtspOptionsUrl = config.rtspOptionsUrl;
  const postRtspUrl = config.postRtspUrl;
  const patchOptionsUrl = config.patchOptionsUrl || "/tracking/options";
  const wsPath = config.wsPath;

  let ws = null;
  let pc = null;
  let started = false;
  let sourceReady = false;

  var SESSION_STORAGE_KEY = "yolo_session_id";

  function storeSessionId(opts) {
    if (opts && opts.session_id) {
      try {
        sessionStorage.setItem(SESSION_STORAGE_KEY, opts.session_id);
      } catch (e) {}
    }
  }

  function wsUrl(path) {
    var p = path && path.charAt(0) === "/" ? path : "/" + (path || "");
    var base = (location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + p;
    var sid = null;
    try {
      sid = sessionStorage.getItem(SESSION_STORAGE_KEY);
    } catch (e) {}
    if (sid) {
      base += (base.indexOf("?") >= 0 ? "&" : "?") + "session_id=" + encodeURIComponent(sid);
    }
    return base;
  }

  function setStreamButtons() {
    startBtn.disabled = started || !sourceReady;
    stopBtn.disabled = !started;
  }

  function stopStream() {
    if (!started) return;
    started = false;
    setStreamButtons();
    status.textContent = "Stopped.";
    try {
      if (video.srcObject) {
        video.srcObject.getTracks().forEach(function (t) {
          t.stop();
        });
      }
    } catch (e) {}
    video.srcObject = null;
    try {
      if (pc) pc.close();
    } catch (e) {}
    try {
      if (ws) ws.close();
    } catch (e) {}
    pc = null;
    ws = null;
  }

  function startStream() {
    if (started || !sourceReady) return;
    started = true;
    setStreamButtons();
    status.textContent = "Connecting...";

    pc = new RTCPeerConnection();
    pc.ontrack = function (e) {
      status.textContent = "Starting video stream...";
      video.srcObject = new MediaStream([e.track]);
      function onFirstFrame() {
        status.textContent = "Stream connected";
        video.removeEventListener("loadeddata", onFirstFrame);
        video.removeEventListener("playing", onFirstFrame);
      }
      video.addEventListener("loadeddata", onFirstFrame);
      video.addEventListener("playing", onFirstFrame);
      video.play().catch(function () {});
    };
    pc.oniceconnectionstatechange = function () {
      if (pc.iceConnectionState === "failed" || pc.iceConnectionState === "closed") {
        status.textContent = "Connection " + pc.iceConnectionState;
      }
    };

    ws = new WebSocket(wsUrl(wsPath));
    ws.onopen = function () {
      status.textContent = "WebSocket connected, negotiating...";
      pc.addTransceiver("video", { direction: "recvonly" });
      pc.createOffer()
        .then(function (offer) {
          return pc.setLocalDescription(offer);
        })
        .then(function () {
          ws.send(JSON.stringify({ type: pc.localDescription.type, sdp: pc.localDescription.sdp }));
        })
        .catch(function (e) {
          status.textContent = "Error: " + e.message;
        });
    };

    ws.onmessage = function (ev) {
      var msg = JSON.parse(ev.data);
      if (msg.type === "answer" && msg.sdp) {
        pc.setRemoteDescription(new RTCSessionDescription(msg)).catch(function () {});
      } else if (msg.type === "ice" && msg.candidate) {
        try {
          pc.addIceCandidate(new RTCIceCandidate(msg.candidate));
        } catch (e) {}
      }
    };

    ws.onclose = function () {
      status.textContent = "WebSocket closed";
      if (started) stopStream();
    };
    ws.onerror = function () {
      status.textContent = "WebSocket error";
    };
  }

  startBtn.addEventListener("click", startStream);
  stopBtn.addEventListener("click", stopStream);
  window.addEventListener("beforeunload", stopStream);

  function buildTuningForPost() {
    var body = {};
    if (modelSelect && modelSelect.value !== "") body.model_key = modelSelect.value;
    if (fpsInput && fpsInput.value !== "") body.target_fps = parseInt(fpsInput.value, 10);
    if (confInput && confInput.value !== "") body.conf = parseFloat(confInput.value);
    if (classesSelect) {
      var selected = [].slice.call(classesSelect.selectedOptions).map(function (o) {
        return o.value;
      });
      body.classes = selected.length ? selected : null;
    }
    return body;
  }

  function patchTrackingOptions() {
    if (!sourceReady) return;
    var body = buildTuningForPost();
    fetch(patchOptionsUrl, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(body),
    })
      .then(function (res) {
        if (!res.ok) return res.text().then(function (t) {
          throw new Error(t || res.statusText);
        });
        return res.json();
      })
      .then(function (opts) {
        storeSessionId(opts);
      })
      .catch(function (e) {
        console.error("PATCH tracking options", e);
        if (status) status.textContent = "Failed to apply settings: " + (e.message || e);
      });
  }

  function warmupLoadModelBackground() {
    if (!loadModelConfig) return;
    logRtsp("warmup: GET", loadModelConfig.url);
    fetch(loadModelConfig.url, { credentials: "same-origin" })
      .then(function (res) {
        logRtsp("warmup: response", res.status, res.ok);
        if (!res.ok) throw new Error("Load model failed");
        status.textContent = loadModelConfig.readyText;
        setStreamButtons();
      })
      .catch(function (e) {
        logRtsp("warmup: error", e);
        status.textContent =
          "Stream ready. Model warmup failed (" +
          e.message +
          "); it will load when you press Start.";
        setStreamButtons();
      });
  }

  function connectRtsp() {
    var u = urlInput ? urlInput.value.trim() : "";
    if (!u) {
      status.textContent = "Enter a stream URL.";
      return;
    }
    logRtsp("connect: POST", postRtspUrl);
    status.textContent = "Connecting to stream...";
    connectBtn.disabled = true;
    var payload = Object.assign({ url: u }, buildTuningForPost());
    fetch(postRtspUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify(payload),
    })
      .then(function (res) {
        if (!res.ok) return res.text().then(function (t) {
          throw new Error(t || res.statusText);
        });
        return res.json();
      })
      .then(function (opts) {
        logRtsp("connect: POST ok, session_id=", opts && opts.session_id);
        storeSessionId(opts);
        sourceReady = true;
        connectBtn.disabled = false;
        status.textContent =
          "Stream URL saved. Warming up model in the background — you can press Start anytime.";
        setStreamButtons();
        warmupLoadModelBackground();
      })
      .catch(function (e) {
        logRtsp("connect: POST failed", e);
        connectBtn.disabled = false;
        status.textContent = "Failed: " + (e.message || e);
        console.error(RTSP_LOG, postRtspUrl, e);
      });
  }

  connectBtn.addEventListener("click", connectRtsp);

  var optionsFetchStart = typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
  var optionsController = typeof AbortController !== "undefined" ? new AbortController() : null;
  var optionsTimeoutMs = 120000;
  var optionsTimeoutId = null;
  if (optionsController) {
    optionsTimeoutId = setTimeout(function () {
      optionsController.abort();
      logRtsp("options fetch: TIMEOUT after " + optionsTimeoutMs + "ms — server may be stuck in yolo load_config or similar");
    }, optionsTimeoutMs);
  }

  status.textContent = "Requesting " + rtspOptionsUrl + " …";
  logRtsp("options: GET", rtspOptionsUrl, "started");

  fetch(rtspOptionsUrl, {
    credentials: "same-origin",
    signal: optionsController ? optionsController.signal : undefined,
  })
    .then(function (res) {
      var elapsed =
        (typeof performance !== "undefined" && performance.now ? performance.now() : Date.now()) - optionsFetchStart;
      logRtsp("options: response", res.status, res.ok, "elapsed_ms=", Math.round(elapsed));
      if (optionsTimeoutId) clearTimeout(optionsTimeoutId);
      if (!res.ok) throw new Error("GET options failed: HTTP " + res.status);
      return res.json();
    })
    .then(function (opts) {
      logRtsp("options: JSON parsed, session_id=", opts && opts.session_id, "models=", opts && opts.available_models && opts.available_models.length);
      storeSessionId(opts);
      if (modelSelect && opts.available_models) {
        modelSelect.innerHTML = "";
        opts.available_models.forEach(function (k) {
          var opt = document.createElement("option");
          opt.value = k;
          opt.textContent = k;
          if (opts.model_key === k) opt.selected = true;
          modelSelect.appendChild(opt);
        });
      }
      if (fpsInput && opts.target_fps != null) fpsInput.value = opts.target_fps;
      if (confInput && opts.conf != null) confInput.value = opts.conf;
      if (classesSelect && opts.available_classes) {
        classesSelect.innerHTML = "";
        opts.available_classes.forEach(function (name) {
          var opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          if (opts.classes && opts.classes.indexOf(name) !== -1) opt.selected = true;
          classesSelect.appendChild(opt);
        });
      }
      if (urlInput && opts.url) urlInput.value = opts.url;
      if (opts.url) {
        logRtsp("options: restored session has url, starting warmup");
        sourceReady = true;
        status.textContent = "Session restored. Warming up model in the background...";
        setStreamButtons();
        warmupLoadModelBackground();
        return;
      }
      status.textContent = "Enter a URL and click Connect, or change model/FPS first.";
      logRtsp("options: ready (no saved url)");
    })
    .catch(function (e) {
      if (optionsTimeoutId) clearTimeout(optionsTimeoutId);
      var name = e && e.name ? e.name : "";
      var msg = e && e.message ? e.message : String(e);
      logRtsp("options: FAILED", name, msg);
      console.error(RTSP_LOG, "GET", rtspOptionsUrl, e);
      if (name === "AbortError") {
        status.textContent =
          "Timed out waiting for " +
          rtspOptionsUrl +
          " (" +
          optionsTimeoutMs / 1000 +
          "s). See server terminal for logs after “GET /tracking/rtsp/options: start”.";
      } else {
        status.textContent = "Failed to load options: " + msg + " (see console).";
      }
    });

  if (modelSelect) modelSelect.addEventListener("change", patchTrackingOptions);
  if (fpsInput) fpsInput.addEventListener("change", patchTrackingOptions);
  if (confInput) confInput.addEventListener("change", patchTrackingOptions);
  if (classesSelect) classesSelect.addEventListener("change", patchTrackingOptions);

  setStreamButtons();
}

window.trackingRtspPage = trackingRtspPage;
