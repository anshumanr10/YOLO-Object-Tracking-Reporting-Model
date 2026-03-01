/**
 * Shared WebRTC stream UI: start/stop, status, options.
 * config.wsPath: WebSocket path, e.g. "/ws" or "/ws-tracking"
 * config.loadModel (or config.warmup): { url, readyText } – run on load, then enable Start
 * config.optionsUrl: if set (e.g. "/tracking/options"), GET on load and PATCH on option change
 */
function streamPage(config) {
  const status = document.getElementById("status");
  const video = document.getElementById("video");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const cameraSelect = document.getElementById("cameraSelect");
  const modeSelect = document.getElementById("modeSelect");
  const modelSelect = document.getElementById("modelSelect");
  const fpsInput = document.getElementById("fpsInput");
  const confInput = document.getElementById("confInput");
  const classesSelect = document.getElementById("classesSelect");

  const loadModelConfig = config.loadModel || config.warmup;
  const optionsUrl = config.optionsUrl;
  const isRawStream = config.wsPath === "/ws";

  let ws = null;
  let pc = null;
  let started = false;

  function wsUrl(path) {
    var p = (path && path.charAt(0) === "/") ? path : "/" + (path || "");
    return (location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + p;
  }

  function setButtons() {
    startBtn.disabled = started;
    stopBtn.disabled = !started;
  }

  function stopStream() {
    if (!started) return;
    started = false;
    setButtons();
    status.textContent = "Stopped.";
    try {
      if (video.srcObject) {
        video.srcObject.getTracks().forEach(function(t) { t.stop(); });
      }
    } catch (e) {}
    video.srcObject = null;
    try { if (pc) pc.close(); } catch (e) {}
    try { if (ws) ws.close(); } catch (e) {}
    pc = null;
    ws = null;
  }

  function startStream() {
    if (started) return;
    started = true;
    setButtons();
    status.textContent = "Connecting...";

    pc = new RTCPeerConnection();
    pc.ontrack = function(e) {
      status.textContent = "Starting video stream...";
      video.srcObject = new MediaStream([e.track]);
      function onFirstFrame() {
        status.textContent = "Stream connected";
        video.removeEventListener("loadeddata", onFirstFrame);
        video.removeEventListener("playing", onFirstFrame);
      }
      video.addEventListener("loadeddata", onFirstFrame);
      video.addEventListener("playing", onFirstFrame);
      video.play().catch(function() {});
    };
    pc.oniceconnectionstatechange = function() {
      if (pc.iceConnectionState === "failed" || pc.iceConnectionState === "closed") {
        status.textContent = "Connection " + pc.iceConnectionState;
      }
    };

    ws = new WebSocket(wsUrl(config.wsPath));
    ws.onopen = function() {
      status.textContent = "WebSocket connected, negotiating...";
      pc.addTransceiver("video", { direction: "recvonly" });
      pc.createOffer().then(function(offer) {
        return pc.setLocalDescription(offer);
      }).then(function() {
        ws.send(JSON.stringify({ type: pc.localDescription.type, sdp: pc.localDescription.sdp }));
      }).catch(function(e) {
        status.textContent = "Error: " + e.message;
      });
    };

    ws.onmessage = function(ev) {
      var msg = JSON.parse(ev.data);
      if (msg.type === "answer" && msg.sdp) {
        pc.setRemoteDescription(new RTCSessionDescription(msg)).catch(function() {});
      } else if (msg.type === "ice" && msg.candidate) {
        try {
          pc.addIceCandidate(new RTCIceCandidate(msg.candidate));
        } catch (e) {}
      }
    };

    ws.onclose = function() {
      status.textContent = "WebSocket closed";
      if (started) stopStream();
    };
    ws.onerror = function() {
      status.textContent = "WebSocket error";
    };
  }

  startBtn.addEventListener("click", startStream);
  stopBtn.addEventListener("click", stopStream);
  window.addEventListener("beforeunload", stopStream);

  setButtons();

  function patchStreamOptions() {
    var body = {};
    if (cameraSelect && cameraSelect.value !== "") body.camera_index = parseInt(cameraSelect.value, 10);
    if (modeSelect && modeSelect.value !== "") body.sensor_mode_index = parseInt(modeSelect.value, 10);
    if (Object.keys(body).length === 0) return;
    fetch("/api/stream/options", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).catch(function(e) { console.error("PATCH stream options", e); });
  }

  function patchTrackingOptions() {
    if (!optionsUrl) return;
    var body = {};
    if (cameraSelect && cameraSelect.value !== "") body.camera_index = parseInt(cameraSelect.value, 10);
    if (modeSelect && modeSelect.value !== "") {
      var v = modeSelect.value;
      body.sensor_mode_index = v === "" ? null : parseInt(v, 10);
    }
    if (modelSelect && modelSelect.value !== "") body.model_key = modelSelect.value;
    if (fpsInput && fpsInput.value !== "") body.target_fps = parseInt(fpsInput.value, 10);
    if (confInput && confInput.value !== "") body.conf = parseFloat(confInput.value);
    if (classesSelect) {
      var selected = [].slice.call(classesSelect.selectedOptions).map(function(o) { return o.value; });
      body.classes = selected.length ? selected : null;
    }
    fetch(optionsUrl, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function(res) { return res.json(); }).catch(function(e) { console.error("PATCH tracking options", e); });
  }

  function runLoadModel() {
    if (!loadModelConfig) return Promise.resolve();
    status.textContent = "Loading model...";
    return fetch(loadModelConfig.url)
      .then(function(res) {
        if (!res.ok) throw new Error("Load model failed");
        status.textContent = loadModelConfig.readyText;
        startBtn.disabled = false;
      })
      .catch(function(e) {
        status.textContent = "Load model error: " + e.message;
      });
  }

  if (optionsUrl) {
    fetch(optionsUrl)
      .then(function(res) { return res.json(); })
      .then(function(opts) {
        if (modelSelect && opts.available_models) {
          modelSelect.innerHTML = "";
          opts.available_models.forEach(function(k) {
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
          opts.available_classes.forEach(function(name) {
            var opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            if (opts.classes && opts.classes.indexOf(name) !== -1) opt.selected = true;
            classesSelect.appendChild(opt);
          });
        }
        if (cameraSelect && opts.camera_index != null) {
          cameraSelect.value = String(opts.camera_index);
          if (modeSelect) {
            loadModesForCamera(opts.camera_index).then(function() {
              if (opts.sensor_mode_index != null && opts.sensor_mode_index !== "")
                modeSelect.value = String(opts.sensor_mode_index);
            });
          }
        }
      })
      .then(function() { return runLoadModel(); })
      .catch(function(e) { console.error("GET tracking options", e); });

    if (modelSelect) modelSelect.addEventListener("change", patchTrackingOptions);
    if (fpsInput) fpsInput.addEventListener("change", patchTrackingOptions);
    if (confInput) confInput.addEventListener("change", patchTrackingOptions);
    if (classesSelect) classesSelect.addEventListener("change", patchTrackingOptions);
  }

  if (cameraSelect && modeSelect) {
    var camerasPromise = fetch("/api/cameras")
      .then(function(res) { return res.json(); })
      .then(function(data) {
        var cams = data.cameras || [];
        cameraSelect.innerHTML = "";
        cams.forEach(function(cam) {
          var opt = document.createElement("option");
          opt.value = String(cam.index);
          opt.textContent = cam.index + ": " + cam.model + " (" + cam.location + ")";
          cameraSelect.appendChild(opt);
        });
        if (cams.length > 0 && !cameraSelect.value) {
          cameraSelect.value = String(cams[0].index);
          loadModesForCamera(cams[0].index);
        }
      })
      .catch(function(e) { console.error("Failed to load cameras", e); });

    if (isRawStream) {
      camerasPromise.then(function() {
        return fetch("/api/stream/options").then(function(res) { return res.json(); });
      }).then(function(opts) {
        if (!opts || opts.camera_index == null) return;
        if (cameraSelect.querySelector('option[value="' + opts.camera_index + '"]')) {
          cameraSelect.value = String(opts.camera_index);
          loadModesForCamera(opts.camera_index).then(function() {
            if (opts.sensor_mode_index != null && modeSelect.querySelector('option[value="' + opts.sensor_mode_index + '"]'))
              modeSelect.value = String(opts.sensor_mode_index);
          });
        }
      }).catch(function() {});
    }

    cameraSelect.addEventListener("change", function() {
      var idx = parseInt(cameraSelect.value, 10);
      if (!isNaN(idx)) loadModesForCamera(idx);
      if (isRawStream) patchStreamOptions();
      else if (optionsUrl) patchTrackingOptions();
    });
    modeSelect.addEventListener("change", function() {
      if (isRawStream) patchStreamOptions();
      else if (optionsUrl) patchTrackingOptions();
    });
  }

  function loadModesForCamera(cameraIndex) {
    if (!modeSelect) return Promise.resolve();
    modeSelect.innerHTML = "";
    return fetch("/api/cameras/" + encodeURIComponent(cameraIndex) + "/modes")
      .then(function(res) { return res.json(); })
      .then(function(data) {
        var modes = data.modes || [];
        modes.forEach(function(m) {
          var size = m.size || [0, 0];
          var fps = m.fps != null && typeof m.fps.toFixed === "function" ? m.fps.toFixed(2) : (m.fps || "0");
          var label = m.index + ": " + size[0] + "x" + size[1] + " @" + fps + " fps (" + (m.format || "") + ")";
          var opt = document.createElement("option");
          opt.value = String(m.index);
          opt.textContent = label;
          modeSelect.appendChild(opt);
        });
      })
      .catch(function(e) { console.error("Failed to load sensor modes", e); });
  }
}
