(() => {
  const $ = (selector) => document.querySelector(selector);
  const fileInput = $('#fileInput'), dropzone = $('#dropzone'), colorizeButton = $('#colorizeButton');
  const errorBox = $('#errorBox'), results = $('#results'), progress = $('#progressPanel');
  let selectedFile = null, previewUrl = null, resultUrl = null, busy = false;
  let maxUploadBytes = 12 * 1024 * 1024, maxImagePixels = 20_000_000, previewAvailable = false;

  function clearResult() {
    results.hidden = true;
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    resultUrl = null;
    $('#resultImage').removeAttribute('src');
  }
  function setError(message) { errorBox.textContent = message; errorBox.hidden = !message; }
  function chooseFile() { fileInput.click(); }
  function selectFile(file, localUrl, dimensions = null) {
    previewAvailable = Boolean(dimensions);
    if (previewAvailable) {
      $('#originalPreview').src = localUrl;
      $('#originalPreview').hidden = false;
      $('#previewFallback').hidden = true;
    } else {
      $('#originalPreview').removeAttribute('src');
      $('#originalPreview').hidden = true;
      $('#previewFallback').hidden = false;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = localUrl; selectedFile = file;
    $('#fileName').textContent = file.name;
    $('#fileDimensions').textContent = dimensions
      ? `${dimensions.width} × ${dimensions.height} px · ${(file.size / 1048576).toFixed(2)} MB`
      : `Preview unavailable · ${(file.size / 1048576).toFixed(2)} MB`;
    dropzone.hidden = true; $('#selectedPanel').hidden = false; colorizeButton.disabled = false;
  }
  function loadFile(file) {
    setError(''); clearResult(); $('#modelNotice').hidden = true;
    if (!file) return;
    if (file.size > maxUploadBytes) return setError(`This file is larger than the ${Math.floor(maxUploadBytes / 1048576)} MB limit. Choose a smaller image.`);
    const localUrl = URL.createObjectURL(file), image = new Image();
    image.onload = () => {
      if (image.naturalWidth * image.naturalHeight > maxImagePixels) { URL.revokeObjectURL(localUrl); return setError(`This image exceeds the ${maxImagePixels.toLocaleString()}-pixel processing limit.`); }
      selectFile(file, localUrl, { width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = () => {
      selectFile(file, localUrl);
      setError('Preview is unavailable in this browser for this image. You can still try colorizing; the server will validate the format.');
    };
    image.src = localUrl;
  }
  async function colorize() {
    if (!selectedFile || busy) return;
    busy = true; colorizeButton.disabled = true; progress.hidden = false; setError(''); $('#modelNotice').hidden = true; clearResult();
    try {
      const body = new FormData(); body.append('image', selectedFile);
      const response = await fetch('/api/colorize', { method: 'POST', body });
      if (!response.ok) {
        let message = 'The image could not be colorized. Please try again.';
        try { const payload = await response.json(); message = payload.error?.message || message; } catch (_) {}
        if (response.status === 503) { $('#modelNotice').textContent = message; $('#modelNotice').hidden = false; }
        throw new Error(message);
      }
      const blob = await response.blob();
      if (!blob.type.startsWith('image/')) throw new Error('The server returned an unexpected response.');
      resultUrl = URL.createObjectURL(blob); $('#resultImage').src = resultUrl;
      $('#sideView').hidden = true; $('#compareView').hidden = false;
      $('#viewControls').hidden = !previewAvailable;
      $('#beforeLayer').hidden = !previewAvailable;
      $('#compareDivider').hidden = !previewAvailable;
      $('#compareView input').hidden = !previewAvailable;
      document.querySelectorAll('.compare-label').forEach((label) => { label.hidden = !previewAvailable; });
      if (previewAvailable) {
        $('#beforeImage').src = previewUrl; $('#sideOriginal').src = previewUrl; $('#sideColor').src = resultUrl;
      }
      $('#downloadButton').href = resultUrl;
      const ms = response.headers.get('X-Inference-Time-Ms'); $('#inferenceTime').textContent = ms ? `${Number(ms).toFixed(0)} ms inference` : 'Inference complete';
      const width = response.headers.get('X-Image-Width'), height = response.headers.get('X-Image-Height');
      $('#outputDimensions').textContent = width && height ? `${width} × ${height} px output` : 'Output ready';
      results.hidden = false; results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (error) {
      setError(error.message || 'Network error. Check that the ColorGen server is running and retry.');
    } finally { busy = false; progress.hidden = true; colorizeButton.disabled = !selectedFile; }
  }
  function reset() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null; selectedFile = null; previewAvailable = false; fileInput.value = ''; $('#originalPreview').removeAttribute('src');
    $('#originalPreview').hidden = false; $('#previewFallback').hidden = true;
    $('#selectedPanel').hidden = true; dropzone.hidden = false; colorizeButton.disabled = true;
    $('#viewControls').hidden = false; $('#beforeLayer').hidden = false; $('#compareDivider').hidden = false;
    $('#compareView input').hidden = false; document.querySelectorAll('.compare-label').forEach((label) => { label.hidden = false; });
    $('#sideView').hidden = true; $('#compareView').hidden = false;
    setError(''); $('#modelNotice').hidden = true; clearResult();
  }
  $('#browseButton').addEventListener('click', (event) => { event.stopPropagation(); chooseFile(); });
  dropzone.addEventListener('click', chooseFile); dropzone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); chooseFile(); } });
  fileInput.addEventListener('change', () => loadFile(fileInput.files[0]));
  $('#removeButton').addEventListener('click', reset); $('#anotherButton').addEventListener('click', reset); colorizeButton.addEventListener('click', colorize);
  for (const type of ['dragenter','dragover']) dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.add('dragging'); });
  for (const type of ['dragleave','drop']) dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.remove('dragging'); });
  dropzone.addEventListener('drop', (event) => loadFile(event.dataTransfer.files[0]));
  const range = $('#compareRange'), before = $('#beforeLayer'), divider = $('#compareDivider');
  range.addEventListener('input', () => { const value = `${range.value}%`; before.style.clipPath = `inset(0 ${100 - Number(range.value)}% 0 0)`; divider.style.left = value; });
  document.querySelectorAll('.view-toggle').forEach((button) => button.addEventListener('click', () => {
    document.querySelectorAll('.view-toggle').forEach((item) => {
      const active = item === button;
      item.classList.toggle('active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    const side = button.dataset.view === 'side'; $('#sideView').hidden = !side; $('#compareView').hidden = side;
  }));
  window.addEventListener('beforeunload', () => { if (previewUrl) URL.revokeObjectURL(previewUrl); if (resultUrl) URL.revokeObjectURL(resultUrl); });
  fetch('/api/health').then((response) => response.json()).then((health) => {
    if (Number.isFinite(health.max_upload_bytes) && health.max_upload_bytes > 0) {
      maxUploadBytes = health.max_upload_bytes;
    }
    if (Number.isFinite(health.max_image_pixels) && health.max_image_pixels > 0) maxImagePixels = health.max_image_pixels;
    if (Array.isArray(health.supported_formats) && health.supported_formats.length) {
      const formats = health.supported_formats;
      fileInput.accept = [...new Set(formats.flatMap((item) => [...item.extensions, ...item.mime_types]))].join(',');
      document.querySelector('.format-note').textContent = `${formats.map((item) => item.label).join(', ')} · up to ${Math.floor(maxUploadBytes / 1048576)} MB`;
    }
  }).catch(() => {});
})();
