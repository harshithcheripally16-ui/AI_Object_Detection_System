document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('detect-form');
  const imageInput = document.getElementById('image-input');
  const dropzone = document.getElementById('dropzone');
  const previewWrapper = document.getElementById('preview-wrapper');
  const sourcePreview = document.getElementById('source-preview');
  const confidenceSlider = document.getElementById('confidence-slider');
  const confidenceVal = document.getElementById('confidence-val');
  const submitBtn = document.getElementById('submit-btn');
  const btnText = document.getElementById('btn-text');
  const btnSpinner = document.getElementById('btn-spinner');

  const alertContainer = document.getElementById('alert-container');
  const noResultPlaceholder = document.getElementById('no-result-placeholder');
  const resultContent = document.getElementById('result-content');
  const resultImage = document.getElementById('result-image');
  const latencyBadge = document.getElementById('latency-badge');
  const countBadge = document.getElementById('detection-count-badge');
  const confBadge = document.getElementById('conf-badge');
  const inspectBtn = document.getElementById('inspect-btn');
  const boxesTbody = document.getElementById('boxes-tbody');

  function showAlert(message, type = 'danger') {
    alertContainer.innerHTML = `
      <div class="alert alert-${type}">
        <span>${message}</span>
        <button onclick="this.parentElement.remove()" style="background: none; border: none; color: inherit; font-size: 1.2rem; cursor: pointer;">&times;</button>
      </div>
    `;
    setTimeout(() => {
      if (alertContainer.firstElementChild) {
        alertContainer.firstElementChild.remove();
      }
    }, 6000);
  }

  // Confidence Slider event
  if (confidenceSlider && confidenceVal) {
    confidenceSlider.addEventListener('input', () => {
      confidenceVal.textContent = `${Math.round(confidenceSlider.value * 100)}%`;
    });
  }

  // Handle Drag & Drop styling
  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag-over');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      imageInput.files = e.dataTransfer.files;
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  imageInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  function handleFileSelection(file) {
    if (!file) return;

    if (file.size > 16 * 1024 * 1024) {
      showAlert('File size exceeds the 16MB limit. Please upload a smaller image.');
      imageInput.value = '';
      previewWrapper.classList.add('hidden');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      sourcePreview.src = event.target.result;
      previewWrapper.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  // Form submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (!imageInput.files || imageInput.files.length === 0) {
      showAlert('Please choose or drop an image file first.');
      return;
    }

    const formData = new FormData(form);

    submitBtn.disabled = true;
    btnText.textContent = 'Running YOLOv8 Inference...';
    btnSpinner.classList.remove('hidden');

    try {
      const response = await fetch('/detect', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP Error ${response.status}`);
      }

      noResultPlaceholder.classList.add('hidden');
      resultContent.classList.remove('hidden');

      resultImage.src = data.result_image + '?t=' + Date.now();
      latencyBadge.textContent = `⚡ ${data.processing_time_ms} ms`;
      latencyBadge.classList.remove('hidden');

      countBadge.textContent = `${data.total_count} ${data.total_count === 1 ? 'Object' : 'Objects'} Detected`;
      confBadge.textContent = `Conf: ${Math.round(data.confidence_used * 100)}%`;

      inspectBtn.href = data.result_page_url;

      // Render coordinates table
      boxesTbody.innerHTML = '';
      const items = data.detections || (data.stats ? data.stats.objects : []);

      if (items && items.length > 0) {
        items.forEach((obj, i) => {
          const row = document.createElement('tr');
          const bboxStr = obj.bbox ? `[${obj.bbox.join(', ')}]` : '-';
          const confPct = Math.round(obj.confidence * 100);
          row.innerHTML = `
            <td><strong>#${i + 1}</strong></td>
            <td><span class="badge badge-success" style="font-size: 0.8rem; text-transform: uppercase;">${obj.class}</span></td>
            <td><code>${confPct}%</code></td>
            <td><code>${bboxStr}</code></td>
          `;
          boxesTbody.appendChild(row);
        });
      } else {
        boxesTbody.innerHTML = `
          <tr>
            <td colspan="4" style="text-align: center; color: var(--text-secondary); padding: 16px;">
              No objects detected above the confidence threshold. Try lowering the threshold slider.
            </td>
          </tr>
        `;
      }

      showAlert('YOLOv8 Detection completed successfully!', 'success');

    } catch (err) {
      console.error(err);
      showAlert(err.message || 'An error occurred while communicating with the server.');
    } finally {
      submitBtn.disabled = false;
      btnText.textContent = 'Run YOLOv8 Detection Engine';
      btnSpinner.classList.add('hidden');
    }
  });
});
