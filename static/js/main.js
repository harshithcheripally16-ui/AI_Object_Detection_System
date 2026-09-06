document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('detect-form');
  const imageInput = document.getElementById('image-input');
  const dropzone = document.getElementById('dropzone');
  const previewWrapper = document.getElementById('preview-wrapper');
  const sourcePreview = document.getElementById('source-preview');
  const submitBtn = document.getElementById('submit-btn');
  const btnText = document.getElementById('btn-text');
  const btnSpinner = document.getElementById('btn-spinner');

  const alertContainer = document.getElementById('alert-container');
  const noResultPlaceholder = document.getElementById('no-result-placeholder');
  const resultContent = document.getElementById('result-content');
  const resultImage = document.getElementById('result-image');
  const latencyBadge = document.getElementById('latency-badge');
  const countBadge = document.getElementById('detection-count-badge');
  const modelTagBadge = document.getElementById('model-tag-badge');
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

    // Check size < 16MB
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

    // Set loading state
    submitBtn.disabled = true;
    btnText.textContent = 'Processing Image...';
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

      // Populate detection results
      noResultPlaceholder.classList.add('hidden');
      resultContent.classList.remove('hidden');

      resultImage.src = data.result_image + '?t=' + Date.now();
      latencyBadge.textContent = `⚡ ${data.processing_time_ms} ms`;
      latencyBadge.classList.remove('hidden');

      countBadge.textContent = `${data.count} ${data.stats.total_detections === 1 ? 'Object' : 'Objects'} Detected`;
      modelTagBadge.textContent = data.model_name || data.model_used;

      inspectBtn.href = data.result_page_url;

      // Render coordinates table
      boxesTbody.innerHTML = '';
      if (data.stats.coordinates && data.stats.coordinates.length > 0) {
        data.stats.coordinates.forEach((box, i) => {
          const centroid = data.stats.centroids[i] || { cx: box.x + (box.w / 2), cy: box.y + (box.h / 2) };
          const row = document.createElement('tr');
          row.innerHTML = `
            <td><strong>#${i + 1}</strong></td>
            <td>${box.x}</td>
            <td>${box.y}</td>
            <td>${box.w} px</td>
            <td>${box.h} px</td>
            <td>(${centroid.cx}, ${centroid.cy})</td>
          `;
          boxesTbody.appendChild(row);
        });
      } else {
        boxesTbody.innerHTML = `
          <tr>
            <td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 16px;">
              No matching objects found with selected model.
            </td>
          </tr>
        `;
      }

      showAlert('Detection executed successfully!', 'success');

    } catch (err) {
      console.error(err);
      showAlert(err.message || 'An error occurred while communicating with the server.');
    } finally {
      submitBtn.disabled = false;
      btnText.textContent = 'Run Detection Engine';
      btnSpinner.classList.add('hidden');
    }
  });
});
