document.addEventListener('DOMContentLoaded', () => {
  let timelineChartInstance = null;
  let modelsChartInstance = null;

  const alertContainer = document.getElementById('analytics-alert');
  const refreshBtn = document.getElementById('refresh-logs-btn');
  const historyTbody = document.getElementById('history-tbody');

  function showAlert(message, type = 'success') {
    if (!alertContainer) return;
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
    }, 5000);
  }

  async function fetchAnalyticsData() {
    try {
      const res = await fetch('/api/analytics-data');
      if (!res.ok) throw new Error('Failed to fetch analytics');
      const data = await res.json();

      updateKPIs(data);
      renderTimelineChart(data.timeline || []);
      renderModelsChart(data.models_breakdown || []);
    } catch (err) {
      console.error('Error fetching analytics:', err);
    }
  }

  function updateKPIs(data) {
    const elImages = document.getElementById('kpi-total-images');
    const elDetections = document.getElementById('kpi-total-detections');
    const elAvgDet = document.getElementById('kpi-avg-detections');
    const elAvgLat = document.getElementById('kpi-avg-latency');

    if (elImages) elImages.textContent = data.total_images;
    if (elDetections) elDetections.textContent = data.total_detections;
    if (elAvgDet) elAvgDet.textContent = data.avg_detections;
    if (elAvgLat) elAvgLat.innerHTML = `${data.avg_latency_ms} <span style="font-size: 1rem; font-weight: normal;">ms</span>`;
  }

  function renderTimelineChart(timeline) {
    const ctx = document.getElementById('timelineChart');
    if (!ctx) return;

    const labels = timeline.map(item => `#${item.id}`);
    const latencies = timeline.map(item => item.latency_ms);
    const counts = timeline.map(item => item.count);

    if (timelineChartInstance) {
      timelineChartInstance.destroy();
    }

    timelineChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels.length > 0 ? labels : ['No Runs'],
        datasets: [
          {
            label: 'Inference Latency (ms)',
            data: latencies.length > 0 ? latencies : [0],
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            yAxisID: 'yLatency',
            tension: 0.3,
            fill: true
          },
          {
            label: 'Objects Count',
            data: counts.length > 0 ? counts : [0],
            borderColor: '#38bdf8',
            backgroundColor: 'rgba(56, 189, 248, 0.1)',
            yAxisID: 'yCount',
            tension: 0.3,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          yLatency: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Latency (ms)', color: '#f59e0b' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#f59e0b' }
          },
          yCount: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Detections Count', color: '#38bdf8' },
            grid: { drawOnChartArea: false },
            ticks: { color: '#38bdf8', stepSize: 1 }
          }
        },
        plugins: {
          legend: { labels: { color: '#f8fafc' } }
        }
      }
    });
  }

  function renderModelsChart(breakdown) {
    const ctx = document.getElementById('modelsChart');
    if (!ctx) return;

    const labels = breakdown.map(item => item.model.toUpperCase());
    const dataVals = breakdown.map(item => item.runs);

    if (modelsChartInstance) {
      modelsChartInstance.destroy();
    }

    modelsChartInstance = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels.length > 0 ? labels : ['No Models Run'],
        datasets: [{
          data: dataVals.length > 0 ? dataVals : [1],
          backgroundColor: ['#38bdf8', '#10b981', '#f59e0b', '#a855f7'],
          borderWidth: 1,
          borderColor: '#1e293b'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#f8fafc', padding: 16 } }
        }
      }
    });
  }

  // Handle Delete Actions
  document.addEventListener('click', async (e) => {
    if (e.target && e.target.classList.contains('delete-btn')) {
      const recordId = e.target.getAttribute('data-id');
      if (!confirm(`Are you sure you want to delete Detection Record #${recordId}?`)) return;

      try {
        const res = await fetch(`/api/history/${recordId}`, { method: 'DELETE' });
        const result = await res.json();
        if (res.ok) {
          const row = document.getElementById(`row-${recordId}`);
          if (row) row.remove();
          showAlert(`Record #${recordId} deleted successfully.`, 'success');
          fetchAnalyticsData();
        } else {
          showAlert(result.error || 'Failed to delete record.', 'danger');
        }
      } catch (err) {
        showAlert('Network error while deleting record.', 'danger');
      }
    }
  });

  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      window.location.reload();
    });
  }

  // Initial load
  fetchAnalyticsData();
});
