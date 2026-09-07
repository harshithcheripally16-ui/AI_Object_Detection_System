document.addEventListener('DOMContentLoaded', () => {
  let classesChartInstance = null;
  let timelineChartInstance = null;

  const alertContainer = document.getElementById('analytics-alert');
  const refreshBtn = document.getElementById('refresh-logs-btn');

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
      renderClassesChart(data.class_distribution || []);
      renderTimelineChart(data.timeline || []);
    } catch (err) {
      console.error('Error fetching analytics:', err);
    }
  }

  function updateKPIs(data) {
    const elImages = document.getElementById('kpi-total-images');
    const elDetections = document.getElementById('kpi-total-detections');
    const elAvgLat = document.getElementById('kpi-avg-latency');
    const elTopClass = document.getElementById('kpi-top-class');

    if (elImages) elImages.textContent = data.total_images;
    if (elDetections) elDetections.textContent = data.total_objects;
    if (elAvgLat) elAvgLat.innerHTML = `${data.avg_processing_time_ms} <span style="font-size: 1rem; font-weight: normal;">ms</span>`;
    if (elTopClass) elTopClass.textContent = data.top_class || 'None';
  }

  function renderClassesChart(distribution) {
    const ctx = document.getElementById('classesBarChart');
    if (!ctx) return;

    const labels = distribution.map(item => item.class.toUpperCase());
    const dataVals = distribution.map(item => item.count);

    if (classesChartInstance) {
      classesChartInstance.destroy();
    }

    classesChartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels.length > 0 ? labels : ['No Detections'],
        datasets: [{
          label: 'Total Detected Instances',
          data: dataVals.length > 0 ? dataVals : [0],
          backgroundColor: 'rgba(56, 189, 248, 0.75)',
          borderColor: '#38bdf8',
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            beginAtZero: true,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', stepSize: 1 }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  function renderTimelineChart(timeline) {
    const ctx = document.getElementById('timelineChart');
    if (!ctx) return;

    const labels = timeline.map(item => `${item.id}`);
    const latencies = timeline.map(item => item.latency_ms);
    const counts = timeline.map(item => item.total_count);

    if (timelineChartInstance) {
      timelineChartInstance.destroy();
    }

    timelineChartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels.length > 0 ? labels : ['No Runs'],
        datasets: [
          {
            label: 'Processing Time (ms)',
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
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
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
            title: { display: true, text: 'Objects Count', color: '#10b981' },
            grid: { drawOnChartArea: false },
            ticks: { color: '#10b981', stepSize: 1 }
          }
        },
        plugins: {
          legend: { labels: { color: '#f8fafc' } }
        }
      }
    });
  }

  function renumberTableRows() {
    const rows = document.querySelectorAll('#history-tbody tr:not(#empty-row)');
    rows.forEach((row, index) => {
      const idxCell = row.querySelector('.row-index strong');
      if (idxCell) {
        idxCell.textContent = index + 1;
      }
    });
  }

  // Handle Delete Actions
  document.addEventListener('click', async (e) => {
    if (e.target && e.target.classList.contains('delete-btn')) {
      const recordId = e.target.getAttribute('data-id');
      if (!confirm(`Are you sure you want to delete YOLOv8 Record ${recordId}?`)) return;

      try {
        const res = await fetch(`/api/history/${recordId}`, { method: 'DELETE' });
        const result = await res.json();
        if (res.ok) {
          const row = document.getElementById(`row-${recordId}`);
          if (row) row.remove();
          renumberTableRows();
          showAlert(`Record ${recordId} deleted successfully.`, 'success');
          fetchAnalyticsData();
        } else {
          showAlert(result.error || 'Failed to delete record.', 'danger');
        }
      } catch (err) {
        showAlert('Network error while deleting record.', 'danger');
      }
    }
  });

  const clearAllBtn = document.getElementById('clear-all-btn');
  if (clearAllBtn) {
    clearAllBtn.addEventListener('click', async () => {
      if (!confirm('Are you sure you want to CLEAR ALL detection history and reset ID counter back to 1? This cannot be undone.')) return;

      try {
        const res = await fetch('/api/history/clear-all', { method: 'DELETE' });
        const result = await res.json();
        if (res.ok) {
          const tbody = document.getElementById('history-tbody');
          if (tbody) {
            tbody.innerHTML = `
              <tr id="empty-row">
                <td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 32px;">
                  No detection logs found. Upload and process images to populate the database.
                </td>
              </tr>
            `;
          }
          showAlert('All history cleared and ID counter reset to 1.', 'success');
          fetchAnalyticsData();
        } else {
          showAlert(result.error || 'Failed to clear history.', 'danger');
        }
      } catch (err) {
        showAlert('Network error while clearing history.', 'danger');
      }
    });
  }

  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      window.location.reload();
    });
  }

  // Initial load
  fetchAnalyticsData();
});
