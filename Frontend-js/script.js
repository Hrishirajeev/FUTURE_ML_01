let latestForecast = [];
let latestErrorMetrics = {};
let latestKPIs = {};
let forecastChart = null;
let latestAnomalies = [];

function showSection(sectionId) {
  document.getElementById('kpi-section').style.display = 'none';
  document.getElementById('chart-section').style.display = 'none';
  document.getElementById('table-section').style.display = 'none';
  document.getElementById('error-section').style.display = 'none';
  document.getElementById('anomaly-section').style.display = 'none'; // Added anomaly section
  document.getElementById(sectionId).style.display = 'block';

  if (sectionId === 'kpi-section') renderKPIs(latestKPIs);
  if (sectionId === 'chart-section') renderForecastChart(latestForecast);
  if (sectionId === 'table-section') renderForecastTable(latestForecast);
  if (sectionId === 'error-section') renderErrorMetrics(latestErrorMetrics);
  if (sectionId === 'anomaly-section') renderAnomalyAlerts(latestAnomalies); // Added anomaly section
}

function getForecast(event) {
  if (event) event.preventDefault();
  document.getElementById('error-message').textContent = '';
  latestForecast = [];
  latestErrorMetrics = {};
  latestKPIs = {};
  latestAnomalies = [];

  const store = document.getElementById('store').value;
  const department = document.getElementById('department').value;
  const startDate = document.getElementById('startDate').value;
  const endDate = document.getElementById('endDate').value;
  const model = document.getElementById('model').value;

  fetch('http://127.0.0.1:5000/api/forecast', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      store: store,
      dept: department,
      start_date: startDate,
      end_date: endDate,
      model: model
    })
  })
    .then(response => response.json())
    .then(data => {
      // Extract forecast and metrics
      latestForecast = (data.forecast && data.forecast.forecast) ? data.forecast.forecast : [];
      latestKPIs = (data.forecast && data.forecast.model_metrics) ? data.forecast.model_metrics : {};
      latestErrorMetrics = (data.forecast && data.forecast.model_metrics) ? data.forecast.model_metrics : {};
      latestAnomalies = (data.forecast && data.forecast.anomalies) ? data.forecast.anomalies : [];

      // Render all sections
      renderForecastTable(latestForecast);
      renderForecastChart(latestForecast);
      renderKPIs(latestKPIs);
      renderErrorMetrics(latestErrorMetrics);
      renderAnomalyAlerts(latestAnomalies);

      // Show chart by default
      showSection('chart-section');
    })
    .catch(error => {
      document.getElementById('error-message').textContent = "Error: " + error;
      latestForecast = [];
      latestErrorMetrics = {};
      latestKPIs = {};
      latestAnomalies = []; // Clear anomalies on error
      showSection('chart-section');
    });
}

function renderForecastTable(forecastArray) {
  const table = document.getElementById('forecast-table');
  console.log('Rendering table with:', forecastArray);
  if (!forecastArray || forecastArray.length === 0) {
    table.innerHTML = '<tr><td colspan="4">No forecast data available.</td></tr>';
    return;
  }
  let html = '<tr><th>Date</th><th>Forecasted Sales</th><th>Lower CI</th><th>Upper CI</th></tr>';
  forecastArray.forEach(item => {
    html += `<tr>
      <td>${item.date}</td>
      <td>${item.forecasted_sales}</td>
      <td>${item.lower_ci}</td>
      <td>${item.upper_ci}</td>
    </tr>`;
  });
  table.innerHTML = html;
}

function renderForecastChart(forecast) {
  const ctx = document.getElementById('forecastChart').getContext('2d');
  if (!forecast || forecast.length === 0) {
    if (forecastChart instanceof Chart) {
      forecastChart.destroy();
    }
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    forecastChart = new Chart(ctx, {
      type: 'line',
      data: { labels: [], datasets: [] },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          title: { display: true, text: 'No forecast data available.' }
        },
        scales: {
          x: { display: false },
          y: { display: false }
        }
      }
    });
    return;
  }

  const labels = forecast.map(item => item.date);
  const sales = forecast.map(item => item.forecasted_sales);
  const lower = forecast.map(item => item.lower_ci);
  const upper = forecast.map(item => item.upper_ci);

  if (forecastChart instanceof Chart) {
    forecastChart.destroy();
  }

  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Forecasted Sales',
          data: sales,
          borderColor: '#2d3e50',
          backgroundColor: 'rgba(45,62,80,0.1)',
          fill: false,
          tension: 0.2
        },
        {
          label: 'Lower CI',
          data: lower,
          borderColor: '#4caf50',
          borderDash: [5, 5],
          fill: false,
          tension: 0.2
        },
        {
          label: 'Upper CI',
          data: upper,
          borderColor: '#f44336',
          borderDash: [5, 5],
          fill: false,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'top' },
        title: { display: true, text: 'Sales Forecast Trend' }
      }
    }
  });
}

function renderKPIs(kpis) {
  const kpiDiv = document.getElementById('kpi-section');
  kpiDiv.innerHTML = '';
  if (!kpis || Object.keys(kpis).length === 0) {
    kpiDiv.innerHTML = '<p style="color:#888;">No KPIs available.</p>';
    return;
  }
  // Add predicted sales KPI if forecast data is available
  if (latestForecast && latestForecast.length > 0) {
    const predictedTotal = latestForecast.reduce((sum, item) => sum + (item.forecasted_sales || 0), 0);
    kpiDiv.innerHTML += `<div class="kpi-card"><strong>predicted sales:</strong> ${predictedTotal.toFixed(2)}</div>`;
  }
  for (const [key, value] of Object.entries(kpis)) {
    kpiDiv.innerHTML += `<div class="kpi-card"><strong>${key.replace(/_/g, ' ')}:</strong> ${value}</div>`;
  }
}

function renderErrorMetrics(metrics) {
  const errorDiv = document.getElementById('error-section');
  errorDiv.innerHTML = '';
  if (!metrics || Object.keys(metrics).length === 0) {
    errorDiv.innerHTML = '<p style="color:#888;">No error metrics available.</p>';
    return;
  }
  for (const [key, value] of Object.entries(metrics)) {
    errorDiv.innerHTML += `<div class="error-card"><strong>${key.replace(/_/g, ' ')}:</strong> ${value}</div>`;
  }
}

function renderAnomalyAlerts(anomalies) {
  const anomalyDiv = document.getElementById('anomaly-section');
  anomalyDiv.innerHTML = '';
  if (!anomalies || anomalies.length === 0) {
    anomalyDiv.innerHTML = '<p style="color:#888;">No anomalies detected.</p>';
    return;
  }
  anomalyDiv.innerHTML = '<h3 style="color:#d32f2f;">Anomaly Alerts</h3>';
  anomalies.forEach(item => {
    anomalyDiv.innerHTML += `<div class="error-card"><strong>Date:</strong> ${item.date} &nbsp; <strong>Sales:</strong> ${item.sales}</div>`;
  });
}

// Attach event listeners
window.onload = function() {
  showSection('chart-section');
  document.getElementById('get-forecast-btn').onclick = getForecast;
  document.getElementById('show-kpi-btn').onclick = function() {
    // Fetch KPIs from backend
    const store = document.getElementById('store').value;
    const department = document.getElementById('department').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    fetch('http://127.0.0.1:5000/api/kpis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        store: store,
        department: department,
        start_date: startDate,
        end_date: endDate
      })
    })
      .then(response => response.json())
      .then(data => {
        latestKPIs = data.kpis || {};
        renderKPIs(latestKPIs);
        showSection('kpi-section');
      })
      .catch(error => {
        latestKPIs = {};
        renderKPIs(latestKPIs);
        showSection('kpi-section');
      });
  };
  document.getElementById('show-chart-btn').onclick = () => showSection('chart-section');
  document.getElementById('show-table-btn').onclick = () => showSection('table-section');
  document.getElementById('show-error-btn').onclick = () => showSection('error-section');
  // Add anomaly button
  const anomalyBtn = document.createElement('button');
  anomalyBtn.textContent = 'Show Anomaly Alerts';
  anomalyBtn.type = 'button';
  anomalyBtn.onclick = () => showSection('anomaly-section');
  document.querySelector('.section-buttons').appendChild(anomalyBtn);
};