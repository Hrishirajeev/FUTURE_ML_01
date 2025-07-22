AI Sales Forecasting Dashboard
A modern AI-powered sales forecasting dashboard using Python Flask (backend) and vanilla JavaScript/HTML/CSS (frontend).

Features
Interactive Dashboard: Visualize and explore sales data
AI Forecasting: Predict future sales using ARIMA and Prophet
Anomaly Detection: Highlight unusual sales spikes/drops
KPI Tracking: View key performance indicators
Automatic Model Tuning: Auto-selects best ARIMA/Prophet settings
CSV Upload: Import your own sales data
Export: Download historical data as CSV
Responsive UI: Works on desktop and mobile
Tech Stack
Backend: Python Flask, Pandas, NumPy, Prophet, statsmodels (ARIMA)
Frontend: HTML, CSS, JavaScript (vanilla), Chart.js
Project Structure
AISalesForecastDashbord/
├── backend/
│   ├── app.py            # Flask app and API endpoints
│   ├── app/
│   │   ├── models/       # ML models (forecast.py, trend_analysis.py)
│   │   └── utils/        # Data processing helpers
│   └── dataset/          # Sales data CSVs
├── Frontend-js/
│   ├── index.html        # Main dashboard UI
│   ├── script.js         # Frontend logic
│   └── style.css         # Dashboard styles
└── README.md
Getting Started
Prerequisites
Python 3.8+
pip
Installation
Clone the repository

git clone <repository-url>
cd AISalesForecastDashbord
Setup Backend

cd backend
pip install -r requirements.txt
python app.py
Open Frontend

Open Frontend-js/index.html in your browser
Usage
Select store, department, date range, and model
Click 'Get Forecast' to generate predictions
View KPIs, error metrics, anomaly alerts, and charts
Upload your own CSV data if needed
API Endpoints
GET /api/sales-data - Retrieve sample sales data
POST /api/forecast - Generate sales forecasts (ARIMA/Prophet)
POST /api/kpis - Get KPIs for selected range
POST /api/historical-data - Get historical sales data
POST /api/export-csv - Download filtered data as CSV
POST /api/upload-data - Upload new sales data
License
MIT License - see LICENSE file for details

##author hrishi
