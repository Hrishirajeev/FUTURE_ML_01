from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from app.models.forecast import SalesForecaster
from app.models.trend_analysis import TrendAnalyzer
from app.utils.data_processor import DataProcessor
import io
from io import StringIO

app = Flask(__name__)
CORS(app)

# Initialize AI models
forecaster = SalesForecaster()
trend_analyzer = TrendAnalyzer()
data_processor = DataProcessor()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'AI Sales Forecasting API is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/sales-data', methods=['GET'])
def get_sales_data():
    """Get sample sales data or uploaded data"""
    try:
        # For demo purposes, generate sample data
        sample_data = data_processor.generate_sample_data()
        if isinstance(sample_data, pd.DataFrame):
            sample_data = sample_data.to_dict(orient='records')
        return jsonify({
            'success': True,
            'data': sample_data,
            'message': 'Sales data retrieved successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        """Generate sales forecast using AI models"""
        try:
            data = request.get_json()
            forecast_periods = data.get('forecast_periods', 12)
            model_type = data.get('model', 'arima').lower()
            store = data.get('store', None)
            department = data.get('department', None)
            start_date = data.get('start_date', None)
            end_date = data.get('end_date', None)

            forecast_result = forecaster.generate_forecast(
                forecast_periods=forecast_periods,
                model_type=model_type,
                store=store,
                department=department,
                start_date=start_date,
                end_date=end_date
            )

            return jsonify({
                'success': True,
                'forecast': forecast_result,
                'message': f'Forecast generated successfully using {model_type} model'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    elif request.method == 'GET':
        # parse query parameters and call the same logic
        store = request.args.get('store')
        department = request.args.get('department')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        model = request.args.get('model')
        # call your forecast logic here and return the result as JSON
        ...

@app.route('/api/trends', methods=['POST'])
def analyze_trends():
    """Analyze sales trends and patterns"""
    try:
        data = request.get_json()
        historical_data = data.get('historical_data', [])
        
        if not historical_data:
            return jsonify({
                'success': False,
                'error': 'Historical data is required'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        
        # Analyze trends
        trend_analysis = trend_analyzer.analyze_trends(df)
        
        return jsonify({
            'success': True,
            'trends': trend_analysis,
            'message': 'Trend analysis completed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    # Get the uploaded file
    file = request.files['file']
    # Read CSV into DataFrame
    df = pd.read_csv(file)
    
    # Standardize column names (case-insensitive, strip spaces)
    df.columns = [col.strip().lower() for col in df.columns]
    
    # Map columns if needed
    col_map = {
        'store': 'store',
        'dept': 'dept',
        'date': 'date',
        'weekly_sales': 'weekly_sales',
        'isholiday': 'isholiday'
    }
    # If your columns are named differently, update col_map accordingly

    # Filter for user-selected store/department (get from request.form or JSON)
    store = int(request.form.get('store'))
    dept = int(request.form.get('department'))
    model = request.form.get('model')
    # ...get other params as needed

    filtered = df[(df[col_map['store']] == store) & (df[col_map['dept']] == dept)]
    # Now use filtered for your model prediction

    # Example: return the first 5 rows as a test
    return jsonify(filtered.head().to_dict(orient='records'))

@app.route('/api/kpis', methods=['POST'])
def get_kpis():
    """
    API endpoint to get KPIs (total, average, max, min sales, etc.)
    for a specific store, department, and date range.
    """
    try:
        data = request.get_json()
        store = data.get('store', None)
        department = data.get('department', None)
        start_date = data.get('start_date', None)
        end_date = data.get('end_date', None)

        # Load the full sales data
        df = forecaster.data_processor.load_sales_data(full=True)

        # Filter by store if provided
        if store is not None:
            df = df[df['Store'] == int(store)]

        # Filter by department if provided
        if department is not None:
            df = df[df['Dept'] == int(department)]

        # Filter by date range if provided
        if start_date is not None:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['Date'] <= pd.to_datetime(end_date)]

        print(df.head(), df.shape)

        # Calculate KPIs
        total_sales = float(df['Weekly_Sales'].sum())
        avg_sales = float(df['Weekly_Sales'].mean())
        max_sales = float(df['Weekly_Sales'].max())
        min_sales = float(df['Weekly_Sales'].min())
        data_points = int(len(df))

        # Find the date with max and min sales
        peak_date = df.loc[df['Weekly_Sales'].idxmax(), 'Date'].strftime('%Y-%m-%d') if data_points > 0 else None
        low_date = df.loc[df['Weekly_Sales'].idxmin(), 'Date'].strftime('%Y-%m-%d') if data_points > 0 else None

        # Prepare the KPIs dictionary
        kpis = {
            'total_sales': total_sales,
            'average_sales': avg_sales,
            'max_sales': max_sales,
            'min_sales': min_sales,
            'data_points': data_points,
            'peak_date': peak_date,
            'low_date': low_date
        }

        return jsonify({
            'success': True,
            'kpis': kpis,
            'message': 'KPIs calculated successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# -------------------------------
# Historical Data API Endpoint
# -------------------------------
@app.route('/api/historical-data', methods=['POST'])
def get_historical_data():
    """
    API endpoint to get historical sales data for a specific store, department, and date range.
    """
    try:
        data = request.get_json()
        store = data.get('store', None)
        department = data.get('department', None)
        start_date = data.get('start_date', None)
        end_date = data.get('end_date', None)

        # Load the full sales data (not aggregated)
        df = forecaster.data_processor.load_sales_data(full=True)

        # Filter by store if provided
        if store is not None:
            df = df[df['Store'] == store]

        # Filter by department if provided
        if department is not None:
            df = df[df['Dept'] == department]

        # Filter by date range if provided
        if start_date is not None:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['Date'] <= pd.to_datetime(end_date)]

        # Prepare the data for output (only keep relevant columns)
        df = df[['Date', 'Store', 'Dept', 'Weekly_Sales']]
        df = df.rename(columns={'Date': 'date', 'Store': 'store', 'Dept': 'department', 'Weekly_Sales': 'sales'})

        # Convert to list of dicts for JSON response
        historical_data = df.to_dict(orient='records')

        return jsonify({
            'success': True,
            'historical_data': historical_data,
            'message': 'Historical sales data retrieved successfully'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# -------------------------------
# Download/Export Data as CSV API Endpoint
# -------------------------------
@app.route('/api/export-csv', methods=['POST'])
def export_csv():
    """
    API endpoint to export historical sales data as a CSV file
    for a specific store, department, and date range.
    """
    try:
        data = request.get_json()
        store = data.get('store', None)
        department = data.get('department', None)
        start_date = data.get('start_date', None)
        end_date = data.get('end_date', None)

        # Load the full sales data
        df = forecaster.data_processor.load_sales_data(full=True)

        # Filter by store if provided
        if store is not None:
            df = df[df['Store'] == store]

        # Filter by department if provided
        if department is not None:
            df = df[df['Dept'] == department]

        # Filter by date range if provided
        if start_date is not None:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['Date'] <= pd.to_datetime(end_date)]

        # Prepare the data for output (only keep relevant columns)
        df = df[['Date', 'Store', 'Dept', 'Weekly_Sales']]
        df = df.rename(columns={'Date': 'date', 'Store': 'store', 'Dept': 'department', 'Weekly_Sales': 'sales'})

        # Convert DataFrame to CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Send the CSV file as a download
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='historical_data.csv'
        )
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/forecast', methods=['GET'])
def forecast_get():
    # Placeholder response or implement as needed
    return jsonify({'success': False, 'error': 'This endpoint is not implemented. Use /api/forecast (POST) instead.'}), 404

@app.route('/upload', methods=['POST'])
def upload():
    # Placeholder response or implement as needed
    return jsonify({'success': False, 'error': 'This endpoint is not implemented. Use /api/upload-data (POST) instead.'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 