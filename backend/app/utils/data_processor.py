import pandas as pd
import os

class DataProcessor:
    def __init__(self):
        # Path to the dataset
        self.data_path = os.path.join(os.path.dirname(__file__), '../../dataset/train_extended.csv')

    def load_sales_data(self, full=False):
        """Load and prepare sales data from train.csv"""
        try:
            df = pd.read_csv(self.data_path)
            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            if full:
                return df
            # Otherwise, aggregate as before
            daily_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
            daily_sales = daily_sales.rename(columns={'Date': 'date', 'Weekly_Sales': 'sales'})
            return daily_sales
        except Exception as e:
            print(f"Error loading data: {e}")
            # Return sample data if file not found
            return self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample sales data for testing"""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate 30 days of sample data
        dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
        sales = np.random.normal(1000, 200, 30)  # Random sales around 1000
        
        sample_data = pd.DataFrame({
            'date': dates,
            'sales': sales
        })
        return sample_data.to_dict(orient='records')

    def process_uploaded_file(self, file):
        """Process uploaded CSV file"""
        try:
            df = pd.read_csv(file)
            # Basic validation
            if 'Date' in df.columns and 'Weekly_Sales' in df.columns:
                return self.load_sales_data().to_dict(orient='records')
            else:
                return {'error': 'File must contain Date and Weekly_Sales columns'}
        except Exception as e:
            return {'error': f'Error processing file: {str(e)}'}

    def calculate_kpis(self, sales_data):
        """Calculate key performance indicators"""
        try:
            df = pd.DataFrame(sales_data)
            total_sales = df['sales'].sum()
            avg_sales = df['sales'].mean()
            max_sales = df['sales'].max()
            min_sales = df['sales'].min()
            
            return {
                'total_sales': float(total_sales),
                'average_sales': float(avg_sales),
                'max_sales': float(max_sales),
                'min_sales': float(min_sales),
                'data_points': len(df)
            }
        except Exception as e:
            return {'error': f'Error calculating KPIs: {str(e)}'} 