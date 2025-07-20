import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

class TrendAnalyzer:
    def __init__(self):
        self.model = LinearRegression()
        
    def analyze_trends(self, df):
        """
        Analyze sales trends and patterns in the data
        
        Args:
            df: DataFrame with 'date' and 'sales' columns
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            # Ensure proper date format
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Calculate basic statistics
            total_sales = df['sales'].sum()
            avg_sales = df['sales'].mean()
            max_sales = df['sales'].max()
            min_sales = df['sales'].min()
            
            # Calculate trend (simple linear regression)
            X = np.arange(len(df)).reshape(-1, 1)
            y = df['sales'].values
            
            self.model.fit(X, y)
            trend_slope = self.model.coef_[0]
            trend_direction = "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
            
            # Calculate growth rate
            if len(df) > 1:
                first_sales = df['sales'].iloc[0]
                last_sales = df['sales'].iloc[-1]
                growth_rate = ((last_sales - first_sales) / first_sales) * 100 if first_sales > 0 else 0
            else:
                growth_rate = 0
            
            # Find peak and low periods
            peak_date = df.loc[df['sales'].idxmax(), 'date'].strftime('%Y-%m-%d')
            low_date = df.loc[df['sales'].idxmin(), 'date'].strftime('%Y-%m-%d')
            
            return {
                'total_sales': float(total_sales),
                'average_sales': float(avg_sales),
                'max_sales': float(max_sales),
                'min_sales': float(min_sales),
                'trend_direction': trend_direction,
                'trend_slope': float(trend_slope),
                'growth_rate_percent': float(growth_rate),
                'peak_date': peak_date,
                'low_date': low_date,
                'data_points': len(df),
                'date_range': {
                    'start': df['date'].min().strftime('%Y-%m-%d'),
                    'end': df['date'].max().strftime('%Y-%m-%d')
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'message': 'Failed to analyze trends'
            } 