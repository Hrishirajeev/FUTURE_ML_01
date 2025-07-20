import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import warnings
from ..utils.data_processor import DataProcessor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')

class SalesForecaster:
    def __init__(self):
        self.scaler = StandardScaler()
        self.linear_model = LinearRegression()
        self.data_processor = DataProcessor()
        
    def generate_forecast(self, forecast_periods=12, model_type='arima', store=None, department=None, start_date=None, end_date=None):
        """
        Generate sales forecast using specified model
        
        Args:
            forecast_periods: Number of days to forecast
            model_type: 'arima', 'linear', 'prophet', or 'ensemble'
            
        Returns:
            List of forecasted sales with dates

        """
        df = self.data_processor.load_sales_data(full=True)
        df = df.rename(columns={'Date': 'date', 'Weekly_Sales': 'sales'})
        df['Store'] = df['Store'].astype(int)
        df['Dept'] = df['Dept'].astype(int)
        print("Before filtering:", df.head(), df.shape)
        if store is not None:
            df = df[df['Store'] == int(store)]
        if department is not None:
            df = df[df['Dept'] == int(department)]
        if start_date is not None:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        df.set_index('date', inplace=True)
        sales_series = df['sales']
        print("After filtering:", df.head(), df.shape)

        # Anomaly detection on historical sales
        sales_mean = sales_series.mean()
        sales_std = sales_series.std()
        anomalies = [
            {'date': date.strftime('%Y-%m-%d'), 'sales': float(value)}
            for date, value in sales_series.items()
            if abs(value - sales_mean) > 2 * sales_std
        ]

        try:
            # Use already filtered df and sales_series
            if model_type == 'arima':
                result = self._arima_forecast(sales_series, forecast_periods)
            elif model_type == 'linear':
                result = self._linear_forecast(sales_series, forecast_periods)
            elif model_type == 'prophet':
                result = self._prophet_forecast(df, forecast_periods)
            elif model_type == 'ensemble':
                result = self._ensemble_forecast(sales_series, forecast_periods)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            # Attach anomalies to the result
            if isinstance(result, dict):
                result['anomalies'] = anomalies
            print('Forecast result:', result)
            return result
                
        except Exception as e:
            print('Forecast error:', str(e))
            return {
                'error': str(e),
                'forecast': [],
                'confidence_intervals': [],
                'model_metrics': {}
            }
    
    def _arima_forecast(self, sales_series, forecast_periods):
        """ARIMA model forecasting with error metrics and auto hyperparameter tuning"""
        try:
            # Auto-tune (p,d,q) in small ranges
            best_aic = float('inf')
            best_order = (1, 1, 1)
            best_model = None
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(sales_series, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                                best_model = fitted
                        except Exception:
                            continue
            if best_model is None:
                raise Exception("No valid ARIMA model found.")

            # In-sample predictions for error metrics
            in_sample_pred = best_model.predict(start=1, end=len(sales_series)-1, typ='levels')
            y_true = sales_series[1:]
            y_pred = in_sample_pred

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            # Forecast
            forecast = best_model.forecast(steps=forecast_periods)
            last_date = sales_series.index[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_periods, freq='D')

            # Confidence intervals
            forecast_ci = best_model.get_forecast(steps=forecast_periods)
            conf_int = forecast_ci.conf_int()

            # Return as list of dicts, plus metrics
            return {
                'forecast': [
                    {'date': date.strftime('%Y-%m-%d'), 'forecasted_sales': float(value), 'lower_ci': float(conf_int.iloc[i, 0]), 'upper_ci': float(conf_int.iloc[i, 1])}
                    for i, (date, value) in enumerate(zip(future_dates, forecast))
                ],
                'model_metrics': {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'aic': float(best_model.aic),
                    'bic': float(best_model.bic),
                    'model_type': f'ARIMA{best_order}'
                }
            }
        except Exception as e:
            return {'error': f'ARIMA forecast failed: {str(e)}'}
    
    def _linear_forecast(self, sales_series, forecast_periods):
        """Linear regression forecasting"""
        try:
            # Create features
            X = np.arange(len(sales_series)).reshape(-1, 1)
            y = sales_series.values
            
            # Fit linear model
            self.linear_model.fit(X, y)
            
            # Generate future features
            future_X = np.arange(len(sales_series), len(sales_series) + forecast_periods).reshape(-1, 1)
            future_forecast = self.linear_model.predict(future_X)
            
            # Generate future dates
            last_date = sales_series.index[-1]
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_periods,
                freq='D'
            )
            
            # Calculate confidence intervals (simplified)
            residuals = y - self.linear_model.predict(X)
            std_residuals = np.std(residuals)
            
            # Return as list of dicts
            return [
                {'date': date.strftime('%Y-%m-%d'), 'forecasted_sales': float(value), 'lower_ci': float(value - 1.96 * std_residuals), 'upper_ci': float(value + 1.96 * std_residuals)}
                for date, value in zip(future_dates, future_forecast)
            ]
        except Exception as e:
            return {'error': f'Linear forecast failed: {str(e)}'}
    
    def _prophet_forecast(self, df, forecast_periods):
        """Prophet model forecasting with simple hyperparameter tuning"""
        try:
            from prophet import Prophet
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            import numpy as np

            sales_series = df['sales']
            prophet_df = df.reset_index()[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
            best_mae = float('inf')
            best_params = None
            best_model = None
            best_forecast = None
            best_y_true = None
            best_y_pred = None
            param_grid = [
                {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.05},
                {'seasonality_mode': 'additive', 'changepoint_prior_scale': 0.5},
                {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.05},
                {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.5},
            ]
            for params in param_grid:
                try:
                    model = Prophet(seasonality_mode=params['seasonality_mode'], changepoint_prior_scale=params['changepoint_prior_scale'])
                    model.fit(prophet_df)
                    future = model.make_future_dataframe(periods=forecast_periods)
                    forecast = model.predict(future)
                    # In-sample predictions
                    y_true = prophet_df['y']
                    y_pred = forecast['yhat'][:len(y_true)]
                    mae = mean_absolute_error(y_true, y_pred)
                    if mae < best_mae:
                        best_mae = mae
                        best_params = params
                        best_model = model
                        best_forecast = forecast
                        best_y_true = y_true
                        best_y_pred = y_pred
                except Exception:
                    continue
            if best_model is None:
                raise Exception("No valid Prophet model found.")
            # Prepare forecast output
            forecast_rows = []
            for i in range(len(prophet_df), len(best_forecast)):
                row = best_forecast.iloc[i]
                forecast_rows.append({
                    'date': row['ds'].strftime('%Y-%m-%d'),
                    'forecasted_sales': float(row['yhat']),
                    'lower_ci': float(row['yhat_lower']),
                    'upper_ci': float(row['yhat_upper'])
                })
            rmse = np.sqrt(mean_squared_error(best_y_true, best_y_pred))
            return {
                'forecast': forecast_rows,
                'model_metrics': {
                    'mae': float(best_mae),
                    'rmse': float(rmse),
                    'aic': None,
                    'bic': None,
                    'model_type': f"Prophet({best_params})"
                }
            }
        except Exception as e:
            print('Prophet error:', str(e))
            return {'error': f'Prophet forecast failed: {str(e)}'}
    
    def _ensemble_forecast(self, sales_series, forecast_periods):
        """Ensemble forecasting combining multiple models"""
        try:
            # Get forecasts from different models
            arima_result = self._arima_forecast(sales_series, forecast_periods)
            linear_result = self._linear_forecast(sales_series, forecast_periods)
            
            if 'error' in arima_result or 'error' in linear_result:
                # Fallback to best available model
                if 'error' not in linear_result:
                    return linear_result
                elif 'error' not in arima_result:
                    return arima_result
                else:
                    return {'error': 'All models failed'}
            
            # Combine forecasts (simple average)
            ensemble_forecast = []
            for i in range(forecast_periods):
                arima_val = arima_result['forecast'][i]['forecasted_sales']
                linear_val = linear_result[i]['forecasted_sales']
                
                ensemble_forecast.append({
                    'date': arima_result['forecast'][i]['date'],
                    'forecasted_sales': float((arima_val + linear_val) / 2),
                    'lower_ci': float(min(arima_result['forecast'][i]['lower_ci'], 
                                        linear_result[i]['lower_ci'])),
                    'upper_ci': float(max(arima_result['forecast'][i]['upper_ci'], 
                                        linear_result[i]['upper_ci']))
                })
            
            return {
                'forecast': ensemble_forecast,
                'model_metrics': {
                    'mae': float(arima_result['model_metrics']['mae']),
                    'rmse': float(arima_result['model_metrics']['rmse']),
                    'aic': float(arima_result['model_metrics']['aic']),
                    'bic': float(arima_result['model_metrics']['bic']),
                    'model_type': 'Ensemble'
                }
            }
        except Exception as e:
            return {'error': f'Ensemble forecast failed: {str(e)}'} 