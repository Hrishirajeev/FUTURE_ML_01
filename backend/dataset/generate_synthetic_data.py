import pandas as pd
import numpy as np
from datetime import datetime

dates = pd.date_range('2011-01-01', '2025-12-31', freq='D')
data = []
np.random.seed(42)
for date in dates:
    sales = np.random.normal(20000, 5000)
    is_holiday = 'TRUE' if date.weekday() >= 5 else 'FALSE'
    data.append([1, 1, date.strftime('%Y-%m-%d'), round(max(sales, 0), 2), is_holiday])
df = pd.DataFrame(data, columns=['Store','Dept','Date','Weekly_Sales','IsHoliday'])
df.to_csv('backend/dataset/train_extended.csv', index=False) 