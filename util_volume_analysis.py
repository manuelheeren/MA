from pathlib import Path
import pandas as pd
from data_handler import DukasCopyDataHandler

def analyze_volume():
    handler = DukasCopyDataHandler()
    data_path = Path("datasets/raw/XAUUSD/Nov23-Nov24/XAUUSD_1M_BID.csv")
    data = handler.load_data(data_path)
    
    # Find zero volume periods
    zero_volume = data[data['volume'] == 0]
    
    # Group by date to find daily patterns
    zero_volume_by_day = zero_volume.groupby('date').agg({
        'volume': 'count',  # Count minutes with zero volume
        'asian_session': 'any',
        'london_session': 'any',
        'us_session': 'any'
    })
    
    print("\nDays with Zero Volume Periods:")
    for date, row in zero_volume_by_day.iterrows():
        print(f"\nDate: {date}")
        print(f"Minutes with zero volume: {row['volume']}")
        print("Sessions affected:")
        if row['asian_session']: print("- Asian")
        if row['london_session']: print("- London")
        if row['us_session']: print("- US")
        
        # Get time ranges for this day
        day_data = zero_volume[zero_volume['date'] == date]
        times = day_data.index.time
        print(f"Time range: {min(times)} - {max(times)}")

if __name__ == '__main__':
    analyze_volume()