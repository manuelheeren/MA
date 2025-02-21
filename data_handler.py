import pandas as pd
from pathlib import Path
import logging
from datetime import time
from enum import Enum
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Asset(Enum):
    """Trading assets with their specific properties"""
    XAUUSD = {
        "name": "XAUUSD",
        "files": [
            "datasets/raw/XAUUSD/Nov23-Nov24/XAUUSD_1M_BID.csv"
        ]
    }
    BTCUSD = {
        "name": "BTCUSD",
        "files": [
            "datasets/raw/BTCUSD/Jan22-Oct22/BTCUSD_1M_BID_01.01.2022-31.10.2022.csv",
            "datasets/raw/BTCUSD/Nov22-Oct23/BTCUSD_1M_BID_01.11.2022-31.10.2023.csv",
            "datasets/raw/BTCUSD/Nov23-Oct24/BTCUSD_1M_BID_01.11.2023-01.11.2024.csv"
        ]
    }
    SPY = {
        "name": "SPY",
        "files": [
            "datasets/raw/SPY/Jan22-Jun22/SPY.USUSD_Candlestick_1_M_BID_01.01.2020-30.06.2020.csv",
            "datasets/raw/SPY/Jul22-Dec22/SPY.USUSD_Candlestick_1_M_BID_01.07.2020-31.12.2020.csv",
            "datasets/raw/SPY/Jan23-Jun23/SPY.USUSD_Candlestick_1_M_BID_01.01.2021-30.06.2021.csv",
            "datasets/raw/SPY/Jul23-Dec23/SPY.USUSD_Candlestick_1_M_BID_01.07.2021-31.12.2021.csv",
            "datasets/raw/SPY/Jan22-Jun22/SPY.USUSD_Candlestick_1_M_BID_01.01.2022-30.06.2022.csv",
            "datasets/raw/SPY/Jul22-Dec22/SPY.USUSD_Candlestick_1_M_BID_01.07.2022-31.12.2022.csv",
            "datasets/raw/SPY/Jan23-Jun23/SPY.USUSD_Candlestick_1_M_BID_01.01.2023-30.06.2023.csv",
            "datasets/raw/SPY/Jul23-Dec23/SPY.USUSD_Candlestick_1_M_BID_01.07.2023-31.12.2023.csv",
            "datasets/raw/SPY/Jan24-Jun24/SPY.USUSD_Candlestick_1_M_BID_01.01.2024-30.06.2024.csv",
            "datasets/raw/SPY/Jul24-Nov24/SPY.USUSD_Candlestick_1_M_BID_01.07.2024-30.11.2024.csv"
        ]
    }
    WTI = {
        "name": "WTI",
        "files": [
            "datasets/raw/WTI/LIGHT.CMDUSD_Candlestick_1_M_BID_01.01.2020-30.06.2020.csv",
            "datasets/raw/WTI/LIGHT.CMDUSD_Candlestick_1_M_BID_01.07.2020-31.12.2020.csv",
            "datasets/raw/WTI/LIGHT.CMDUSD_Candlestick_1_M_BID_01.01.2021-30.06.2021.csv",
            "datasets/raw/WTI/LIGHT.CMDUSD_Candlestick_1_M_BID_01.07.2021-31.12.2021.csv",
            "datasets/raw/WTI/LIGHT.CMDUSD_Candlestick_1_M_BID_01.01.2022-30.06.2022.csv",
            "datasets/raw/WTI/LIGHT.CMDUSD_Candlestick_1_M_BID_01.07.2022-31.12.2022.csv",
            "datasets/raw/WTI/LIGHT.CMDUSD_Candlestick_1_M_BID_01.01.2023-30.06.2023.csv",
            "datasets/raw/WTI/LIGHT.CMDUSD_Candlestick_1_M_BID_01.07.2023-31.12.2023.csv",
            "datasets/raw/WTI/LIGHT.CMDUSD_Candlestick_1_M_BID_01.01.2024-30.06.2024.csv",
            "datasets/raw/WTI/LIGHT.CMDUSD_Candlestick_1_M_BID_01.07.2024-01.11.2024.csv"
        ]
    }

    @property
    def files(self) -> List[str]:
        return self.value["files"]
    
    @property
    def name(self) -> str:
        return self.value["name"]

class DataHandler:
    """Basic data handler with session handling"""
    
    # Standard market sessions for all assets
    SESSIONS = {
        'asian': (time(0, 0), time(8, 0)),
        'london': (time(8, 0), time(16, 0)),
        'us': (time(13, 0), time(21, 0))
    }
    
    def process_asset_data(self, asset: Asset) -> None:
        """Process all data files for a given asset"""
        logger.info(f"Processing {asset.name} data...")
        
        # Create processed directory
        processed_dir = Path(f"datasets/processed/{asset.name}")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and combine all data files
        dfs = []
        for file_path in asset.files:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue
                
            logger.info(f"Reading file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(
                df['Gmt time'],
                format='%d.%m.%Y %H:%M:%S.%f',
                utc=True
            )
            
            # Standardize column names
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            dfs.append(df)
        
        if not dfs:
            logger.error(f"No valid data files found for {asset.name}")
            return
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates and sort
        combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
        combined_df.sort_values('timestamp', inplace=True)
        
        # Set timestamp as index
        combined_df.set_index('timestamp', inplace=True)
        
        # Process and validate data
        self.processed_data = self._process_data(combined_df)
        
        # Save processed data
        output_path = processed_dir / "combined_data.csv"
        self.processed_data.to_csv(output_path)
        logger.info(f"Saved processed data to {output_path}")
        
        # Print basic statistics
        start_date = self.processed_data.index.min().strftime('%Y-%m-%d')
        end_date = self.processed_data.index.max().strftime('%Y-%m-%d')
        logger.info(f"Data range: {start_date} to {end_date}")
        logger.info(f"Total records: {len(self.processed_data)}")
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate data with basic session marking"""
        # Make a copy to avoid modifying the original
        processed = df.copy()
        
        # Add date and time columns
        processed['date'] = processed.index.date
        processed['time'] = processed.index.time
        
        # Add session markers
        for session_name, (start, end) in self.SESSIONS.items():
            if start < end:  # Normal session
                session_mask = (processed['time'] >= start) & (processed['time'] < end)
            else:  # Session crosses midnight
                session_mask = (processed['time'] >= start) | (processed['time'] < end)
            
            processed[f'{session_name}_session'] = session_mask
        
        return processed

def process_selected_assets(assets_to_process: List[str]) -> None:
    """Process only selected assets"""
    handler = DataHandler()
    
    for asset_name in assets_to_process:
        try:
            asset = Asset[asset_name]
            handler.process_asset_data(asset)
        except KeyError:
            logger.error(f"Invalid asset name: {asset_name}")
        except Exception as e:
            logger.error(f"Error processing {asset_name}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Process only SPY data
    process_selected_assets(["WTI"])
    
    # For all assets:
    # process_selected_assets([asset.name for asset in Asset])