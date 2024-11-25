"""
Data Handling:
    Implemented DukasCopy data loading
    Added data cleaning and validation
    Created market session separation (Asian/London/US)
    Handle timezone conversions and data formatting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union
import pytz
from datetime import datetime, time
from typing import List, Union


class DukasCopyDataHandler:
    """
    Handles data loading and processing for DukasCopy historical price data.
    Focuses on market session separation and data validation.
    """
    
    # Market session times in UTC
    SESSIONS = {
        'asian': (time(0, 0), time(8, 0)),
        'london': (time(8, 0), time(16, 0)),
        'us': (time(13, 0), time(21, 0))
    }
    
    def __init__(self):
        """Initialize the data handler"""
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.session_data: Dict[str, pd.DataFrame] = {}
        
    def load_data(self, file_paths: List[Union[str, Path]]) -> pd.DataFrame:
    
        all_data = []  # To store data from all files
    
        for file_path in file_paths:
            # Read the CSV file
            raw_data = pd.read_csv(file_path, parse_dates=[0])
        
            # Rename columns to standard format
            raw_data.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
            # Convert timestamp to UTC
            raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'], format='%d.%m.%Y %H:%M:%S.%f', utc=True)
        
         # Set timestamp as index
            raw_data.set_index('timestamp', inplace=True)
        
            # Append to the list
            all_data.append(raw_data)
    
        # Concatenate all the dataframes into one
        self.raw_data = pd.concat(all_data).sort_index()

        # Perform basic data validation
        self._validate_data()
    
        # Process the data
        self._process_data()
    
        return self.processed_data

    
    def _validate_data(self) -> None:
        """
        Perform basic data validation checks
        """
        # Check for missing values
        if self.raw_data.isnull().any().any():
            print("Warning: Dataset contains missing values")
            self.raw_data.fillna(method='ffill', inplace=True)
        
        # Validate OHLC relationships
        invalid_rows = (
            (self.raw_data['high'] < self.raw_data['low']) |
            (self.raw_data['high'] < self.raw_data['open']) |
            (self.raw_data['high'] < self.raw_data['close']) |
            (self.raw_data['low'] > self.raw_data['open']) |
            (self.raw_data['low'] > self.raw_data['close'])
        )
        
        if invalid_rows.any():
            print(f"Warning: Found {invalid_rows.sum()} rows with invalid OHLC relationships")
        
        # Check for negative volumes
        if (self.raw_data['volume'] < 0).any():
            print("Warning: Negative volumes found in data")
    
    def _process_data(self) -> None:
        """
        Process the raw data into the final format
        """
        self.processed_data = self.raw_data.copy()
        
        # Add helper columns
        self.processed_data['date'] = self.processed_data.index.date
        self.processed_data['time'] = self.processed_data.index.time
        
        # Create session markers
        self._create_market_sessions()
        
        # Calculate daily values
        self._calculate_daily_stats()
        
        # Drop rows with NaN prev_close
        self.processed_data.dropna(subset=['prev_close'], inplace=True)
        
        # Update session data after dropping NaN values
        self._create_market_sessions()
    
    def _create_market_sessions(self) -> None:
        """
        Create market session markers for Asian, London, and US sessions
        """
        for session_name, (session_start, session_end) in self.SESSIONS.items():
            # Create session mask
            if session_start < session_end:
                session_mask = (self.processed_data['time'] >= session_start) & \
                             (self.processed_data['time'] < session_end)
            else:  # Handle sessions that cross midnight
                session_mask = (self.processed_data['time'] >= session_start) | \
                             (self.processed_data['time'] < session_end)
            
            # Add session column
            self.processed_data[f'{session_name}_session'] = session_mask
            
            # Create session-specific DataFrame
            self.session_data[session_name] = self.processed_data[session_mask].copy()
    
    def _calculate_daily_stats(self) -> None:
        """
        Calculate daily statistics including previous day's close
        """
        # Calculate daily OHLCV
        daily_data = self.processed_data.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Calculate previous day's close
        daily_data['prev_close'] = daily_data['close'].shift(1)
        
        # Merge back to processed data
        self.processed_data = self.processed_data.join(
            daily_data[['prev_close']], 
            on='date'
        )
    
    def get_session_data(self, session_name: str) -> pd.DataFrame:
        """
        Get data for a specific market session
        
        Parameters:
        -----------
        session_name : str
            Name of the session ('asian', 'london', or 'us')
            
        Returns:
        --------
        pd.DataFrame
            Data for the specified session
        """
        if session_name not in self.SESSIONS:
            raise ValueError(f"Invalid session name. Must be one of {list(self.SESSIONS.keys())}")
        
        return self.session_data[session_name]
    
    def save_processed_data(self, output_path: Union[str, Path]) -> None:
        """
        Save processed data to CSV
        
        Parameters:
        -----------
        output_path : str or Path
            Path where to save the processed CSV file
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run load_data first.")
        
        self.processed_data.to_csv(output_path)

    def get_timestamp_range(self) -> tuple:
        """
        Get the date range of the data
        
        Returns:
        --------
        tuple
            Start and end timestamps of the data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        return (
            self.processed_data.index.min(),
            self.processed_data.index.max()
        )
    
    def get_data_quality_metrics(self) -> dict:
        """
        Get data quality metrics
        
        Returns:
        --------
        dict
            Dictionary containing various data quality metrics
        """
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        metrics = {
            'total_rows': len(self.processed_data),
            'missing_values': self.processed_data.isnull().sum().to_dict(),
            'unique_dates': self.processed_data['date'].nunique(),
            'avg_daily_volume': self.processed_data.groupby('date')['volume'].sum().mean(),
            'session_distribution': {
                session: self.session_data[session].shape[0] 
                for session in self.SESSIONS.keys()
            }
        }
        
        return metrics