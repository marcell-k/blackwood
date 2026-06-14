
import numpy as np
import pandas as pd
from src.indicators.core import calculate_atr
from src.indicators.cycle import (
    adaptive_atr_ehlers,
    ehler_dominant_cycle,
    get_typical_price,
)


class RegimeFeatureEngineer:
    def __init__(self, df,  timeframes: list):
        self.timeframes = timeframes
        self.data = df
    
        self._resampled_cache: dict[str, pd.DataFrame] = {}
        self._features_cache: dict[str, pd.DataFrame] = {}    

    def volatility_features(self, df):
        """
        Volatility features:
            - atr
            - realized vol
            - vol zscore
            - vol of vol
            - vol percentile
        """
        features = pd.DataFrame(index=df.index)
        close = df['Close']
        log_returns = np.log(df['Close']).diff()

        features['atr_5'] = calculate_atr(df, atr_length=5) / close
        features['atr_14'] = calculate_atr(df, atr_length=14) / close
        features['atr_21'] = calculate_atr(df, atr_length=21) / close
        features['atr_40'] = calculate_atr(df, atr_length=40) / close

        log_returns = np.log(close / close.shift(1))
        realized_vol = log_returns.rolling(20, min_periods=10).std() * np.sqrt(252)
        features['realized_vol_20'] = realized_vol

        features['vol_of_vol_20'] = log_returns.rolling(window=20, min_periods=10).std()
        features['vol_of_vol_40'] = log_returns.rolling(window=40, min_periods=20).std()
        
        return features.fillna(0)

    def stationary_features(self, df):
        features = pd.DataFrame(index=df.index)
        close = df['Close']

        log_returns = np.log(close / close.shift(1))
        
        # TODO: add rolling hurst exponent
        
        return features
    
    def correlation_features(self, df):
        features = pd.DataFrame(index=df.index)

        # TODO acf, pacf
        return features

    def technical_features(self, df):
        features = pd.DataFrame(index=df.index)
                
        dominant_period = ehler_dominant_cycle(get_typical_price(df), 0.5)
        features['adp_atr'] = adaptive_atr_ehlers(df, dominant_period) / df['Close']
        return features

    def get_all_features(self, df):
        feature_list = []
        feature_list.append(self.volatility_features(df))
        # feature_list.append(self.stationary_features(df))
        # feature_list.append(self.correlation_features(df))
        feature_list.append(self.technical_features(df))
        
        return pd.concat(feature_list, axis=1)
    

    def run(self, ):
        aligned_features = []
        for tf in self.timeframes:
            df_resampled = self._resample_ohlcv(tf)
            features = self.get_all_features(df_resampled)
            features_shifted = features.shift(1)
            features_aligned = features_shifted.reindex(self.data.index, method='ffill')
            features_aligned = features_aligned.add_suffix(f'_{tf}')
            aligned_features.append(features_aligned)
        
        result = pd.concat(aligned_features, axis=1)
        result = result.fillna(0.0)
        return result

    # Helper
    def _resample_ohlcv(self, timeframe: str) -> pd.DataFrame:
        if timeframe in self._resampled_cache:
            return self._resampled_cache[timeframe]
        
        resampled = self.data.resample(timeframe).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        self._resampled_cache[timeframe] = resampled
        return resampled
