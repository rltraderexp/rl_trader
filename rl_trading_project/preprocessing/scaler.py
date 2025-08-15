from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from typing import List, Optional
import warnings

class IdentityScaler:
    """A scaler that does nothing, for API compatibility."""
    def fit(self, data):
        return self
    def transform(self, data):
        return data
    def fit_transform(self, data):
        return data
    def inverse_transform(self, data):
        return data

class ScalerWrapper:
    """Wrapper around sklearn scalers to fit on DataFrame numeric columns and transform DataFrames."""
    def __init__(self, method: str = 'standard', feature_columns: Optional[List[str]] = None):
        self.method = method
        self.feature_columns = feature_columns
        self.scaler = None
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'identity':
            self.scaler = IdentityScaler()
        else:
            raise ValueError('Unknown scaler method: ' + str(method))

    def fit(self, df: pd.DataFrame, start: Optional[int] = None, end: Optional[int] = None):
        if self.feature_columns is None:
            # Auto-detect numeric columns, excluding common non-feature columns
            exclude = {'timestamp', 'expiry', 'open', 'high', 'low', 'close', 'volume'}
            self.feature_columns = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        
        if not self.feature_columns:
            self.scaler = IdentityScaler() # No features to scale
            return self
            
        seg = df.iloc[start:end] if start is not None or end is not None else df
        X = seg[self.feature_columns].astype(float).fillna(0.0).values
        if len(X) == 0:
            warnings.warn('No rows to fit scaler on, scaler will not be fitted.')
            return self
            
        self.scaler.fit(X)
        return self

    def transform_df(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        df2 = df if inplace else df.copy()
        if not self.feature_columns or isinstance(self.scaler, IdentityScaler):
            return df2
        
        X = df2[self.feature_columns].astype(float).fillna(0.0).values
        Xt = self.scaler.transform(X)
        df2[self.feature_columns] = Xt
        return df2

    def fit_transform_df(self, df: pd.DataFrame, start: Optional[int] = None, end: Optional[int] = None, inplace: bool = False):
        self.fit(df, start, end)
        return self.transform_df(df, inplace=inplace)