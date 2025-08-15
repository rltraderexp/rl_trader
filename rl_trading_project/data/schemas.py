from pydantic import BaseModel, validator
from typing import Optional
from datetime import datetime

class OHLCVSchema(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    trade_count: Optional[int] = None
    exchange: Optional[str] = None
    asset: Optional[str] = None

    @validator('high')
    def high_ge_open_low(cls, v, values):
        o = values.get('open')
        l = values.get('low')
        if o is not None and v < o:
            raise ValueError('high < open')
        if l is not None and v < l:
            raise ValueError('high < low')
        return v