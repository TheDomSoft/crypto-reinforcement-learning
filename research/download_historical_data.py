import ccxt, pandas as pd
ex = ccxt.binance()
ohlcv = ex.fetch_ohlcv('ETH/USDT', timeframe='1m', limit=1000)
df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
df["time"] = pd.to_datetime(df["time"], unit="ms")
print(df.tail())
