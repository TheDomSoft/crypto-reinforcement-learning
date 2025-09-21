from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import time
import asyncio
from typing import Dict, Any, Optional
import uvicorn

app = FastAPI(title="Crypto Order Book API", description="API for fetching and processing order book data")
 
# Configuration for binning and history
CONFIG = {
    "symbol": "ETHUSDT",
    "price_bins_count": 80,   # more vertical resolution
    "history": 150,           # more time steps
    "window_pct": 0.05,       # +/-5% price window around mid price
    "bins": None              # will be initialized on first fetch
}

def make_bins(center: float, pct: float, count: int) -> np.ndarray:
    """Create fixed price bins around a center price with +/- pct window."""
    span = center * pct
    return np.linspace(center - span, center + span, count)

# Global variables for data storage
orderbook_data = {
    "frames": [],
    "current_prices": None,
    "current_timestamp": None,
    "timestamps": [],
    "price_series": []
}

async def fetch_orderbook_data(limit: int = 200, source: str = "bybit"):
    """Fetch order book data from Bybit API"""
    try:
        if source.lower() == "binance":
            # Binance USDT-margined futures depth
            url = "https://fapi.binance.com/fapi/v1/depth"
            params = {"symbol": CONFIG["symbol"], "limit": min(int(limit), 1000)}
            r = requests.get(url, params=params).json()
            if "bids" not in r or "asks" not in r:
                raise HTTPException(status_code=400, detail="Failed to fetch Binance depth")
            bids = np.array(r["bids"], dtype=float)
            asks = np.array(r["asks"], dtype=float)
        else:
            # Bybit linear perpetual orderbook
            url = "https://api.bybit.com/v5/market/orderbook"
            params = {"category": "linear", "symbol": CONFIG["symbol"], "limit": limit}
            r = requests.get(url, params=params).json()
            if r.get("retCode") != 0:
                raise HTTPException(status_code=400, detail="Failed to fetch order book data")
            bids = np.array(r["result"]["b"], dtype=float)  # [["price","size"],...]
            asks = np.array(r["result"]["a"], dtype=float)

        # Current mid price from best bid/ask if available
        current_price = None
        if bids.size > 0 and asks.size > 0:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            current_price = (best_bid + best_ask) / 2.0

        return bids, asks, current_price
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Crypto Order Book API", "status": "running"}

@app.get("/orderbook")
async def get_orderbook(
    bins_count: Optional[int] = None,
    window_pct: Optional[float] = None,
    history: Optional[int] = None,
    limit: Optional[int] = 500,
    transform: Optional[str] = "log",  # 'log' or 'linear'
    mode: Optional[str] = "static",    # 'static' or 'follow'
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    center: Optional[float] = None,
    source: Optional[str] = "bybit",
    norm: Optional[str] = "none",       # 'none' or 'frame'
) -> Dict[str, Any]:
    """Get current order book data with heatmap"""
    try:
        # Apply any runtime overrides
        if bins_count and bins_count != CONFIG["price_bins_count"]:
            CONFIG["price_bins_count"] = int(bins_count)
            CONFIG["bins"] = None  # force re-init
            orderbook_data["frames"].clear()
            orderbook_data["timestamps"].clear()
            orderbook_data["price_series"].clear()
        if window_pct and window_pct != CONFIG["window_pct"]:
            CONFIG["window_pct"] = float(window_pct)
            CONFIG["bins"] = None
            orderbook_data["frames"].clear()
            orderbook_data["timestamps"].clear()
            orderbook_data["price_series"].clear()
        if history and history != CONFIG["history"]:
            CONFIG["history"] = int(history)

        fetch_limit = int(limit) if limit else 200

        bids, asks, current_price = await fetch_orderbook_data(limit=fetch_limit, source=(source or "bybit"))

        # Combine bids + asks into histogram by price level
        orderbook = np.vstack([bids, asks])
        prices = orderbook[:, 0].astype(float)
        sizes = orderbook[:, 1].astype(float)

        # Initialize or update bins
        if CONFIG["bins"] is None:
            if y_min is not None and y_max is not None and y_max > y_min:
                CONFIG["bins"] = np.linspace(float(y_min), float(y_max), CONFIG["price_bins_count"])
            else:
                # initialize around provided center or current price
                c0 = float(center) if center is not None else float(current_price if current_price is not None else prices.mean())
                CONFIG["bins"] = make_bins(c0, CONFIG["window_pct"], CONFIG["price_bins_count"])
        else:
            # Only recenter in 'follow' mode
            if mode == "follow":
                bmin, bmax = CONFIG["bins"][0], CONFIG["bins"][-1]
                width = bmax - bmin
                if current_price is not None and (current_price < bmin + 0.1 * width or current_price > bmax - 0.1 * width):
                    CONFIG["bins"] = make_bins(float(current_price), CONFIG["window_pct"], CONFIG["price_bins_count"])
                    orderbook_data["frames"].clear()
                    orderbook_data["timestamps"].clear()
                    orderbook_data["price_series"].clear()

        # Histogram using fixed bins to keep rows aligned over time
        bins = CONFIG["bins"]
        hist, _ = np.histogram(prices, bins=bins, weights=sizes)
        # Gentle vertical smoothing to reduce speckle
        kernel = np.array([1, 4, 6, 4, 1], dtype=float) / 16.0
        hist = np.convolve(hist, kernel, mode="same")

        # Add to history
        # Exponential moving average smoothing to reduce flicker
        if orderbook_data["frames"]:
            prev = orderbook_data["frames"][-1]
            smoothed = 0.6 * hist + 0.4 * prev
        else:
            smoothed = hist
        # Optional log transform to bring out weaker levels
        if transform == "log":
            smoothed = np.log1p(np.maximum(smoothed, 0))
        # Optional per-frame normalization to highlight relative levels across price
        if norm == "frame":
            denom = np.quantile(smoothed, 0.98) if np.any(np.isfinite(smoothed)) else None
            if denom is None or denom <= 0:
                denom = smoothed.max() if np.isfinite(smoothed.max()) and smoothed.max() > 0 else 1.0
            smoothed = smoothed / float(denom)
        orderbook_data["frames"].append(smoothed)
        if len(orderbook_data["frames"]) > CONFIG["history"]:
            orderbook_data["frames"].pop(0)

        # Append time and price series
        now_ms = int(time.time() * 1000)
        orderbook_data["timestamps"].append(now_ms)
        orderbook_data["price_series"].append(float(current_price) if current_price is not None else None)
        # Enforce history length on time and price series as well
        if len(orderbook_data["timestamps"]) > CONFIG["history"]:
            orderbook_data["timestamps"].pop(0)
        if len(orderbook_data["price_series"]) > CONFIG["history"]:
            orderbook_data["price_series"].pop(0)

        orderbook_data["current_prices"] = prices
        orderbook_data["current_timestamp"] = now_ms

        # Convert to heatmap
        heatmap_data = np.array(orderbook_data["frames"]).T

        # Use bin centers for y axis
        price_levels = ((bins[:-1] + bins[1:]) / 2.0)

        return {
            "current_price": current_price,
            "price_range": {"min": float(bins[0]), "max": float(bins[-1])},
            "heatmap_data": heatmap_data.tolist(),
            "time_steps": len(orderbook_data["frames"]),
            "price_bins": price_levels.tolist(),
            "total_volume": float(sizes.sum()),
            "timestamp": orderbook_data["current_timestamp"],
            "timestamps": orderbook_data["timestamps"],
            "price_series": orderbook_data["price_series"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing order book: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "frames_count": len(orderbook_data["frames"])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
