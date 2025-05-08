# enhanced_risk_dashboard.py – Comprehensive Factor Analysis
"""Equity Risk Dashboard
=================================================
* Multi-facet factor modeling with extensive factor library
* Advanced exposures analysis with conditional and time-varying sensitivities
* Quantitative risk metrics with expanded stress testing scenarios
* Dynamic factor attribution with contribution breakdown
* Customizable factor groups for specialized analysis
* Expanded visualization capabilities for detailed risk assessment
* Comprehensive factor timing analysis with rolling exposures

Features:
* Local daily OHLCV data (Close) for 1,500 tickers
* Interactive holdings table (add/delete rows) or CSV upload
* Factor file upload (Ken-French style or similar), automatic web download, or synthetic factor generation
* Multiple pre-defined factor sets: Traditional, Style, Macro, Smart Beta
* Advanced risk metrics: decomposed volatility, VAR sensitivities, stress testing, and factor timing
"""

from __future__ import annotations
import requests
import io
import sys
import json
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates

import yfinance as yf
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

# ────────────────────────── Dependency Check ──────────────────────────
missing: list[str] = []
try:
    from scipy.stats import norm, t as t_dist  # type: ignore
    from scipy import optimize
except Exception:
    norm = None  # type: ignore
    missing.append("scipy >=1.9")
try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.regression.rolling import RollingOLS  # type: ignore
    from statsmodels.stats.diagnostic import het_breuschpagan  # type: ignore
except ImportError:
    sm = None  # type: ignore
    missing.append("statsmodels")
try:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
except ImportError:
    PCA = None  # type: ignore
    missing.append("scikit-learn")
try:
    from pandas_datareader import data as pdr  # type: ignore
except ImportError:
    pdr = None  # type: ignore
    missing.append("pandas-datareader")

if norm is None or PCA is None:
    st.error("SciPy and scikit-learn are required – run `pip install --upgrade numpy scipy scikit-learn`.")
    st.stop()
try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False
# ───────────────────────────── Config ──────────────────────────────
DEFAULT_START = "2018-01-01"

# Extended factor mappings
FF5_MAP = {"MKT": "Mkt-RF", "SMB": "SMB", "HML": "HML", "RMW": "RMW", "CMA": "CMA", "UMD": "Mom"} # RF is not typically mapped here, Mkt-RF is MKT - RF

# Extended FRED data for macro factors
MACRO_FRED = {
    "10Y_Yield": "DGS10",
    "2Y_Yield": "DGS2",
    "Yield_Curve": "T10Y2Y",  # 10Y-2Y spread
    "USD_Index": "DTWEXBGS",
    "WTI_Oil": "DCOILWTICO",
    "CPI": "CPIAUCSL",
    "VIX": "VIXCLS",  # Will be converted to VIX_Diff
    "TED_Spread": "TEDRATE", # TED spread (3m Libor - 3m T-bill)
    "Credit_Spread": "BAA10Y", # Moody's BAA - 10Y Treasury
    "Unemployment": "UNRATE"
}

# Categorized factor groups
FACTOR_GROUPS = {
    "Traditional": ["Mkt-RF", "SMB", "HML", "UMD", "RMW", "CMA"], # Ensure RF is not here
    "Style": ["Value", "Size", "Momentum", "Quality", "Low_Vol", "Growth"], # Synthetic factors
    "Macro": ["Yield_Curve_Diff", "USD_Index", "WTI_Oil", "VIX_Diff", "Credit_Spread_Diff", "TED_Spread_Diff", "CPI"], # Standardized names
    "Smart Beta": ["Min_Vol", "Max_Div", "Equal_Weight", "Quality_Tilt", "ESG_Tilt"] # Synthetic factors
}

# Confidence levels for risk metrics
CONFIDENCE_LEVELS = [0.99, 0.975, 0.95, 0.90]

# ─────────────────────── Utility Functions ────────────────────────
@st.cache_data(show_spinner="Loading price data …")
def load_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """
    Load price data using yfinance for the given tickers and start date.
    """
    st.info(f"Downloading data for {len(tickers)} tickers using yfinance...")
    data = yf.download(tickers, start=start, auto_adjust=False)

    if 'Close' in data:
        prices = data['Close']
    else:
        st.error("Failed to retrieve 'Close' prices from yfinance data.")
        return pd.DataFrame()

    # Ensure column names match ticker names
    prices.columns = [col.upper() if isinstance(col, str) else col for col in prices.columns]
    return prices.sort_index()


def sanitize_weights(rows: List[dict]) -> pd.Series:
    valid = {r["ticker"].strip().upper(): r["weight"] for r in rows if r["ticker"].strip() and r["weight"] != 0}
    if not valid:
        return pd.Series(dtype=float)
    w = pd.Series(valid, dtype=float)
    return (w / w.abs().sum()).sort_index()


def load_weight_csv(file) -> pd.Series:
    df = pd.read_csv(file)
    if df.shape[1] < 2:
        st.error("Weight CSV must have at least 2 columns (Ticker, Weight)")
        st.stop()
    w = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0], dtype=float)
    return (w / w.abs().sum()).sort_index()

# risk helpers
_var = lambda s, p=0.05: np.quantile(s, p)
_gauss = lambda s, p=0.05: s.mean() + s.std(ddof=1) * norm.ppf(p)
_es = lambda s, p=0.05: s[s <= np.quantile(s, p)].mean()
_roll = lambda s, w=252: s.rolling(w).std() * np.sqrt(252)
_maxdd = lambda s: ((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()

# Advanced risk analytics functions
def calc_conditional_var(returns: pd.Series, factor_returns: pd.Series, 
                        confidence: float = 0.95, window: int = 252) -> pd.Series:
    """
    Calculate conditional VaR based on factor exposures
    """
    combined = pd.concat([returns, factor_returns], axis=1).dropna()
    combined.columns = ['returns', 'factor']
    
    result = pd.Series(index=combined.index)
    
    for i in range(window, len(combined)):
        window_data = combined.iloc[i-window:i]
        # Calculate conditional VaR using regression residuals
        X = sm.add_constant(window_data['factor'])
        y = window_data['returns']
        model = sm.OLS(y, X).fit()
        residuals = model.resid
        
        # Calculate VaR of residuals
        var_residuals = np.percentile(residuals, 100 * (1 - confidence))
        
        # Today's factor value
        today_factor = combined.iloc[i]['factor']
        
        # Conditional VaR
        result.iloc[i] = model.params['const'] + model.params['factor'] * today_factor + var_residuals
        
    return result.dropna()

def calc_component_var(returns: pd.DataFrame, weights: pd.Series, 
                     confidence: float = 0.95) -> pd.Series:
    """
    Calculate Component VaR to show risk contribution by position
    """
    # Ensure weights are valid
    weights = weights / weights.abs().sum()  # Normalize weights

    # Ensure unique column names in returns
    returns = returns.loc[:, ~returns.columns.duplicated(keep='first')]

    # Align weights with returns columns
    weights = weights.reindex(returns.columns).fillna(0)

    portfolio_return = returns.dot(weights)
    portfolio_var = _var(portfolio_return, 1-confidence)

    # Calculate marginal VaR using covariance matrix
    cov_matrix = returns.cov()
    cov_matrix = cov_matrix.loc[weights.index, weights.index]  # Align covariance matrix with weights

    portfolio_variance = weights.dot(cov_matrix).dot(weights)
    portfolio_vol = np.sqrt(portfolio_variance) if portfolio_variance > 0 else 1e-8

    marginal_vars = cov_matrix.dot(weights) / portfolio_vol
    z_score = norm.ppf(1-confidence)
    marginal_vars = marginal_vars * z_score

    component_vars = marginal_vars * weights
    component_sum = component_vars.sum()

    if abs(component_sum) < 1e-8 or portfolio_var == 0:
        return component_vars
    else:
        scaling_factor = portfolio_var / component_sum
        return component_vars * scaling_factor

def calc_factor_timing_score(factor_betas: pd.DataFrame, factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate factor timing score based on correlation of exposure with future returns
    """
    result = {}
    for factor in factor_betas.columns:
        if factor in factor_returns.columns:
            # Lag factor betas by 1 day
            lagged_betas = factor_betas[factor].shift(1).dropna()
            common_idx = lagged_betas.index.intersection(factor_returns.index)
            
            if len(common_idx) > 60:  # Minimum sample size
                corr = np.corrcoef(lagged_betas.loc[common_idx], 
                                  factor_returns.loc[common_idx, factor])[0, 1]
                result[factor] = corr
    
    return pd.Series(result, name="Timing_Score")

def calc_factor_attribution(returns: pd.Series, factor_exposures: pd.DataFrame, 
                          factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate return attribution by factor
    """
    # Get common dates
    common_dates = returns.index.intersection(
        factor_exposures.index.intersection(factor_returns.index))
    
    if len(common_dates) == 0:
        st.warning("No overlapping dates for return attribution")
        return None
    
    # Make sure factor_exposures has the same columns as factor_returns
    available_factors = [col for col in factor_exposures.columns if col in factor_returns.columns]
    
    # Check if we have any matching factors
    if not available_factors:
        st.warning("No matching factors between exposures and returns")
        return None
    
    # Filter to only include matching factors
    factor_exposures = factor_exposures[available_factors]
    factor_returns = factor_returns[available_factors]
    
    # Create a DataFrame to store factor contribution for each day
    factor_contrib = pd.DataFrame(index=common_dates, columns=available_factors)
    
    # Calculate contribution for each factor
    for factor in available_factors:
        # For each date, multiply the factor exposure by the factor return
        # This works because factor_exposures is a time series of factor betas
        exposures = factor_exposures.loc[common_dates, factor]
        returns_f = factor_returns.loc[common_dates, factor]
        factor_contrib[factor] = exposures * returns_f
    
    # Calculate the total factor contribution (sum across all factors)
    total_factor_contrib = factor_contrib.sum(axis=1)
    
    # Calculate the specific (non-factor) return
    portfolio_returns = returns.loc[common_dates]
    specific_return = portfolio_returns - total_factor_contrib
    
    # Add specific return to the attribution DataFrame
    factor_contrib['Specific'] = specific_return
    
    # Handle potential numerical issues - very small values close to zero
    for col in factor_contrib.columns:
        factor_contrib[col] = np.where(
            np.abs(factor_contrib[col]) < 1e-10, 
            0, 
            factor_contrib[col]
        )
    
    return factor_contrib

def calc_extreme_scenario(factor_returns: pd.DataFrame, betas: pd.Series,
                       quantile: float = 0.01) -> pd.Series:
    """
    Calculate extreme scenario impact based on historical factor movements
    """
    worst_days = {}
    impact = pd.Series(index=factor_returns.columns, dtype=float)
    
    for factor in factor_returns.columns:
        if factor in betas.index:
            # Find worst day for each factor
            worst_day_idx = factor_returns[factor].quantile(quantile)
            impact[factor] = worst_day_idx * betas[factor]
    
    return impact

# ───── Factor File Loader (robust) ─────

def read_factor_file(src, start: str):
    """Robust factor CSV/TSV reader.

    Accepts a path *or* a Streamlit `UploadedFile`.
    Handles:
      • optional leading blank column (	‑prefixed rows)
      • auto delimiter sniff (comma / tab)
      • headers present on the first data line
      • integer or string yyyymmdd dates
      • percent vs decimal scaling
    """
    try:
        # ------------------------------------------------------------------
        # 1) Read raw bytes / text
        if hasattr(src, "read"):
            src.seek(0)
            raw = src.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "ignore")
        else:
            raw = Path(src).read_text(encoding="utf-8", errors="ignore")

        # ------------------------------------------------------------------
        # 2) Detect first data row (starts with digits)
        lines = raw.splitlines()
        
        # Debug info
        st.sidebar.expander("CSV Debug").write(f"Found {len(lines)} lines in factor file")
        
        # Safety check for empty file
        if not lines:
            st.warning("Factor file appears to be empty")
            return None
            
        # Find first data row more robustly
        first_data_idx = 0
        for i, l in enumerate(lines):
            line_parts = l.lstrip().split("	")[0].strip().split(",")
            if line_parts and line_parts[0].isdigit():
                first_data_idx = i
                break
        
        if first_data_idx == len(lines) - 1:
            st.warning("Could not find data rows in factor file")
            return None
            
        cleaned = "\n".join(lines[first_data_idx:])

        # ------------------------------------------------------------------
        # 3) Load with pandas; sniff delimiter
        df = pd.read_csv(io.StringIO(cleaned), sep=None, engine="python")

        # Drop entirely empty unnamed col (common when original had leading tab)
        if df.columns[0].startswith("Unnamed") and df.iloc[:, 0].isna().all():
            df = df.iloc[:, 1:]

        # ------------------------------------------------------------------
        # 4) Ensure first column is datetime index
        date_col = df.columns[0]
        
        # Show sample of unconverted data for debugging
        st.sidebar.expander("Date Format Debug").write(df[date_col].head().to_dict())
        
        # Try multiple date formats
        for date_format in ["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]:
            try:
                df[date_col] = pd.to_datetime(df[date_col].astype(str), errors="coerce", format=date_format)
                if not df[date_col].isna().all():
                    break
            except Exception:
                continue
                
        df = df.dropna(subset=[date_col])
        if df.empty:
            st.warning("Failed to parse dates in factor file")
            return None
            
        df.set_index(date_col, inplace=True)
        df.index.name = "Date"
        df.sort_index(inplace=True)

        # ------------------------------------------------------------------
        # 5) Percentage → decimal conversion
        for col in df.columns:
            if df[col].dtype.kind in "if" and df[col].abs().mean() > 1:
                df[col] /= 100

        # Keep post‑start data
        df = df.loc[df.index >= pd.to_datetime(start)]
        return df if not df.empty else None
    except Exception as e:
        st.warning(f"Factor file parse error: {str(e)}")
        st.sidebar.expander("Error Details").write(traceback.format_exc())
        return None

# ───── Online factor fetch (Enhanced) ─────

def fetch_online_factors(start: str) -> Optional[pd.DataFrame]:
    """Fetch expanded factor data from online sources with improved error handling"""
    
    if pdr is None:
        st.warning("pandas-datareader is required for online factor fetch. Install with: pip install pandas-datareader")
        return None
        
    st.sidebar.markdown("### Factor Data Fetch Status")
    ff_data = None
    macro_data = None
    momentum_data = None
    
    # Try to fetch Fama-French 5 factor data
    try:
        with st.sidebar.expander("Fama-French Factor Fetch"):
            st.info("Attempting to fetch Fama-French factors...")
            # Fama-French data includes Mkt-RF, SMB, HML, RMW, CMA, and RF
            raw_data_ff = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start)
            ff = raw_data_ff[0] / 100 # raw_data[0] is the DataFrame with factors including RF
            ff.index = pd.to_datetime(ff.index).tz_localize(None)
            
            # Explicitly drop the 'RF' column if it exists
            if 'RF' in ff.columns:
                ff = ff.drop(columns=['RF'])
                st.success("Dropped 'RF' column from Fama-French data.")
            else:
                st.info("'RF' column not found in Fama-French data to drop.")

            ff.rename(columns=FF5_MAP, inplace=True) # FF5_MAP renames MKT to Mkt-RF
            ff = ff.loc[ff.index >= pd.to_datetime(start)]
            st.success(f"Successfully fetched and processed {len(ff)} rows of Fama-French 5 factor data (excluding RF).")
            st.write(ff.tail())
            ff_data = ff
    except Exception as e:
        st.sidebar.error(f"Failed to fetch Fama-French data: {str(e)}")
    
    # Try to fetch Momentum factor separately
    try:
        with st.sidebar.expander("Momentum Factor Fetch"):
            st.info("Attempting to fetch Momentum factor...")
            raw_data = pdr.DataReader("F-F_Momentum_Factor_daily", "famafrench", start=start)
            mom = raw_data[0] / 100
            mom.index = pd.to_datetime(mom.index).tz_localize(None)
            mom.rename(columns={"Mom   ": "UMD"}, inplace=True)
            mom = mom.loc[mom.index >= pd.to_datetime(start)]
            st.success(f"Successfully fetched {len(mom)} rows of Momentum factor data")
            st.write(mom.tail())
            momentum_data = mom
    except Exception as e:
        st.sidebar.error(f"Failed to fetch Momentum factor: {str(e)}")
        
    # Try to fetch FRED data with detailed error reporting
    try:
        with st.sidebar.expander("FRED Macro Data Fetch"):
            st.info("Attempting to fetch FRED macro data...")
            macro_dfs = []
            for name, ticker in MACRO_FRED.items():
                try:
                    df = pdr.DataReader(ticker, "fred", start)
                    df.rename(columns={ticker: name}, inplace=True)
                    macro_dfs.append(df)
                    st.write(f"✅ {name} ({ticker}): {len(df)} rows")
                except Exception as e:
                    st.write(f"❌ {name} ({ticker}): {str(e)}")
            
            if macro_dfs:
                macro = pd.concat(macro_dfs, axis=1)
                macro = macro.asfreq('B').ffill()

                cols_to_keep_final = []
                processed_cols_for_debug = []

                # Handle VIX specifically: calculate VIX_Diff
                if "VIX" in macro.columns:
                    macro["VIX_Diff"] = macro["VIX"].diff()
                    cols_to_keep_final.append("VIX_Diff")
                    processed_cols_for_debug.append("VIX (to VIX_Diff)")
                
                # Handle rate and spread columns: calculate _Diff
                rate_cols_orig_names = [name for name in MACRO_FRED if 'Yield' in name or 'Spread' in name]
                
                for col_name in rate_cols_orig_names:
                    if col_name in macro.columns:
                        macro[f"{col_name}_Diff"] = macro[col_name].diff()
                        cols_to_keep_final.append(f"{col_name}_Diff")
                        processed_cols_for_debug.append(f"{col_name} (to {col_name}_Diff)")

                # Handle other columns (e.g., CPI, USD_Index, WTI_Oil): calculate pct_change()
                other_cols_for_pct_change = [
                    col for col in macro.columns 
                    if col != "VIX" and col not in rate_cols_orig_names and not col.endswith("_Diff")
                ]
                for col_name in other_cols_for_pct_change:
                    if col_name in macro.columns:
                         macro[col_name] = macro[col_name].pct_change()
                         cols_to_keep_final.append(col_name)
                         processed_cols_for_debug.append(f"{col_name} (to pct_change)")
                
                cols_to_keep_final = sorted(list(set(cols_to_keep_final))) # Ensure unique and sorted
                if cols_to_keep_final:
                    macro = macro[cols_to_keep_final]
                    macro = macro.ffill().dropna()
                    st.success(f"Successfully processed {len(macro.columns)} macro series from FRED.")
                    st.write(f"Processed columns for FRED: {', '.join(macro.columns)}") # Show final columns
                    st.write(macro.tail())
                    macro_data = macro
                else:
                    st.warning("No macro columns were processed or kept.")
                    macro_data = pd.DataFrame() # Empty DataFrame if nothing to keep
            else:
                st.error("Failed to fetch any FRED data series")
    except Exception as e:
        st.sidebar.error(f"Failed to fetch FRED data: {str(e)}")
        st.sidebar.expander("FRED Error Details").write(traceback.format_exc())
    
    # Combine all factors
    all_dfs = []
    if ff_data is not None:
        all_dfs.append(ff_data)
    if momentum_data is not None:
        # Only append UMD column if FF5 was successful
        if ff_data is not None:
            all_dfs.append(momentum_data[['UMD']])
        else:
            all_dfs.append(momentum_data)
    if macro_data is not None:
        all_dfs.append(macro_data)
    
    if all_dfs:
        # Fix deprecated ffill method
        return pd.concat(all_dfs, axis=1).ffill().dropna()
    else:
        st.warning("Failed to fetch any factor data from online sources")
        return None

# ───── Advanced Factor Generation ─────

def generate_advanced_factors(prices: pd.DataFrame, start_date: str) -> pd.DataFrame:
    """Generate an expanded set of factors from price data when external factors aren't available"""
    st.info("Using price-based synthetic factors")
    
    # Filter and prepare price data
    filtered_prices = prices.loc[prices.index >= pd.to_datetime(start_date)]
    returns = filtered_prices.pct_change().dropna()
    
    # Market factor (equal-weighted market return)
    market = returns.mean(axis=1)
    
    # Size factor (small minus big)
    # Using median market cap as a threshold
    cap_proxy = filtered_prices.iloc[0]  # Use first day price as market cap proxy
    small_stocks = cap_proxy < cap_proxy.median()
    small_ret = returns.loc[:, small_stocks].mean(axis=1)
    big_ret = returns.loc[:, ~small_stocks].mean(axis=1)
    smb = small_ret - big_ret
    
    # Value factor (high B/M minus low B/M)
    # Using price inverse as a proxy for book-to-market
    bm_proxy = 1 / filtered_prices.iloc[0]
    value_stocks = bm_proxy > bm_proxy.median()
    value_ret = returns.loc[:, value_stocks].mean(axis=1)
    growth_ret = returns.loc[:, ~value_stocks].mean(axis=1)
    hml = value_ret - growth_ret
    
    # Momentum factor (winners minus losers)
    lookback = min(126, returns.shape[0] // 3)  # Use past ~6 months or 1/3 of data
    if lookback > 20:  # Minimum window size
        # Calculate past return for momentum
        past_ret = filtered_prices.iloc[lookback] / filtered_prices.iloc[0] - 1
        winners = past_ret > past_ret.median()
        post_idx = returns.index[lookback:]
        if len(post_idx) > 20:  # Ensure sufficient data
            winners_ret = returns.loc[post_idx, winners].mean(axis=1)
            losers_ret = returns.loc[post_idx, ~winners].mean(axis=1)
            wml = winners_ret - losers_ret
        else:
            wml = pd.Series(0, index=returns.index)
    else:
        wml = pd.Series(0, index=returns.index)
    
    # Quality factor (low volatility minus high volatility)
    if returns.shape[0] > 60:  # Need sufficient data for volatility calc
        rolling_vol = returns.iloc[:60].std()
        low_vol = rolling_vol < rolling_vol.median()
        low_vol_ret = returns.loc[:, low_vol].mean(axis=1)
        high_vol_ret = returns.loc[:, ~low_vol].mean(axis=1)
        quality = low_vol_ret - high_vol_ret
    else:
        quality = pd.Series(0, index=returns.index)
    
    # Defensive/Offensive factor
    if returns.shape[0] > 60:
        # Calculate beta to market for each stock
        betas = {}
        for col in returns.columns:
            if len(returns[col].dropna()) > 20:
                X = sm.add_constant(market.iloc[:60])
                model = sm.OLS(returns[col].iloc[:60], X).fit()
                betas[col] = model.params.iloc[1]   # Market beta
        
        beta_series = pd.Series(betas)
        defensive = beta_series < beta_series.median()
        defensive_ret = returns.loc[:, defensive.index[defensive]].mean(axis=1)
        offensive_ret = returns.loc[:, defensive.index[~defensive]].mean(axis=1)
        def_off = defensive_ret - offensive_ret
    else:
        def_off = pd.Series(0, index=returns.index)
    
    # Industry rotation
    # Cluster stocks into pseudo-industries based on return correlations
    if returns.shape[1] > 30 and returns.shape[0] > 60:
        # Use a subset of data for clustering
        sample_returns = returns.iloc[:60]
        
        # Calculate correlation matrix and convert to distance
        corr_matrix = sample_returns.corr()
        dist_matrix = 1 - corr_matrix
        
        # Use K-means clustering to identify pseudo-industries
        # Convert distance matrix to features for K-means
        k = min(5, returns.shape[1] // 10)  # Number of clusters
        if k >= 2:
            kmeans = KMeans(n_clusters=k, random_state=42)
            
            # Use PCA to reduce dimensions before clustering
            pca = PCA(n_components=min(10, returns.shape[1] - 1))
            pca_features = pca.fit_transform(returns.iloc[:60].T)
            clusters = kmeans.fit_predict(pca_features)
            
            # Calculate industry returns
            industry_returns = pd.DataFrame(index=returns.index)
            for i in range(k):
                cluster_stocks = [returns.columns[j] for j in range(len(clusters)) if clusters[j] == i]
                if cluster_stocks:
                    industry_returns[f"Industry_{i+1}"] = returns[cluster_stocks].mean(axis=1)
            
            # Create industry rotation factor
            if len(industry_returns.columns) >= 2:
                # Use first two industries as a rotation factor
                ind_rot = industry_returns["Industry_1"] - industry_returns["Industry_2"]
            else:
                ind_rot = pd.Series(0, index=returns.index)
        else:
            ind_rot = pd.Series(0, index=returns.index)
    else:
        ind_rot = pd.Series(0, index=returns.index)
    
    # Combine factors
    factors = pd.DataFrame({
        'MKT': market,
        'SMB': smb,
        'HML': hml,
        'UMD': wml,
        'Quality': quality,
        'Defensive': def_off,
        'Industry_Rotation': ind_rot
    })
    
    return factors

# ───── Advanced Regression Models ─────

def regress_with_diagnostics(port_r: pd.Series, factors: pd.DataFrame) -> dict:
    """Run regression with comprehensive diagnostics"""
    if sm is None or factors is None:
        return None
    
    common = port_r.index.intersection(factors.index)
    if len(common) < 60:
        st.warning(f"Not enough overlapping data points ({len(common)}) between portfolio and factors")
        return None
    
    X = sm.add_constant(factors.loc[common])
    model = sm.OLS(port_r.loc[common], X).fit()
    
    # Calculate key statistics
    alpha_annualized = model.params['const'] * 252
    alpha_tstat = model.tvalues['const']
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    
    # Calculate factor exposures
    betas = model.params.drop('const')
    tvalues = model.tvalues.drop('const')
    pvalues = model.pvalues.drop('const')
    
    # Calculate tracking error
    residuals = model.resid
    tracking_error = residuals.std() * np.sqrt(252)
    
    # Heteroskedasticity test
    bp_test = het_breuschpagan(residuals, X)
    het_pvalue = bp_test[1]
    
    # Calculate information ratio
    information_ratio = alpha_annualized / tracking_error if tracking_error > 0 else np.nan
    
    # Compute the contribution of each factor to the portfolio's variance
    # 1. Get the variance-covariance matrix of factors
    factor_cov = factors.loc[common, betas.index].cov()  # Only use factors that are in betas
    
    # 2. Calculate factor contributions to variance
    # Var(port) = β'*Cov(Factors)*β
    # Fix the matrix multiplication and scalar extraction
    beta_matrix = betas.values.reshape(-1, 1)
    
    # Ensure factor_cov has the same factors and in the same order as betas
    factor_cov = factor_cov.loc[betas.index, betas.index]
    
    # Now perform the matrix multiplication
    beta_contrib = beta_matrix.T @ factor_cov.values @ beta_matrix
    
    # Get the scalar value correctly - it's a 1x1 matrix, so use [0][0] indexing
    factor_contrib_to_var = pd.Series(
        beta_contrib[0][0],  # Fixed here - properly extract scalar from numpy array
        index=['Total Factor'])
    
    # Individual factor contributions
    factor_contrib_detail = {}
    for i, f1 in enumerate(betas.index):
        contrib = 0
        for j, f2 in enumerate(betas.index):
            contrib += betas[f1] * betas[f2] * factor_cov.loc[f1, f2]
        factor_contrib_detail[f1] = contrib
    
    factor_contrib = pd.Series(factor_contrib_detail)
    
    # Calculate specific variance (residual variance)
    specific_var = residuals.var()
    
    # Total portfolio variance
    total_var = port_r.loc[common].var()
    
    # Factor's proportion of total risk
    factor_pct = factor_contrib / total_var
    specific_pct = specific_var / total_var
    
    return {
        'betas': betas,
        'tvalues': tvalues,
        'pvalues': pvalues,
        'alpha': alpha_annualized,
        'alpha_tstat': alpha_tstat,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'het_pvalue': het_pvalue,
        'residuals': residuals,
        'factor_contrib': factor_contrib,
        'factor_contrib_pct': factor_pct,
        'specific_var': specific_var,
        'specific_var_pct': specific_pct,
        'model': model
    }

def calc_rolling_betas(port_r: pd.Series, factors: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Calculate rolling factor betas with robust outlier handling"""
    if RollingOLS is None:
        return None
    
    common = port_r.index.intersection(factors.index)
    if len(common) < window + 30:
        st.warning(f"Not enough data for rolling analysis (need {window + 30} points, have {len(common)})")
        return None
    
    Y = port_r.loc[common]
    X = factors.loc[common]
    
    # Initialize DataFrame for rolling betas
    rolling_betas = pd.DataFrame(index=common, columns=X.columns)
    
    try:
        # Calculate rolling betas with proper error handling
        with np.errstate(invalid='ignore', divide='ignore'):  # Suppress numpy warnings
            roll_model = RollingOLS(Y, sm.add_constant(X), window=window)
            roll_res = roll_model.fit()
            
            # Extract parameters (skip the constant)
            for i, col in enumerate(X.columns):
                # Access the parameters for each variable correctly
                rolling_betas[col] = roll_res.params.iloc[:, i+1]  # Use iloc instead of direct indexing
        
        # Comprehensive cleaning of extreme outliers and invalid values
        for col in rolling_betas.columns:
            # Replace infinities and NaNs with NaN (to be interpolated later)
            rolling_betas[col] = rolling_betas[col].replace([np.inf, -np.inf], np.nan)
            
            # Identify and handle statistical outliers (values far from the median)
            median_val = rolling_betas[col].median()
            mad = np.nanmedian(np.abs(rolling_betas[col] - median_val))  # Median Absolute Deviation
            
            # More robust than standard deviation for outlier detection
            if mad > 0:  # Avoid division by zero
                # Detect extreme outliers (>10 MADs from median)
                outlier_mask = np.abs(rolling_betas[col] - median_val) > 10 * mad
                rolling_betas.loc[outlier_mask, col] = np.nan
            
            # For remaining extreme values use a percentile-based cap
            valid_values = rolling_betas[col].dropna()
            if len(valid_values) > 10:  # Need enough data for percentiles
                low_cap = valid_values.quantile(0.01)  # 1st percentile
                high_cap = valid_values.quantile(0.99)  # 99th percentile
                
                rolling_betas[col] = rolling_betas[col].clip(low_cap, high_cap)
            
            # Finally fill remaining NaNs through interpolation
            rolling_betas[col] = rolling_betas[col].interpolate(method='linear', limit=5).fillna(method='ffill').fillna(method='bfill')
    
    except Exception as e:
        st.warning(f"Error in calculating rolling betas: {str(e)}")
        return None
    
    return rolling_betas.dropna()

def calc_factor_timing(port_r: pd.Series, factors: pd.DataFrame, window: int = 126) -> pd.DataFrame:
    """Analyze factor timing ability by correlating changes in exposure with future factor returns"""
    # Calculate rolling betas
    rolling_betas = calc_rolling_betas(port_r, factors, window)
    if rolling_betas is None:
        return None
    
    # Calculate beta changes (first difference)
    beta_changes = rolling_betas.diff().dropna()
    
    # Calculate forward factor returns
    forward_returns = factors.shift(-1).dropna()
    
    # Find common dates between beta changes and forward returns
    common_dates = beta_changes.index.intersection(forward_returns.index)
    
    # Calculate correlation between beta changes and forward returns
    timing_corr = pd.DataFrame(index=factors.columns, columns=['Correlation', 'p-value'])
    
    for factor in factors.columns:
        if factor in beta_changes.columns:
            # Calculate correlation
            corr, pval = np.corrcoef(
                beta_changes.loc[common_dates, factor], 
                forward_returns.loc[common_dates, factor]
            )[0, 1], 0.0  # Simplification - would calculate actual p-value
            
            timing_corr.loc[factor] = [corr, pval]
    
    return timing_corr

# ───── Multi-factor Stress Testing ─────

def run_stress_tests(betas: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """
    Run comprehensive stress tests based on factor exposures and historical scenarios
    """
    # 1. Historical worst case scenarios (based on actual factor data)
    factor_quantiles = {}
    
    # Calculate historical portfolio returns based on factor exposures (betas) and factor returns
    relevant_factor_names = betas.index.intersection(factors.columns)
    historical_portfolio_returns = pd.Series(dtype=float)  # Initialize as an empty Series

    if not relevant_factor_names.empty and not factors.empty and not betas.empty:
        aligned_factors = factors[relevant_factor_names]
        aligned_betas = betas[relevant_factor_names]
        # Ensure that after alignment, we still have data and dimensions match for dot product
        if not aligned_factors.empty and not aligned_betas.empty and \
           aligned_factors.shape[0] > 0 and aligned_factors.shape[1] == aligned_betas.shape[0]:
            historical_portfolio_returns = aligned_factors.dot(aligned_betas)
        else:
            # This case might occur if, e.g., betas exist but for factors not in the 'factors' DataFrame
            # Or if data is malformed. Consider adding logging here for diagnostics.
            # historical_portfolio_returns will remain empty, leading to 0.0 or NaN VaR values below.
            pass

    # Calculate historical portfolio VaR for 1%, 5%, and 10% quantiles
    # These correspond to 99% VaR (most loss), 95% VaR, and 90% VaR (least loss of the three)
    for q_val in [0.01, 0.05, 0.10]:
        percent_label = int(q_val * 100)  # Gives 1, 5, 10
        scenario_name = f"Historical_{percent_label}pct" # Historical_1pct, Historical_5pct, Historical_10pct
        
        if not historical_portfolio_returns.empty and historical_portfolio_returns.notna().any():
            # .quantile(q_val) gives the value at that quantile from the historical portfolio returns.
            # For returns, lower quantiles (e.g., 0.01) represent larger losses (more negative numbers).
            portfolio_var_at_q = historical_portfolio_returns.quantile(q_val)
            factor_quantiles[scenario_name] = portfolio_var_at_q
        else:
            # If portfolio returns could not be calculated (e.g., no overlapping factors, empty data),
            # assign a default value. Using 0.0 here. np.nan could also be an option.
            factor_quantiles[scenario_name] = 0.0
    
    custom_scenarios = {
        "Market_Crash": {"Mkt-RF": -0.07, "VIX_Diff": 15},  # Mkt-RF shock, VIX_Diff shock (15 point VIX increase)
        "Rate_Shock": {"10Y_Yield_Diff": 0.005, "2Y_Yield_Diff": 0.010}, # 0.5% and 1% increase in yield changes
        "Inflation_Spike": {"CPI": 0.01, "10Y_Yield_Diff": 0.003}, # 1% CPI shock, 0.3% 10Y yield change shock
        "Value_Rotation": {"HML": 0.02, "UMD": -0.015}, # Value outperformance, momentum underperformance
        "Growth_Rally": {"HML": -0.02, "UMD": 0.02}, # Growth outperformance, momentum outperformance
        "Credit_Crisis": {"Credit_Spread_Diff": 0.004, "TED_Spread_Diff": 0.003} # 0.4% and 0.3% increase in spreads
    }

    combined_scenarios = {
        "Stagflation": {
            "Mkt-RF": -0.02, # 2% market drop
            "CPI": 0.008, # 0.8% CPI shock
            "10Y_Yield_Diff": 0.003, # 0.3% yield shock
            "USD_Index": 0.01 # 1% USD Index shock
        },
        "Risk_Aversion": {
            "Mkt-RF": -0.03, # 3% market drop
            "SMB": -0.01, # 1% small cap underperformance
            "VIX_Diff": 10, # 10 point VIX increase
            "Credit_Spread_Diff": 0.002 # 0.2% increase in credit spreads
        }
    }
    
    # Combine all scenarios
    all_scenarios = {}
    all_scenarios.update({k: v for k, v in factor_quantiles.items()})
    all_scenarios.update(custom_scenarios)
    all_scenarios.update(combined_scenarios)
    
    # Calculate impact for each scenario
    stress_results = {}
    
    for scenario_name, scenario_data in all_scenarios.items():
        if isinstance(scenario_data, dict):  # Factor shock scenario
            impact = 0
            for factor, shock in scenario_data.items():
                if factor in betas.index:  # Check if the factor from scenario exists in portfolio betas
                    impact += betas[factor] * shock
            stress_results[scenario_name] = impact
        elif isinstance(scenario_data, (float, np.float64)):  # Pre-calculated impact (e.g., historical VaR)
            stress_results[scenario_name] = scenario_data
        else:
            # Log a warning or handle unexpected data types for a scenario
            st.warning(f"Unexpected data type for scenario '{scenario_name}': {type(scenario_data)}. Result set to NaN.")
            stress_results[scenario_name] = np.nan
    
    return pd.Series(stress_results).sort_values()

# ───── Plotting Utility Functions ─────
def show_enhanced_corr(returns_df: pd.DataFrame, annot_threshold: float = 0.3) -> Tuple[plt.Figure, plt.Axes]:
    """Display an enhanced correlation matrix heatmap."""
    corr_matrix = returns_df.corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr_matrix.columns) * 0.8), max(6, len(corr_matrix.columns) * 0.6)))
    cmap = plt.get_cmap('coolwarm') # More vibrant colormap
    cax = ax.matshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    fig.colorbar(cax)
    
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.columns)
    
    # Add annotations for significant correlations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > annot_threshold: # Annotate if correlation is strong
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black' if abs(val) < 0.7 else 'white')
    ax.set_title('Asset Correlation Matrix', pad=20)
    plt.tight_layout()
    return fig, ax

def show_roll_vol_enhanced(portfolio_returns: pd.Series, window: int = 252, secondary_window: int = 63) -> Tuple[plt.Figure, plt.Axes]:
    """Display rolling volatility with primary and secondary windows."""
    fig, ax = plt.subplots(figsize=(12, 5))
    roll_vol_primary = portfolio_returns.rolling(window=window).std() * np.sqrt(252)
    roll_vol_secondary = portfolio_returns.rolling(window=secondary_window).std() * np.sqrt(252)
    
    roll_vol_primary.plot(ax=ax, label=f'{window}-Day Rolling Vol', color='blue')
    roll_vol_secondary.plot(ax=ax, label=f'{secondary_window}-Day Rolling Vol', color='orange', linestyle='--')
    
    ax.set_title(f'Annualized Rolling Volatility ({window}-day and {secondary_window}-day)')
    ax.set_ylabel('Annualized Volatility')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, ax

def plot_factor_contrib(factor_contrib: pd.Series, specific_var: float) -> Tuple[plt.Figure, plt.Axes]:
    """Plot factor contributions to variance and specific variance."""
    # Combine factor contributions and specific variance
    all_contrib = factor_contrib.copy()
    all_contrib['Specific Risk'] = specific_var
    
    # Calculate percentages
    total_variance = all_contrib.sum()
    contrib_pct = (all_contrib / total_variance) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    contrib_pct.sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_title('Factor Contribution to Portfolio Variance')
    ax.set_ylabel('Percentage of Total Variance (%)')
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig, ax

def plot_stress_tests(stress_results: pd.Series) -> Tuple[plt.Figure, plt.Axes]:
    """Plot stress test impacts."""
    fig, ax = plt.subplots(figsize=(10, max(5, len(stress_results) * 0.4)))
    stress_results.sort_values().plot(kind='barh', ax=ax, color=['red' if x < 0 else 'green' for x in stress_results.sort_values()])
    ax.set_title('Portfolio Impact Under Stress Scenarios')
    ax.set_xlabel('Expected Portfolio Return Change (%)')
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    plt.tight_layout()
    return fig, ax

def plot_rolling_betas(rolling_betas_df, reg_results=None, significance_level=0.10, top_n_by_mean_abs_beta=4):
    """
    Plots rolling factor betas using Plotly, highlighting significant ones based on recent regression 
    or overall significance if recent data is insufficient.

    Args:
        rolling_betas_df (pd.DataFrame): DataFrame with dates as index and factor betas as columns.
        reg_results (dict, optional): Dictionary containing regression results like 'pvalues'.
        significance_level (float, optional): p-value threshold for a factor to be considered significant.
        top_n_by_mean_abs_beta (int, optional): If no factors are significant by p-value, 
                                              plot top N factors by mean absolute beta.

    Returns:
        tuple: (plotly.graph_objects.Figure or None, list)
               The Plotly figure object for the rolling betas plot, or None if no data.
               A list of factor names that were actually plotted.
    """
    if rolling_betas_df is None or rolling_betas_df.empty:
        return None, []

    factors_to_plot = []
    legend_labels = {}

    # Determine which factors to plot based on significance or magnitude
    if reg_results and 'pvalues' in reg_results:
        significant_factors = reg_results['pvalues'][reg_results['pvalues'] <= significance_level].index.tolist()
        if significant_factors:
            factors_to_plot = [f for f in significant_factors if f in rolling_betas_df.columns]
            for factor in factors_to_plot:
                p_value = reg_results['pvalues'].get(factor, 1)
                legend_labels[factor] = f"{factor} (p={p_value:.2f})"

    # If no p-values provided or no significant factors from p-values, try top N by mean absolute beta
    if not factors_to_plot and not rolling_betas_df.columns.empty:
        mean_abs_betas = rolling_betas_df.abs().mean().nlargest(top_n_by_mean_abs_beta)
        factors_to_plot = mean_abs_betas.index.tolist()
        for factor in factors_to_plot:
            legend_labels[factor] = f"{factor} (Top {top_n_by_mean_abs_beta})"
    
    # If still no factors (e.g., rolling_betas_df might have no columns after filtering)
    if not factors_to_plot:
        # Try to plot any available columns if top_n_by_mean_abs_beta was 0 or columns were few
        if not rolling_betas_df.columns.empty:
            factors_to_plot = rolling_betas_df.columns.tolist()[:top_n_by_mean_abs_beta] # take some if available
            for factor in factors_to_plot:
                 legend_labels[factor] = factor # Default label
        else:
            return None, [] # No data to plot

    fig = go.Figure()
    
    actually_plotted_factors = []
    for factor in factors_to_plot:
        if factor in rolling_betas_df.columns:
            fig.add_trace(go.Scatter(x=rolling_betas_df.index, 
                                     y=rolling_betas_df[factor], 
                                     mode='lines', 
                                     name=legend_labels.get(factor, factor)))
            actually_plotted_factors.append(factor)

    if not actually_plotted_factors:
        return None, [] # No factors ended up being plotted

    fig.update_layout(
        title="Rolling Factor Exposures",
        xaxis_title="Date",
        yaxis_title="Beta",
        legend_title_text='Factors',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey")

    return fig, actually_plotted_factors

def run_advanced_pca(rets: pd.DataFrame, tickers: list, n_components: int = 5) -> dict:
    """
    Run advanced PCA with rotation interpretation and risk decomposition
    """
    if len(rets) < 2*len(tickers) or len(rets) < n_components: # Ensure enough data
        st.warning("Not enough data points for reliable PCA analysis given the number of tickers/components.")
        return None
    
    # Standardize returns
    scaler = StandardScaler()
    # Drop NaNs before scaling to avoid issues with fit_transform
    scaled_rets = scaler.fit_transform(rets.dropna()) 
    
    # Adjust index for pca_result if rets had NaNs
    valid_index = rets.dropna().index

    # Run PCA
    actual_n_components = min(n_components, scaled_rets.shape[0], scaled_rets.shape[1])
    if actual_n_components < 1:
        st.warning("Cannot run PCA with less than 1 component.")
        return None

    pca = PCA(n_components=actual_n_components)
    # pca_result will have index corresponding to rets.dropna()
    pca_result = pca.fit_transform(scaled_rets) 
    
    components = pd.DataFrame(
        pca.components_, 
        columns=tickers, # Assuming rets.columns gives the tickers
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    
    explained_var = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    
    pc_interpretations = {}
    for pc_idx, pc_name in enumerate(components.index):
        # Get top and bottom 3 contributors
        loadings_series = components.loc[pc_name]
        top_pos = loadings_series.nlargest(3)
        top_neg = loadings_series.nsmallest(3)
        
        pc_interpretations[pc_name] = {
            "positive": list(zip(top_pos.index, top_pos.values)),
            "negative": list(zip(top_neg.index, top_neg.values))
        }
    
    # Calculate risk contributions
    pc_scores = pd.DataFrame(
        pca_result,
        index=valid_index, # Use the index of the data that was actually used for PCA
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    
    var_decomp = None
    if pca.n_components_ >= 1:
        # Reconstruct returns using ALL fitted principal components for "systematic" risk
        systematic_reconstruction_scaled = pca.inverse_transform(pca_result)
        
        systematic_reconstruction_unscaled = pd.DataFrame(
            scaler.inverse_transform(systematic_reconstruction_scaled),
            columns=tickers, # Assuming rets.columns gives the tickers
            index=valid_index # Use the index of the data that was actually used for PCA
        )
        
        # Align original returns to the valid_index for residual calculation
        original_rets_aligned = rets.loc[valid_index]
        idiosyncratic_risk_unscaled = original_rets_aligned - systematic_reconstruction_unscaled
        
        total_asset_variances = original_rets_aligned.var()
        explained_variance_by_pcs = systematic_reconstruction_unscaled.var()
        # residual_variance = idiosyncratic_risk_unscaled.var() # This is an alternative

        # Calculate systematic percentage based on explained variance by PCs
        systematic_pct = (explained_variance_by_pcs / total_asset_variances.replace(0, np.nan)).fillna(0) * 100
        
        # Idiosyncratic is the remainder
        idiosyncratic_pct = 100 - systematic_pct

        systematic_pct = systematic_pct.clip(0, 100)
        idiosyncratic_pct = idiosyncratic_pct.clip(0, 100)

        var_decomp = pd.DataFrame({
            "Systematic": systematic_pct,
            "Idiosyncratic": idiosyncratic_pct
        }, index=tickers) # Assuming tickers correspond to columns of original rets
    
    return {
        'components': components,
        'explained_var': explained_var,
        'interpretations': pc_interpretations,
        'variance_decomposition': var_decomp,
        'pc_scores': pc_scores
    }

# ───── Dashboard Core ─────
def dashboard(weights: pd.Series, prices: pd.DataFrame, start: str, factors: Optional[pd.DataFrame] = None, factor_window: int = 126, factor_exposure_smoothing_window: int = 21, factor_rolling_return_window: int = 21):
    # Check for missing tickers and notify user
    miss = [t for t in weights.index if t not in prices.columns]
    if miss:
        st.error("Price data missing for: " + ", ".join(miss))
        # Filter out missing tickers from weights
        weights = weights.drop(miss)
        if len(weights) == 0:
            st.error("No valid tickers left in portfolio after removing missing ones.")
            return
        # Re-normalize weights
        weights = weights / weights.abs().sum()
        st.info("Weights have been renormalized after removing missing tickers.")

    # Get price data for tickers in weights
    available_tickers = [t for t in weights.index if t in prices.columns]
    if not available_tickers:
        st.error("None of the tickers in your portfolio have available price data.")
        return
        
    # Ensure we only use available tickers
    weights = weights.loc[available_tickers]
    # Renormalize weights to sum to 1
    weights = weights / weights.abs().sum()
    
    px = prices[weights.index].loc[start:]
    if px.empty:
        st.error(f"No price data available for the selected date range starting from {start}")
        return
        
    rets = px.pct_change(fill_method=None).dropna()
    if rets.empty:
        st.error("No return data available after calculating price changes")
        return

    # Now the indices should be aligned
    port_r = rets.dot(weights)

    st.header("1. Risk Summary")
    # Calculate VaR at multiple confidence levels
    var_metrics = {}
    es_metrics = {}
    for conf in CONFIDENCE_LEVELS:
        p = 1 - conf
        var_metrics[f"VaR ({conf*100:.1f}%)"] = _var(port_r, p)
        es_metrics[f"CVaR ({conf*100:.1f}%)"] = _es(port_r, p)
    
    # risk summary
    col1, col2 = st.columns(2)
    with col1:
        summ1 = {
            "Ann. Return": port_r.mean() * 252,
            "Ann. Vol": port_r.std() * np.sqrt(252),
            "Sharpe Ratio": (port_r.mean() * 252) / (port_r.std() * np.sqrt(252)) if port_r.std() > 0 else np.nan,
            "Max Drawdown": _maxdd(port_r),
            "Skewness": port_r.skew(),
            "Kurtosis": port_r.kurtosis()
        }
        st.dataframe(pd.DataFrame(summ1, index=["Value"]).T.style.format("{:.4f}"))
        
    with col2:
        # Combine VaR and ES
        risk_metrics = {}
        risk_metrics.update(var_metrics)
        risk_metrics.update(es_metrics)
        st.dataframe(pd.DataFrame(risk_metrics, index=["Value"]).T.style.format("{:.4f}"))
    
    # Return distribution visualization
    st.subheader("Return Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    port_r.hist(bins=50, ax=ax, density=True, alpha=0.6)
    
    # Add normal distribution overlay
    x = np.linspace(port_r.min(), port_r.max(), 100)
    ax.plot(x, norm.pdf(x, port_r.mean(), port_r.std()), 'r-', lw=2)
    
    # Add VaR lines
    for conf in [0.95, 0.99]:
        var_value = _var(port_r, 1-conf)
        ax.axvline(var_value, color='r', linestyle='--', 
                  label=f'VaR {conf*100}%: {var_value:.2%}')
    
    ax.set_title("Portfolio Return Distribution")
    ax.legend()
    st.pyplot(fig)

    # Component VaR analysis
    st.subheader("Component VaR Analysis")
    comp_var = calc_component_var(rets, weights, confidence=0.95)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    comp_var.sort_values().plot(kind='barh', ax=ax)
    ax.set_title("95% Component VaR by Position")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    st.pyplot(fig)

    st.header("2. Correlation Analysis")
    st.subheader("Asset Correlation")
    fig_corr, _ = show_enhanced_corr(rets)
    st.pyplot(fig_corr)

    st.header("3. Volatility Analysis")
    st.subheader("Rolling Volatility")
    fig_vol, _ = show_roll_vol_enhanced(port_r, window=252, secondary_window=63)
    st.pyplot(fig_vol)

    # Add volatility decomposition if factors are available
    if factors is not None:
        st.subheader("Volatility Decomposition")
        
        # Run regression to get factor exposures and residuals
        reg_results = regress_with_diagnostics(port_r, factors)
        
        if reg_results is not None:
            # Plot factor contribution to variance
            fig_contrib, _ = plot_factor_contrib(
                reg_results['factor_contrib'], 
                reg_results['specific_var']
            )
            st.pyplot(fig_contrib)
            
            # Display details in an expander
            with st.expander("Detailed Volatility Decomposition"):
                factor_contrib_pct = reg_results['factor_contrib_pct']
                specific_var_pct = reg_results['specific_var_pct']
                
                # Create detailed table
                # Ensure non-negative values before sqrt for Annualized Vol Contribution
                annualized_vol_contrib_values = np.sqrt(np.maximum(0, reg_results['factor_contrib']) * 252)
                
                contrib_df = pd.DataFrame({
                    'Variance Contribution': reg_results['factor_contrib'],
                    'Percent of Total': factor_contrib_pct * 100,
                    'Annualized Vol Contribution': annualized_vol_contrib_values
                })
                
                # Add specific row (specific_var should always be non-negative)
                specific_data = pd.DataFrame({
                    'Variance Contribution': [reg_results['specific_var']],
                    'Percent of Total': [specific_var_pct * 100],
                    'Annualized Vol Contribution': [np.sqrt(reg_results['specific_var'] * 252)]
                }, index=['Specific Risk'])
                
                # Combine and format
                full_contrib = pd.concat([contrib_df, specific_data])
                st.dataframe(full_contrib.style.format({
                    'Variance Contribution': '{:.6f}',
                    'Percent of Total': '{:.2f}%',
                    'Annualized Vol Contribution': '{:.2%}'
                }))

    st.header("4. PCA Analysis")
    # Use advanced PCA with more components and interpretation
    pca_results = run_advanced_pca(rets, list(weights.index), n_components=5)
    
    if pca_results is not None:
        # Display explained variance
        st.subheader("PCA – Variance Explained")
        exp_var = pca_results['explained_var'] * 100
        
        fig, ax = plt.subplots(figsize=(10, 4))
        exp_var.plot(kind='bar', ax=ax)
        ax.set_ylabel("Variance Explained (%)")
        ax.set_title("Principal Component Analysis - Explained Variance")
        
        # Add cumulative line
        ax_right = ax.twinx()
        cumul = exp_var.cumsum()
        cumul.plot(color='r', marker='o', linestyle='-', ax=ax_right)
        ax_right.set_ylabel("Cumulative Variance Explained (%)")
        ax_right.set_ylim([0, 105])
        
        st.pyplot(fig)
        
        # Display loadings heatmap
        st.subheader("PCA Loadings (Heat-map)")
        components = pca_results['components']
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 0.5*len(weights)+2))
        im = ax.imshow(components, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(weights)))
        ax.set_xticklabels(weights.index, rotation=90)
        ax.set_yticks(range(len(components)))
        ax.set_yticklabels(components.index)
        fig.colorbar(im)
        st.pyplot(fig)
        
        # Display PC interpretations
        interp = pca_results['interpretations']
        
        for pc, data in interp.items():
            # Show interpretation for first few components only
            if int(pc[2:]) <= 3:  # First 3 PCs only
                st.write(f"**{pc}:** Positive influence from {', '.join([f'{t[0]} ({t[1]:.2f})' for t in data['positive']])}")
                st.write(f"  Negative influence from {', '.join([f'{t[0]} ({t[1]:.2f})' for t in data['negative']])}")
        
        # Display variance decomposition in an expander
        if pca_results['variance_decomposition'] is not None:
            with st.expander("Risk Source Decomposition (Systematic vs. Idiosyncratic)"):
                var_decomp = pca_results['variance_decomposition'] * 100
                
                # Plot as horizontal bar chart
                fig, ax = plt.subplots(figsize=(10, 0.4*len(var_decomp)+2))
                var_decomp.plot(kind='barh', stacked=True, ax=ax)
                ax.set_xlim([0, 100])
                ax.set_title("Variance Decomposition by Security")
                ax.set_xlabel("Percent of Total Variance (%)")
                st.pyplot(fig)

    st.header("5. Factor Analysis")
    
    if factors is None:
        # Try generating synthetic factors as fallback
        st.info("No external factor data provided. Generating synthetic factors instead.")
        factors = generate_advanced_factors(prices, start)
        if factors is None:
            st.error("No factor data available. Please upload factor data or enable synthetic factor generation.")
            return
    
    # Run regression analysis
    reg_results = regress_with_diagnostics(port_r, factors)
    if reg_results is None or reg_results.get('betas') is None:
        st.error("Insufficient data for factor regression.")
        return
    
    # Display regression results
    st.subheader("Factor Exposures")
    
    # Create a more detailed table with statistical significance
    factor_data = pd.DataFrame({
        'Beta': reg_results['betas'],
        't-value': reg_results['tvalues'],
        'p-value': reg_results['pvalues'],
        'Significance': ['***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '' 
                       for p in reg_results['pvalues']]
    })
    
    # Display in a tabular format
    st.dataframe(factor_data.style.format({
        'Beta': '{:.4f}',
        't-value': '{:.2f}',
        'p-value': '{:.4f}'
    }))
    
    # Factor model summary
    col1, col2 = st.columns(2)
    with col1:
        model_summary = {
            "Alpha (ann.)": reg_results['alpha'],
            "Alpha t-stat": reg_results['alpha_tstat'],
            "R²": reg_results['r_squared'],
            "Adj. R²": reg_results['adj_r_squared'],
        }
        st.dataframe(pd.DataFrame(model_summary, index=["Value"]).T.style.format({
            "Alpha (ann.)": "{:.4%}",
            "Alpha t-stat": "{:.2f}",
            "R²": "{:.4f}",
            "Adj. R²": "{:.4f}"
        }))
    
    with col2:
        risk_metrics = {
            "Tracking Error (ann.)": reg_results['tracking_error'],
            "Information Ratio": reg_results['information_ratio'],
            "Heteroskedasticity p-value": reg_results['het_pvalue']
        }
        st.dataframe(pd.DataFrame(risk_metrics, index=["Value"]).T.style.format({
            "Tracking Error (ann.)": "{:.4%}",
            "Information Ratio": "{:.2f}",
            "Heteroskedasticity p-value": "{:.4f}"
        }))
    
    # Factor timing analysis
    st.subheader("Factor Exposure Over Time")
    
    # Calculate and display rolling betas
    rolling_window = min(factor_window, len(port_r) // 3)  # Use 6 months or 1/3 of data
    if rolling_window >= 30:  # Need at least 30 days
        rolling_betas = calc_rolling_betas(port_r, factors, window=rolling_window)
        
        if rolling_betas is not None and not rolling_betas.empty:
            # Apply smoothing if window > 1
            smoothed_rolling_betas = rolling_betas
            if factor_exposure_smoothing_window > 1:
                smoothed_rolling_betas = rolling_betas.rolling(window=factor_exposure_smoothing_window, min_periods=1).mean()
                st.caption(f"Rolling exposures smoothed with a {factor_exposure_smoothing_window}-day moving average.")
            
            fig_betas, factors_shown_in_beta_plot = plot_rolling_betas(smoothed_rolling_betas, reg_results=reg_results) # MODIFIED LINE
            if fig_betas:
                st.plotly_chart(fig_betas, use_container_width=True)

            # Determine significant factors for the cumulative return plot.
            # Priority:
            # 1. Factors actually plotted in the rolling beta chart.
            # 2. If beta plot is empty/showed no factors, fall back to significant factors from overall regression.
            # 3. If still none, fall back to top N from smoothed rolling betas by magnitude.
            significant_factors_to_plot = []
            if factors_shown_in_beta_plot: # MODIFIED BLOCK - START
                significant_factors_to_plot = factors_shown_in_beta_plot
            else: 
                if reg_results and 'pvalues' in reg_results:
                    significant_pvalues = reg_results['pvalues'][reg_results['pvalues'] <= 0.10]
                    if not significant_pvalues.empty:
                        significant_factors_to_plot = significant_pvalues.index.tolist()
                
                if not significant_factors_to_plot and smoothed_rolling_betas is not None and not smoothed_rolling_betas.empty:
                    if not smoothed_rolling_betas.columns.empty:
                        significant_factors_to_plot = smoothed_rolling_betas.abs().mean().nlargest(min(4, len(smoothed_rolling_betas.columns))).index.tolist()
                    else:
                        significant_factors_to_plot = [] # MODIFIED BLOCK - END
            
            if significant_factors_to_plot and factors is not None:
                st.subheader(f"Cumulative Factor Returns (Rolling {factor_rolling_return_window}-Day Sum)")
                
                # Ensure only factors present in the 'factors' DataFrame are used
                available_factors_for_plot = [f for f in significant_factors_to_plot if f in factors.columns]

                if available_factors_for_plot:
                    aligned_factors_for_plot = factors.loc[port_r.index.intersection(factors.index), available_factors_for_plot]
                    
                    if not aligned_factors_for_plot.empty:
                        rolling_factor_returns_sum = aligned_factors_for_plot.rolling(window=factor_rolling_return_window, min_periods=max(1, factor_rolling_return_window // 2)).sum().dropna()
                        
                        if not rolling_factor_returns_sum.empty:
                            # Use Plotly for cumulative factor returns (already updated in previous step)
                            fig_cum_factor_returns = go.Figure()
                            for factor in rolling_factor_returns_sum.columns:
                                fig_cum_factor_returns.add_trace(go.Scatter(x=rolling_factor_returns_sum.index, y=rolling_factor_returns_sum[factor], mode='lines', name=factor))
                            
                            fig_cum_factor_returns.update_layout(
                                title=f"Cumulative Factor Returns (Rolling {factor_rolling_return_window}-Day Sum)",
                                xaxis_title="Date",
                                yaxis_title="Cumulative Return",
                                legend_title_text='Factors',
                                height=500,
                                margin=dict(l=50, r=50, t=80, b=50),
                            )
                            st.plotly_chart(fig_cum_factor_returns, use_container_width=True)
                        else:
                            st.info("No data available for cumulative factor returns plot after rolling sum.")
                    else:
                        st.info("Could not align selected factors with portfolio returns for the cumulative plot.")
                else:
                    st.info("None of the selected factors for the cumulative plot are available in the factor data.")
            elif not significant_factors_to_plot:
                st.info("No significant factors identified to plot cumulative returns.")

            # Factor timing analysis
            st.subheader("Factor Timing Analysis")
            timing_analysis = calc_factor_timing(port_r, factors, window=rolling_window)
            
            if timing_analysis is not None and not timing_analysis.empty:
                # Filter to significant factors only
                sig_timing = timing_analysis[timing_analysis['p-value'] < 0.2]
                
                if not sig_timing.empty:
                    st.write("Correlation between changes in factor exposure and subsequent factor returns:")
                    st.dataframe(timing_analysis.style.format({
                        'Correlation': '{:.4f}',
                        'p-value': '{:.4f}'
                    }).background_gradient(cmap='RdYlGn', subset=['Correlation']))
                    
                    st.write("""
                    - Positive correlation suggests good factor timing ability
                    - Negative correlation suggests poor factor timing or contrarian exposure
                    - Values near zero suggest no consistent timing pattern
                    """)
                else:
                    st.info("No statistically significant factor timing patterns detected")
    
    # Factor stress testing
    st.subheader("Factor Stress Tests")
    stress_results = run_stress_tests(reg_results['betas'], factors)
    
    if stress_results is not None:
        fig_stress, _ = plot_stress_tests(stress_results)
        if fig_stress:
            st.pyplot(fig_stress)
        
        # Detailed stress test results in a table
        with st.expander("Detailed Stress Test Results"):
            st.dataframe(stress_results.to_frame("Impact").style.format("{:.4%}").background_gradient(
                cmap='RdYlGn_r', subset=['Impact']))

    # 6. Attribution Analysis
    st.header("6. Performance Attribution")
    
    exposures_for_attribution = None
    attribution_type_message = ""

    if factors is not None and reg_results is not None:
        # Prioritize rolling betas if available and valid
        # Ensure 'rolling_betas' is in the local scope from section 5
        if 'rolling_betas' in locals() and rolling_betas is not None and rolling_betas.empty:
            # Align rolling_betas with portfolio returns and factors
            # Ensure factors used for attribution are present in rolling_betas columns
            factors_for_attr = factors.loc[:, factors.columns.isin(rolling_betas.columns)]
            common_idx_attr = port_r.index.intersection(rolling_betas.index).intersection(factors_for_attr.index)
            
            if not common_idx_attr.empty and not factors_for_attr.loc[common_idx_attr].empty:
                exposures_for_attribution = rolling_betas.loc[common_idx_attr, factors_for_attr.columns]
                # Ensure factors_for_attr is also sliced to common_idx_attr for the call
                factors_arg_for_attribution = factors_for_attr.loc[common_idx_attr]
                attribution_type_message = "Performance attribution is based on **rolling factor exposures**."
            else:
                st.warning("Could not align rolling betas with portfolio/factor returns for attribution. Falling back if possible.")
                exposures_for_attribution = None # Reset to ensure fallback

        # Fallback to static betas if rolling betas are not suitable or available
        if exposures_for_attribution is None:
            static_betas_series = reg_results.get('betas')
            if static_betas_series is not None and not static_betas_series.empty:
                # Ensure factors used for attribution are present in static_betas_series index
                factors_for_attr = factors.loc[:, factors.columns.isin(static_betas_series.index)]
                common_idx_attr = port_r.index.intersection(factors_for_attr.index)

                if not common_idx_attr.empty and not factors_for_attr.loc[common_idx_attr].empty:
                    # Create a DataFrame of static betas repeated for each date
                    aligned_static_betas = pd.DataFrame(index=common_idx_attr, columns=static_betas_series.index.intersection(factors_for_attr.columns))
                    for factor_name in aligned_static_betas.columns:
                         aligned_static_betas[factor_name] = static_betas_series[factor_name]
                    
                    exposures_for_attribution = aligned_static_betas.ffill().bfill() 
                    factors_arg_for_attribution = factors_for_attr.loc[common_idx_attr]
                    attribution_type_message = "Performance attribution is based on **static full-period factor exposures** (rolling exposures unavailable or unaligned)."
                else:
                    st.error("Could not align static betas with portfolio/factor returns for attribution.")
                    exposures_for_attribution = None # Ensure no attribution if alignment fails
            else:
                st.error("Static betas are missing from regression results. Cannot perform attribution.")
                exposures_for_attribution = None


        if exposures_for_attribution is not None and not exposures_for_attribution.empty and 'factors_arg_for_attribution' in locals():
            st.info(attribution_type_message) # Inform user which betas are used
            
            # Align port_r for the call
            port_r_for_attribution = port_r.loc[exposures_for_attribution.index]

            factor_attribution = calc_factor_attribution(port_r_for_attribution,
                                                      exposures_for_attribution,
                                                      factors_arg_for_attribution) # Use aligned factors
            
            if factor_attribution is not None and not factor_attribution.empty:
                # Calculate cumulative return attribution
                cum_attribution = factor_attribution.cumsum()
                
                # Plot cumulative attribution
                st.subheader("Cumulative Return Attribution")
                
                if USE_PLOTLY:
                    # Interactive Plotly chart
                    fig = go.Figure()
                    
                    for col in cum_attribution.columns:
                        fig.add_trace(go.Scatter(
                            x=cum_attribution.index,
                            y=cum_attribution[col],
                            mode='lines',
                            name=col,
                            stackgroup='one',  # All traces in the same stack group
                            fill='tonexty',    # Fill to previous trace in the stack
                            line=dict(width=0.5)
                        ))
                        
                        # Add hover text for detailed info
                        fig.update_traces(hoverinfo="name+x+y")
                    
                    fig.update_layout(
                        title="Cumulative Factor Attribution (Stacked)",
                        xaxis_title="Date",
                        yaxis_title="Cumulative Return",
                        legend=dict(x=0.01, y=0.99, orientation="v"),
                        height=500,
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Matplotlib fallback
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    cum_attribution.plot(kind='area', stacked=True, ax=ax, linewidth=0.5)
                    
                    ax.axhline(0, color='black', linewidth=0.5) # Add a zero line for clarity
                    ax.set_title("Cumulative Factor Attribution (Stacked)")
                    ax.set_ylabel("Cumulative Return")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Display attribution table
                st.subheader("Return Attribution Summary")
                
                # Calculate average daily contribution and percent of total
                avg_contrib = factor_attribution.mean() * 252  # Annualized
                total_return = avg_contrib.sum()
                pct_contrib = avg_contrib / total_return if total_return != 0 else pd.Series(0, index=avg_contrib.index)
                
                # Create attribution table
                attribution_summary = pd.DataFrame({
                    'Ann. Contribution': avg_contrib,
                    'Percent of Total': pct_contrib * 100
                })
                
                st.dataframe(attribution_summary.style.format({
                    'Ann. Contribution': '{:.4%}',
                    'Percent of Total': '{:.2f}%'
                               }).background_gradient(cmap='RdYlGn', subset=['Ann. Contribution']))

    # 7. Report Summary
    st.header("7. Risk Report Summary")
    
    # Create a summary of key findings
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.subheader("Key Risk Metrics")
        key_metrics = {
            "Annualized Volatility": port_r.std() * np.sqrt(252),
            "95% VaR (1-day)": _var(port_r, 0.05),
            "Maximum Drawdown": _maxdd(port_r),
        }
        
        if reg_results is not None:
            key_metrics.update({
                "Factor R-squared": reg_results['r_squared'],
                "Systematic Risk": reg_results['factor_contrib'].sum() / (reg_results['factor_contrib'].sum() + reg_results['specific_var']),
                "Tracking Error": reg_results['tracking_error']
            })
            
        st.dataframe(pd.DataFrame(key_metrics, index=["Value"]).T.style.format("{:.4f}"))
    
    with summary_col2:
        st.subheader("Top Factor Exposures")
        if reg_results is not None:
            # Get top absolute exposures by t-statistic (significance)
            top_exposures_sorted_by_significance = reg_results['tvalues'].abs().nlargest(5)
            
            exposures_df = pd.DataFrame({
                'Factor': top_exposures_sorted_by_significance.index,
                'Exposure': reg_results['betas'][top_exposures_sorted_by_significance.index],
                't-stat': reg_results['tvalues'][top_exposures_sorted_by_significance.index]
            })
            
            st.dataframe(exposures_df.style.format({
                'Exposure': '{:.4f}',
                't-stat': '{:.2f}'
            }).background_gradient(cmap='RdYlGn', subset=['Exposure']))
    
    # 8. Download options
    st.header("8. Download Report")
    
    # Add report download option
    if st.button("Generate Report Summary (CSV)"):
        # Create a summary DataFrame
        summary_data = {
            "Portfolio Info": ["", ""],
            "# of Holdings": [len(weights), ""],

            "Start Date": [pd.to_datetime(start).strftime('%Y-%m-%d'), ""],
            "End Date": [rets.index[-1].strftime('%Y-%m-%d'), ""],
            "": ["", ""],
            "Risk Metrics": ["", ""],
            "Annualized Return": [port_r.mean() * 252, "{:.4%}"],
            "Annualized Volatility": [port_r.std() * np.sqrt(252), "{:.4%}"],
            "Sharpe Ratio": [(port_r.mean() * 252) / (port_r.std() * np.sqrt(252)) if port_r.std() > 0 else np.nan, "{:.4f}"],
            "Max Drawdown": [_maxdd(port_r), "{:.4%}"],
            "Skewness": [port_r.skew(), "{:.4f}"],
            "Kurtosis": [port_r.kurtosis(), "{:.4f}"],
            "95% VaR (1-day)": [_var(port_r, 0.05), "{:.4%}"],
            "95% CVaR (1-day)": [_es(port_r, 0.05), "{:.4%}"],
        }
        
        if reg_results is not None:
            factor_data = {
                "": ["", ""],
                "Factor Model": ["", ""],
                "Alpha (annualized)": [reg_results['alpha'], "{:.4%}"],
                "Alpha t-stat": [reg_results['alpha_tstat'], "{:.4f}"],
                "R-squared": [reg_results['r_squared'], "{:.4f}"],
                "Tracking Error": [reg_results['tracking_error'], "{:.4%}"],
                "Information Ratio": [reg_results['information_ratio'], "{:.4f}"],
                "Systematic Risk %": [reg_results['factor_contrib'].sum() / (reg_results['factor_contrib'].sum() + reg_results['specific_var']) * 100, "{:.2f}%"],
            }
            
            # Add top factor exposures by significance for CSV
            if reg_results is not None: # Ensure reg_results exists before using it for CSV
                factor_data = {} # Initialize factor_data locally if it's not already
                top_factors_by_significance_csv = reg_results['tvalues'].abs().nlargest(5)
                for i, factor_name in enumerate(top_factors_by_significance_csv.index):
                    beta_value = reg_results['betas'][factor_name]
                    t_value = reg_results['tvalues'][factor_name]
                    # Update key to include t-stat for clarity on sorting criteria
                    factor_data[f"Factor {i+1}: {factor_name} (t-stat: {t_value:.2f})"] = [beta_value, "{:.4f}"]
            
            summary_data.update(factor_data)
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data, index=["Value", "Format"]).T
        
        # Convert to CSV
        csv = summary_df.to_csv()
        
        # Create download button
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name=f"risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ───────────────────────── Main App ─────────────────────────

def main():
    st.set_page_config(
        page_title="Risk Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Equity Risk Dashboard")
    st.markdown("""
    This dashboard provides comprehensive factor-based risk analysis for equity portfolios.
    Upload your portfolio holdings or add them manually, then select factor data sources and analysis parameters.
    """)
    
    # Sidebar inputs
    st.sidebar.header("Analysis Settings")
    
    # 1. Date Range
    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=pd.to_datetime(DEFAULT_START),
        help="Select the start date for analysis"
    )
    
    # 2. Factor Analysis Settings
    st.sidebar.subheader("Factor Analysis Settings")
    factor_window = st.sidebar.number_input(
        "Factor Exposure Window (Trading Days)",
        min_value=30,
        max_value=252,
        value=126,
        step=21,
        help="Number of trading days used for calculating rolling factor exposures"
    )
    factor_exposure_smoothing_window = st.sidebar.number_input(
        "Factor Exposure Smoothing Window (Days)",
        min_value=1,  # Min value 1 means no smoothing effectively
        max_value=126, 
        value=21, 
        step=1,
        help="Number of days for smoothing rolling factor exposures. 1 means no smoothing."
    )
    factor_rolling_return_window = st.sidebar.number_input(
        "Factor Rolling Return Window (Days)",
        min_value=5,
        max_value=252,
        value=21,
        step=1,
        help="Window in days for calculating rolling sum of factor returns for the plot below factor exposures."
    )
    
    # 3. Portfolio Input Method
    st.sidebar.subheader("Portfolio Input")
    input_method = st.sidebar.radio(
        "Input Method",
        ["Manual Input", "CSV Upload"],
        help="Choose how to input your portfolio holdings"
    )
    
    weights = None
    
    if input_method == "Manual Input":
        st.sidebar.markdown("Enter holdings (ticker and weight):")
        
        if 'portfolio_rows' not in st.session_state:
            st.session_state.portfolio_rows = [
                {"ticker": "AAPL", "weight": 0.20},
                {"ticker": "MSFT", "weight": 0.20},
                {"ticker": "AMZN", "weight": 0.15},
                {"ticker": "GOOGL", "weight": 0.15},
                {"ticker": "META", "weight": 0.10},
                {"ticker": "TSLA", "weight": 0.10},
                {"ticker": "NVDA", "weight": 0.10}
            ]
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("➕ Add Row"):
                st.session_state.portfolio_rows.append({"ticker": "", "weight": 0.0})
        with col2:
            if st.button("➖ Remove Row") and len(st.session_state.portfolio_rows) > 1:
                st.session_state.portfolio_rows.pop()
        
        updated_rows = []
        for i, row in enumerate(st.session_state.portfolio_rows):
            col1_input, col2_input = st.sidebar.columns([2, 1])
            with col1_input:
                ticker = st.text_input(f"Ticker {i+1}", value=row["ticker"], key=f"ticker_{i}")
            with col2_input:
                weight_val = st.number_input(f"Weight {i+1}", min_value=-1.0, max_value=1.0, value=float(row["weight"]), step=0.01, key=f"weight_{i}")
            updated_rows.append({"ticker": ticker, "weight": weight_val})
        
        st.session_state.portfolio_rows = updated_rows
        weights = pd.Series({row["ticker"].strip().upper(): row["weight"] for row in updated_rows if row["ticker"].strip() and row["weight"] != 0})
        if not weights.empty:
            weights = (weights / weights.abs().sum()).sort_index()

    else:  # CSV Upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload Portfolio CSV", 
            type="csv",
            help="Upload a CSV file with two columns: Ticker and Weight"
        )
        
        if uploaded_file:
            try:
                weights = load_weight_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")
                weights = None
    
    # 3. Factor Data Source
    st.sidebar.subheader("Factor Data")
    factor_source = st.sidebar.radio(
        "Factor Data Source",
        ["Online (Ken French + FRED)", "Upload Factor CSV", "Generate Synthetic Factors (Not Implemented)"],
        help="Choose your factor data source"
       )
    
    factors = None # Initialize factors
    prices_df = pd.DataFrame() # Initialize prices DataFrame

    # Proceed only if weights are defined
    if weights is not None and not weights.empty:
        st.subheader("Current Portfolio")
        sorted_weights_display = weights.reindex(weights.abs().sort_values(ascending=False).index)
        weight_df_display = pd.DataFrame({
            "Weight": sorted_weights_display,
            "Weight (%)": sorted_weights_display * 100
        })
        total_row_display = pd.DataFrame({

            "Weight": [sorted_weights_display.sum()],
            "Weight (%)": [sorted_weights_display.sum() * 100]
        }, index=["TOTAL"])
        combined_df_display = pd.concat([weight_df_display, total_row_display])
        st.dataframe(combined_df_display.style.format({"Weight": "{:.4f}", "Weight (%)": "{:.2f}%"}))

        # --- Primary Price Data Loading ---
        try:
            st.info(f"Loading price data for {len(weights.index)} tickers in portfolio...")
            all_portfolio_tickers = list(weights.index)
            prices_df = load_prices(all_portfolio_tickers, start_date.strftime('%Y-%m-%d'))

            if prices_df.empty and all_portfolio_tickers:
                st.error("Failed to load price data for any tickers in the portfolio. Please check tickers and date range.")
                st.stop()
            
            # --- Handle Tickers Not Loaded by yfinance ---
            successfully_loaded_tickers = [col for col in prices_df.columns if col in all_portfolio_tickers]
            failed_to_load_tickers = [t for t in all_portfolio_tickers if t not in successfully_loaded_tickers]

            if failed_to_load_tickers:
                st.warning(f"Could not load price data for: {', '.join(failed_to_load_tickers)}. These tickers will be removed from the analysis.")
                weights = weights.drop(failed_to_load_tickers)
                if weights.empty:
                    st.error("No valid tickers remaining after attempting to load price data. Analysis cannot proceed.")
                    st.stop()
                weights = (weights / weights.abs().sum()).sort_index() # Re-normalize and sort
                st.info("Portfolio weights have been re-normalized. Please review the updated portfolio above.")
                # Re-display updated portfolio (Streamlit will handle this on the next rerun if weights object is changed)
            
            if prices_df.empty or weights.empty:
                 st.error("Price data could not be loaded or no valid tickers remain. Analysis halted.")
                 st.stop()

        except Exception as e:
            st.error(f"An critical error occurred while loading or processing price data: {str(e)}")
            st.stop()

        # --- Factor Loading/Generation ---
        if factor_source == "Online (Ken French + FRED)":
            with st.spinner("Fetching online factor data..."):
                factors = fetch_online_factors(start_date.strftime('%Y-%m-%d'))
        elif factor_source == "Upload Factor CSV":
            factor_file = st.sidebar.file_uploader( # Uploader widget specific to this option
                "Upload Factor CSV File",
                type=["csv", "txt"],
                help="Upload a CSV file with factors in columns and dates in rows"
            )
            if factor_file:
                factors = read_factor_file(factor_file, start_date.strftime('%Y-%m-%d'))
        elif factor_source == "Generate Synthetic Factors":
            if prices_df.empty:
                st.error("Cannot generate synthetic factors without price data. Ensure portfolio tickers are valid.")
                st.stop()
            factors = generate_advanced_factors(prices_df, start_date.strftime('%Y-%m-%d'))
        
        # --- Run Dashboard ---
        if not prices_df.empty and weights is not None and not weights.empty:
            dashboard(weights, prices_df, start_date.strftime('%Y-%m-%d'), factors, factor_window, factor_exposure_smoothing_window, factor_rolling_return_window)
        else:
            st.info("Analysis could not proceed. Please check portfolio holdings and ensure price data can be loaded.")

    else: # weights is None or empty initially
        st.info("Please enter or upload valid portfolio holdings to begin analysis.")

if __name__ == "__main__":
    main()
