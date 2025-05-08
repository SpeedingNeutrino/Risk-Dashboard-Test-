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

import io
import sys
import json
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates

import yfinance as yf

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
try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False

if norm is None or PCA is None:
    st.error("SciPy and scikit-learn are required – run `pip install --upgrade numpy scipy scikit-learn`.")
    st.stop()

# ───────────────────────────── Config ──────────────────────────────
DATA_PATH = Path(r"C:/Users/devansh.bhatt/OneDrive - Lighthouse Canton Pte Ltd/Desktop/Python Files/Market Data/sp_top1500_ohlc_data.csv")

DEFAULT_START = "2018-01-01"

# Extended factor mappings
FF5_MAP = {"MKT": "Mkt-RF", "SMB": "SMB", "HML": "HML", "RMW": "RMW", "CMA": "CMA", "UMD": "Mom"}

# Extended FRED data for macro factors
MACRO_FRED = {
    "10Y_Yield": "DGS10",
    "2Y_Yield": "DGS2",
    "Yield_Curve": "T10Y2Y",  # 10Y-2Y spread
    "USD_Index": "DTWEXBGS",
    "WTI_Oil": "DCOILWTICO",
    "CPI": "CPIAUCSL",
    "VIX": "VIXCLS",  
    "TED_Spread": "TEDRATE", # TED spread (3m Libor - 3m T-bill)
    "Credit_Spread": "BAA10Y", # Moody's BAA - 10Y Treasury
    "Unemployment": "UNRATE"
}

# Categorized factor groups
FACTOR_GROUPS = {
    "Traditional": ["MKT", "SMB", "HML", "UMD", "RMW", "CMA"],
    "Style": ["Value", "Size", "Momentum", "Quality", "Low_Vol", "Growth"],
    "Macro": ["Yield_Curve", "USD_Index", "WTI_Oil", "VIX", "Credit_Spread", "TED_Spread"],
    "Smart Beta": ["Min_Vol", "Max_Div", "Equal_Weight", "Quality_Tilt", "ESG_Tilt"]
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
            # Fix FutureWarning by not relying on date_parser
            raw_data = pdr.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start=start)
            # Convert index to datetime after fetching if needed
            ff = raw_data[0] / 100
            ff.index = pd.to_datetime(ff.index)
            ff.index = ff.index.tz_localize(None)
            ff.rename(columns=FF5_MAP, inplace=True)
            ff = ff.loc[ff.index >= pd.to_datetime(start)]
            st.success(f"Successfully fetched {len(ff)} rows of Fama-French 5 factor data")
            st.write(ff.tail())
            ff_data = ff
    except Exception as e:
        st.sidebar.error(f"Failed to fetch Fama-French data: {str(e)}")
    
    # Try to fetch Momentum factor separately
    try:
        with st.sidebar.expander("Momentum Factor Fetch"):
            st.info("Attempting to fetch Momentum factor...")
            # Fix FutureWarning by not relying on date_parser
            raw_data = pdr.DataReader("F-F_Momentum_Factor_daily", "famafrench", start=start)
            mom = raw_data[0] / 100
            mom.index = pd.to_datetime(mom.index)
            mom.index = mom.index.tz_localize(None)
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
            # Fetch one by one for better error isolation
            macro_dfs = []
            for name, ticker in MACRO_FRED.items():
                try:
                    df = pdr.DataReader(ticker, "fred", start)
                    df.rename(columns={ticker: name}, inplace=True)
                    macro_dfs.append(df)
                    st.write(f"✅ {name}: {len(df)} rows")
                except Exception as e:
                    st.write(f"❌ {name}: {str(e)}")
            
            if macro_dfs:
                macro = pd.concat(macro_dfs, axis=1)
                # Handle non-trading days differently for macro data - fix deprecated method
                macro = macro.asfreq('B').ffill()
                
                # Create first differences for yield series
                rate_cols = [col for col in macro.columns if 'Yield' in col or 'Spread' in col]
                for col in rate_cols:
                    if col in macro.columns:
                        macro[f"{col}_Diff"] = macro[col].diff()
                
                # Calculate returns for others
                non_rate_cols = [col for col in macro.columns if col not in rate_cols 
                              and not col.endswith('_Diff')]
                for col in non_rate_cols:
                    if col in macro.columns:
                        macro[col] = macro[col].pct_change()
                
                # Drop original yield columns to keep only changes
                macro = macro.drop(columns=rate_cols)
                # Fix deprecated ffill method
                macro = macro.ffill().dropna()
                
                st.success(f"Successfully fetched {len(macro)} rows of FRED data")
                st.write(macro.tail())
                macro_data = macro
            else:
                st.error("Failed to fetch any FRED data series")
    except Exception as e:
        st.sidebar.error(f"Failed to fetch FRED data: {str(e)}")
    
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
    # 1. Historical worst case scenarios
    factor_quantiles = {}
    for q in [0.01, 0.05, 0.10]:
        factor_quantiles[f"Historical_{int(q*100)}pct"] = factors.quantile(q)
    
    # 2. Custom stress scenarios
    custom_scenarios = {
        "Market_Crash": {"MKT": -0.05, "VIX_Diff": 0.15},
        "Rate_Shock": {"10Y_Yield_Diff": 0.005, "2Y_Yield_Diff": 0.010},
        "Inflation_Spike": {"CPI": 0.01, "10Y_Yield_Diff": 0.003},
        "Value_Rotation": {"HML": 0.02, "UMD": -0.015},
        "Growth_Rally": {"HML": -0.02, "UMD": 0.02},
        "Credit_Crisis": {"Credit_Spread_Diff": 0.004, "TED_Spread_Diff": 0.003}
    }
    
    # 3. Combined stress scenarios
    combined_scenarios = {
        "Stagflation": {
            "MKT": -0.02,
            "CPI": 0.008,
            "10Y_Yield_Diff": 0.003,
            "USD_Index": 0.01
        },
        "Risk_Aversion": {
            "MKT": -0.03,
            "SMB": -0.01,
            "VIX_Diff": 0.10,
            "Credit_Spread_Diff": 0.002
        }
    }
    
    # Combine all scenarios
    all_scenarios = {}
    all_scenarios.update({k: v for k, v in factor_quantiles.items()})
    all_scenarios.update(custom_scenarios)
    all_scenarios.update(combined_scenarios)
    
    # Calculate impact for each scenario
    stress_results = {}
    
    for scenario_name, scenario_values in all_scenarios.items():
        impact = 0
        for factor, shock in scenario_values.items():
            if factor in betas.index:
                impact += betas[factor] * shock
        stress_results[scenario_name] = impact
    
    return pd.Series(stress_results).sort_values()

# ───── Advanced PCA Analysis ─────

def run_advanced_pca(rets: pd.DataFrame, tickers: list, n_components: int = 5) -> dict:
    """
    Run advanced PCA with rotation interpretation and risk decomposition
    """
    if len(rets) < 2*len(tickers):
        st.warning("Not enough data points for reliable PCA analysis")
        return None
    
    # Standardize returns
    scaler = StandardScaler()
    scaled_rets = scaler.fit_transform(rets)
    
    # Run PCA
    pca = PCA(n_components=min(n_components, len(tickers), len(rets)))
    pca_result = pca.fit_transform(scaled_rets)
    
    # Extract components and loadings
    components = pd.DataFrame(
        pca.components_, 
        columns=tickers,
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    
    # Calculate explained variance
    explained_var = pd.Series(
        pca.explained_variance_ratio_,
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    
    # Interpret PC loadings
    pc_interpretations = {}
    for pc in components.index:
        # Get top and bottom 3 contributors
        top_pos = components.loc[pc].nlargest(3)
        top_neg = components.loc[pc].nsmallest(3)
        
        pc_interpretations[pc] = {
            "positive": list(zip(top_pos.index, top_pos.values)),
            "negative": list(zip(top_neg.index, top_neg.values))
        }
    
    # Calculate risk contributions
    pc_scores = pd.DataFrame(
        pca_result,
        index=rets.index,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    
    # Reconstruct with only first PC to see systematic risk
    if pca.n_components_ >= 1:
        systematic_risk = pca.inverse_transform(
            np.column_stack([pc_scores.iloc[:, 0], np.zeros((len(pc_scores), pca.n_components_ - 1))])
        )
        systematic_risk = pd.DataFrame(systematic_risk, columns=tickers, index=rets.index)
        
        # Calculate idiosyncratic risk (residual)
        idiosyncratic_risk = rets - pd.DataFrame(
            scaler.inverse_transform(systematic_risk), 
            columns=tickers, 
            index=rets.index
        )
        
        # Variance ratio of systematic vs idiosyncratic
        systematic_var = systematic_risk.var()
        idiosyncratic_var = idiosyncratic_risk.var()
        total_var = rets.var()
        
        var_decomp = pd.DataFrame({
            "Systematic": systematic_var / total_var,
            "Idiosyncratic": idiosyncratic_var / total_var
        }, index=tickers)
    else:
        var_decomp = None
    
    return {
        'components': components,
        'explained_var': explained_var,
        'interpretations': pc_interpretations,
        'variance_decomposition': var_decomp,
        'pc_scores': pc_scores
    }

# ───── Plots and Visualizations ─────

def show_enhanced_corr(rets, annot=True, cmap='coolwarm'):
    """Display correlation matrix with clustered heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Attempt to cluster for better visualization
    corr = rets.corr()
    
    # Optional: hierarchical clustering for better visualization
    try:
        from scipy.cluster import hierarchy
        from scipy.spatial import distance
        
        # Calculate distance matrix
        dist = distance.pdist(corr)
        link = hierarchy.linkage(dist, method='ward')
        
        # Get the cluster order
        clust_order = hierarchy.dendrogram(link, no_plot=True)['leaves']
        
        # Reorder correlation matrix
        corr = corr.iloc[clust_order, clust_order]
    except:
        # Fall back if clustering fails
        pass
    
    # Plot as heatmap
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap=cmap)
    
    # Add annotations and labels
    if annot and len(corr) <= 15:
        for i in range(len(corr)):
            for j in range(len(corr)):
                val = corr.iloc[i, j]
                color = 'white' if abs(val) > 0.7 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    
    # Add labels and colorbar
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    
    # Add decorations
    plt.title("Asset Correlation Matrix")
    plt.tight_layout()
    fig.colorbar(im)
    return fig, ax

def show_roll_vol_enhanced(pr, window=252, secondary_window=63):
    """rolling volatility plot with multiple windows"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Calculate multiple rolling windows
    vol_long = _roll(pr, window)
    vol_short = _roll(pr, secondary_window)
    
    # Plot both
    vol_long.plot(ax=ax, label=f'{window}-day')
    vol_short.plot(ax=ax, label=f'{secondary_window}-day', linestyle='--')
    
    # Add historical average
    avg_vol = vol_long.mean()
    ax.axhline(y=avg_vol, color='r', linestyle='-', alpha=0.7, 
               label=f'Avg: {avg_vol:.2%}')
    
    # Add +/- 1 std band around average
    vol_std = vol_long.std()
    ax.axhline(y=avg_vol + vol_std, color='r', linestyle=':', alpha=0.5,
               label=f'+1σ: {avg_vol + vol_std:.2%}')
    ax.axhline(y=max(0, avg_vol - vol_std), color='r', linestyle=':', alpha=0.5,
               label=f'-1σ: {max(0, avg_vol - vol_std):.2%}')
    
    ax.set_title("Rolling Annualized Volatility Analysis")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax

def plot_rolling_betas(rolling_betas, reg_results=None, significance_level=0.05):
    """
    Plot rolling factor exposures for the most significant factors.
    If reg_results is provided, only show factors that are statistically significant at the specified level.
    """
    if rolling_betas is None or rolling_betas.empty:
        return None, None
    
    # If regression results are provided, filter for significant factors
    significant_factors = []
    if reg_results is not None and 'pvalues' in reg_results:
        significant_factors = reg_results['pvalues'][reg_results['pvalues'] <= significance_level].index.tolist()
        
        if len(significant_factors) == 0:
            # If no factors are significant, use top factors by absolute magnitude
            significant_factors = rolling_betas.abs().mean().nlargest(4).index.tolist()
            
        # Create plot only with significant factors
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for factor in significant_factors:
            if factor in rolling_betas.columns:
                rolling_betas[factor].plot(ax=ax, label=f"{factor} (p={reg_results['pvalues'][factor]:.4f})")
        
        title_suffix = f"Significant at {(1-significance_level)*100:.0f}% Confidence"
        ax.set_title(f"Rolling Factor Exposures ({title_suffix})")
    else:
        # Fallback to previous behavior if no regression results
        # Select top factors by magnitude of average absolute value
        top_factors = rolling_betas.abs().mean().nlargest(4).index.tolist()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for factor in top_factors:
            rolling_betas[factor].plot(ax=ax, label=factor)
        
        ax.set_title(f"Rolling Factor Exposures (Top Factors by Magnitude)")
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    return fig, ax

def plot_factor_contrib(factor_contrib, specific_var):
    """Plot factor contribution to risk"""
    # Calculate total risk
    total_risk = factor_contrib.sum() + specific_var
    
    # Convert to percentages
    factor_pct = (factor_contrib / total_risk * 100).sort_values(ascending=False)
    specific_pct = specific_var / total_risk * 100
    
    # Create a complete series with specific risk
    complete_contrib = factor_pct.copy()
    complete_contrib["Specific"] = specific_pct
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    complete_contrib.plot(kind='bar', ax=ax)
    
    ax.set_title("Factor Contribution to Variance")
    ax.set_ylabel("Percent of Total Variance (%)")
    ax.grid(True, alpha=0.3)
    
    # Add a line for cumulative contribution
    ax_right = ax.twinx()
    cumul = complete_contrib.sort_values(ascending=False).cumsum()
    cumul.plot(color='r', marker='o', linestyle='-', ax=ax_right)
    ax_right.set_ylabel("Cumulative Contribution (%)")
    
    return fig, ax

def plot_stress_tests(stress_results):
    """Plot stress test results"""
    if stress_results is None or len(stress_results) == 0:
        return None, None
    
    # Sort by impact
    sorted_results = stress_results.sort_values()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['r' if x < 0 else 'g' for x in sorted_results]
    sorted_results.plot(kind='barh', ax=ax, color=colors)
    
    ax.set_title("Factor Stress Test Results")
    ax.set_xlabel("Portfolio Return Impact")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True, alpha=0.3)
    
    return fig, ax

# ───── Dashboard Core ─────

def dashboard(weights: pd.Series, prices: pd.DataFrame, start: str, factors: Optional[pd.DataFrame] = None, factor_window: int = 126):
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
                contrib_df = pd.DataFrame({
                    'Variance Contribution': reg_results['factor_contrib'],
                    'Percent of Total': factor_contrib_pct * 100,
                    'Annualized Vol Contribution': np.sqrt(reg_results['factor_contrib'] * 252)
                })
                
                # Add specific row
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
        st.subheader("Principal Component Interpretation")
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
            fig_betas, _ = plot_rolling_betas(rolling_betas, reg_results=reg_results)
            if fig_betas:
                st.pyplot(fig_betas)
            
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
                    **Interpretation:**
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
    
    if factors is not None and reg_results is not None:
        # Calculate factor attribution
        factor_attribution = calc_factor_attribution(port_r, 
                                                  pd.DataFrame(reg_results['betas']).T.reindex(port_r.index), 
                                                  factors)
        
        if factor_attribution is not None and not factor_attribution.empty:
            # Calculate cumulative return attribution
            cum_attribution = factor_attribution.cumsum()
            
            # Plot cumulative attribution
            st.subheader("Cumulative Return Attribution")
            
            if USE_PLOTLY:
                # Interactive Plotly chart
                fig = go.Figure()
                
                # Add each factor contribution
                for col in cum_attribution.columns:
                    fig.add_trace(go.Scatter(
                        x=cum_attribution.index,
                        y=cum_attribution[col],
                        mode='lines',
                        name=col,
                        stackgroup='one'
                    ))
                
                fig.update_layout(
                    title="Cumulative Factor Attribution",
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
                cum_attribution.plot(kind='area', stacked=True, ax=ax)
                ax.set_title("Cumulative Factor Attribution")
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
            # Get top absolute exposures
            top_exposures = reg_results['betas'].abs().nlargest(5)
            exposures_df = pd.DataFrame({
                'Factor': top_exposures.index,
                'Exposure': reg_results['betas'][top_exposures.index],
                't-stat': reg_results['tvalues'][top_exposures.index]
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
            
            # Add top factor exposures
            top_factors = reg_results['betas'].abs().nlargest(5)
            for i, (factor, beta) in enumerate(zip(top_factors.index, reg_results['betas'][top_factors.index])):
                factor_data[f"Factor {i+1}: {factor}"] = [beta, "{:.4f}"]
                
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
    # Replace slider with numerical input
    factor_window = st.sidebar.number_input(
        "Factor Exposure Window (Trading Days)",
        min_value=30,
        max_value=252,
        value=126,
        step=21,
        help="Number of trading days used for calculating rolling factor exposures"
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
        
        # Initialize with a few default rows
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
        
        # Add/remove row buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("➕ Add Row"):
                st.session_state.portfolio_rows.append({"ticker": "", "weight": 0.0})
        with col2:
            if st.button("➖ Remove Row") and len(st.session_state.portfolio_rows) > 1:
                st.session_state.portfolio_rows.pop()
        
        # Show current rows
        updated_rows = []
        for i, row in enumerate(st.session_state.portfolio_rows):
            col1, col2 = st.sidebar.columns([2, 1])
            with col1:
                ticker = st.text_input(f"Ticker {i+1}", value=row["ticker"], key=f"ticker_{i}")
            with col2:
                # Allow negative weights for short positions
                weight = st.number_input(f"Weight {i+1}", min_value=-1.0, max_value=1.0, value=float(row["weight"]), step=0.01, key=f"weight_{i}")
            updated_rows.append({"ticker": ticker, "weight": weight})
        
        st.session_state.portfolio_rows = updated_rows
        weights = pd.Series({row["ticker"]: row["weight"] for row in updated_rows if row["ticker"] and row["weight"] != 0})
    
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
        ["Online (Ken French + FRED)", "Upload Factor CSV", "Generate Synthetic Factors"],
        help="Choose your factor data source"
       )
    
    factors = None
    
       
    if factor_source == "Upload Factor CSV":
        factor_file = st.sidebar.file_uploader(
            "Upload Factor CSV",
            type=["csv", "txt"],
            help="Upload a CSV file with factors in columns and dates in rows"
        )
        
        if factor_file:
            factors = read_factor_file(factor_file, start_date)
    
    elif factor_source == "Online (Ken French + FRED)":
        # This will be fetched during dashboard execution
        pass
    
    elif factor_source == "Generate Synthetic Factors":
        # This will be generated during dashboard execution
        pass
    
    # Load price data
    try:
        # Instead of loading from CSV, we'll use yfinance for all price data loading
        # We'll just pass a minimum set of default tickers to initialize the data
        # The actual tickers will be loaded later based on the portfolio weights
        default_tickers = ["SPY"]
        prices = load_prices(default_tickers, start_date.strftime('%Y-%m-%d'))
    except Exception as e:
        st.error(f"Error loading price data: {str(e)}")
        st.stop()
    
    # Display missing tickers warning if needed and try to download them
    if weights is not None and len(weights) > 0:
        # missing_tickers = [t for t in weights.index if t not in prices.columns]
        missing_tickers = [t for t in weights.index]

        if missing_tickers:
            st.warning(f"Price data not found in CSV for: {', '.join(missing_tickers)}")
            
            # Try to download missing data from yfinance
            missing_data = download_missing_data(missing_tickers, start_date)
            
            if missing_data is not None and not missing_data.empty:
                # Ensure the format matches our main price dataframe (with Close)
                missing_tickers_found = list(missing_data.columns)
                st.success(f"Successfully downloaded data for: {', '.join(missing_tickers_found)}")
                
                # Combine with existing price data
                combined_prices = pd.concat([prices, missing_data], axis=1)
                prices = combined_prices
                
                # Update missing tickers list
                still_missing = [t for t in missing_tickers if t not in missing_data.columns]
                if still_missing:
                    st.warning(f"Still missing data for: {', '.join(still_missing)}")
                    weights = weights.drop(still_missing)
            else:
                # If download failed, drop the missing tickers
                weights = weights.drop(missing_tickers)
    
    # Check if we have valid weights to continue
    if weights is not None and len(weights) > 0:
        # Display current portfolio
        st.subheader("Current Portfolio")
        
        # Sort by absolute weight descending to show largest positions first
        sorted_weights = weights.reindex(weights.abs().sort_values(ascending=False).index)
        
        # Display as a table
        weight_df = pd.DataFrame({
            "Weight": sorted_weights,
            "Weight (%)": sorted_weights * 100
        })
        
        # Show total at the bottom
        total_row = pd.DataFrame({
            "Weight": [sorted_weights.sum()],
            "Weight (%)": [sorted_weights.sum() * 100]
        }, index=["TOTAL"])
        
        combined_df = pd.concat([weight_df, total_row])
        
        st.dataframe(combined_df.style.format({
            "Weight": "{:.4f}",
            "Weight (%)": "{:.2f}%"
        }))
        
        # Fetch online factors if selected
        if factor_source == "Online (Ken French + FRED)":
            with st.spinner("Fetching online factor data..."):
                factors = fetch_online_factors(start_date.strftime('%Y-%m-%d'))
        
        # Ensure the 'start' argument is passed to the load_prices function consistently
        prices = load_prices(list(weights.index), start_date.strftime('%Y-%m-%d'))

        # Run the dashboard
        dashboard(weights, prices, start_date.strftime('%Y-%m-%d'), factors, factor_window)
    else:
        st.info("Please enter valid portfolio holdings to begin analysis.")

def download_missing_data(tickers, start_date):
    """
    Download price data for tickers missing from the CSV file using yfinance
    """
    try:
        import yfinance as yf
        st.info(f"Downloading data for {len(tickers)} tickers using yfinance...")
        
        # Download data with auto_adjust=True which puts adjusted prices in 'Close' column
        data = yf.download(tickers, start=start_date, auto_adjust=True)
        
        if not data.empty:
            if len(tickers) == 1:
                # For a single ticker, yfinance returns a different format
                prices = data['Close'].to_frame(tickers[0])
            else:
                # For multiple tickers, we get a MultiIndex DataFrame
                prices = data['Close']
                
            # Make sure the column names match exactly the ticker names in weights
            # This ensures proper alignment for the dot product operation
            prices.columns = [col.upper() if isinstance(col, str) else col for col in prices.columns]
            
            st.success(f"Successfully downloaded data for {len(prices.columns)} tickers")
            return prices
        else:
            st.warning("No data returned from yfinance")
            return None
        
    except Exception as e:
        st.error(f"Error downloading data from yfinance: {str(e)}")
        return None

if __name__ == "__main__":
    main()
