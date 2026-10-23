"""
Market data loader for empirical experiments (Appendix B.4/B.5).

Market data: QuantX Dataloader — https://quantx.limex.pro/

Price history in this project is loaded through the QuantX Dataloader, a
research-ready market-data service available through a REST API and as an
MCP server for AI agents. It is built on market data from Massive
(https://massive.com/) and prepares every stock history for backtesting:
  - prices adjusted for dividends, ready for total-return research;
  - one continuous history when a company changes its ticker;
  - clean separation when a ticker has belonged to different companies;
  - careful corporate-action handling and fine-grained data polishing.

Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
MASSIVE_API_KEY at https://massive.com/.

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)
"""

from __future__ import annotations

import io
import logging
import os
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_DATALOADER_BASE = os.environ.get(
    "DATALOADER_API_URL", "https://quantx-api.limex.pro/apis/dataloader"
).rstrip("/")
_TIMEOUT = 120


def _get_headers(with_massive_key: bool = True) -> dict[str, str]:
    key = os.environ.get("DATALOADER_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "DATALOADER_API_KEY is not set. "
            "Obtain it at https://quantx.limex.pro/ and set it in your environment."
        )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if with_massive_key:
        massive_key = os.environ.get("MASSIVE_API_KEY", "")
        if massive_key:
            headers["X-Massive-Api-Key"] = massive_key
    return headers


def _post_fetch_ohlcvs(payload: dict, headers: dict) -> requests.Response:
    """POST /fetch-ohlcvs and surface errors clearly."""
    resp = requests.post(
        f"{_DATALOADER_BASE}/fetch-ohlcvs",
        json=payload,
        headers=headers,
        timeout=_TIMEOUT,
    )
    return resp


def fetch_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Fetch adjusted daily OHLCV bars via POST /fetch-ohlcvs.

    Attempts Massive backend first; falls back to Dataloader's Yahoo backstop
    when the MASSIVE_API_KEY is absent or returns 'Unknown API Key', recording
    the deviation in ticker_details data_source fields.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols.
    start_date, end_date : str
        Inclusive date range (YYYY-MM-DD).
    adjust_dividends : bool
        Use dividend-adjusted close (default True; paper uses adjusted prices).
    identity_date : str or None
        As-of date for ticker identity resolution. Set to end_date if None.

    Returns
    -------
    prices : pd.DataFrame
        Adjusted closing prices, shape (trading_days, n_tickers).
        Index is pd.DatetimeIndex of trading dates.
    ticker_details : dict
        Per-ticker metadata including data_source and any warnings.
    """
    if identity_date is None:
        identity_date = end_date

    massive_key = os.environ.get("MASSIVE_API_KEY", "")

    # Primary: try with Massive source if key is available
    resp = None
    use_massive = bool(massive_key)
    if use_massive:
        payload = {
            "tickers": tickers,
            "timeframe": "1d",
            "start_date": start_date,
            "end_date": end_date,
            "source": "massive",
            "adjust_dividends": adjust_dividends,
            "yahoo_backstop": True,
            "identity_date": identity_date,
        }
        resp = _post_fetch_ohlcvs(payload, _get_headers(with_massive_key=True))
        if not resp.ok:
            error_body = resp.text
            # If Massive rejects our key, fall through to Yahoo-backstop path
            if resp.status_code >= 500 and (
                "Unknown API Key" in error_body or "API key" in error_body.lower()
            ):
                logger.warning(
                    "MASSIVE_API_KEY rejected by Massive backend; retrying via "
                    "Dataloader Yahoo backstop (data_source will be 'yahoo')."
                )
                use_massive = False
                resp = None
            else:
                _raise_for_dataloader(resp)

    # Fallback: Dataloader with Yahoo backstop (source not specified)
    if not use_massive or resp is None:
        payload = {
            "tickers": tickers,
            "timeframe": "1d",
            "start_date": start_date,
            "end_date": end_date,
            "adjust_dividends": adjust_dividends,
            "yahoo_backstop": True,
            "identity_date": identity_date,
        }
        resp = _post_fetch_ohlcvs(payload, _get_headers(with_massive_key=False))
        _raise_for_dataloader(resp)

    data = resp.json()
    result = data.get("result", {})
    td_list = result.get("ticker_details", [])
    ticker_details = {td["ticker"]: td for td in td_list} if td_list else {}

    # Log ticker details: note non-Massive sources as deviations
    for ticker, td in ticker_details.items():
        src = td.get("data_source", "unknown")
        err = td.get("error")
        warn = td.get("warnings")
        if err:
            logger.warning("Ticker %s error: %s", ticker, err)
        if src != "massive":
            logger.info("Ticker %s: data_source=%s (deviation: not Massive)", ticker, src)
        if warn:
            logger.info("Ticker %s warnings: %s", ticker, warn)

    if result.get("data_stored", False):
        prices = _retrieve_stored(result, tickers, start_date, end_date,
                                  _get_headers(with_massive_key=False),
                                  ticker_details=ticker_details)
    else:
        ohlcv_df_data = result.get("ohlcv_df", {})
        rows = ohlcv_df_data.get("rows", [])
        if not rows:
            raise RuntimeError(
                f"Dataloader returned 0 bars for tickers={tickers[:5]}... "
                f"start={start_date} end={end_date}. Check API entitlement."
            )
        df_raw = pd.DataFrame(rows)
        prices = _parse_ohlcv_df(df_raw, tickers)

    return prices, ticker_details


def _raise_for_dataloader(resp: requests.Response) -> None:
    """Raise a descriptive error for non-OK Dataloader responses."""
    if resp.status_code == 401:
        raise EnvironmentError(
            "DATALOADER_API_KEY authentication failed. "
            "Check that DATALOADER_API_KEY is correct."
        )
    if resp.status_code == 422:
        raise ValueError(f"Invalid request parameters: {resp.text[:500]}")
    if not resp.ok:
        raise RuntimeError(
            f"Dataloader error {resp.status_code}: {resp.text[:400]}"
        )


def _retrieve_stored(
    result: dict,
    tickers: list[str],
    start_date: str,
    end_date: str,
    headers: dict,
    ticker_details: dict | None = None,
) -> pd.DataFrame:
    """Retrieve bars for a stored (large) Dataloader response.

    Individual-ticker parquets from ``save-dataframe`` use a RangeIndex (no
    date column).  The API returns the *full-batch* parquet — which does carry a
    ``date`` column — when the requested ticker is absent from the stored
    result.  We exploit this: probe with any error ticker (or a dummy) to get
    the full-batch parquet and extract every ticker from it in one shot.
    """
    result_id = result["result_id"]
    ticker_details = ticker_details or {}

    # Build a list of probe tickers: error tickers first (they trigger the
    # full-batch fallback), then a placeholder that is guaranteed not to exist.
    error_tickers = [t for t, td in ticker_details.items() if td.get("error")]
    probe_candidates = error_tickers + ["__PROBE_FULLBATCH__"]

    for probe in probe_candidates:
        try:
            raw_df = _save_and_download_raw(result_id, probe, start_date, end_date, headers)
            if raw_df is None:
                continue
            if "date" in raw_df.columns:
                # Full-batch parquet — extract all tickers at once.
                logger.info(
                    "Full-batch parquet obtained via probe='%s'; "
                    "extracting %d tickers.", probe, len(tickers)
                )
                return _extract_from_batch_parquet(raw_df, tickers)
        except Exception as exc:
            logger.debug("Probe '%s' failed: %s", probe, exc)
            continue

    # Fallback: per-ticker saves (dates may be approximate if parquet has
    # RangeIndex — this path is only reached when no probe succeeded).
    logger.warning(
        "Full-batch probe did not succeed; falling back to per-ticker saves. "
        "Date alignment may be approximate (business-day estimate)."
    )
    dfs = []
    failed = []
    for ticker in tickers:
        try:
            df_t = _fetch_one_stored(result_id, ticker, start_date, end_date, headers)
            if df_t is not None:
                dfs.append(df_t)
        except Exception as exc:
            logger.warning("Failed to retrieve stored data for %s: %s", ticker, exc)
            failed.append(ticker)
    if not dfs:
        raise RuntimeError(
            f"No data retrieved from stored response. Failed tickers: {failed}"
        )
    if failed:
        logger.warning("Tickers not retrieved: %s", failed)
    return pd.concat(dfs, axis=1)


def _save_and_download_raw(
    result_id: str,
    ticker: str,
    start_date: str,
    end_date: str,
    headers: dict,
) -> pd.DataFrame | None:
    """Call save-dataframe for *ticker* and download the resulting parquet.

    Returns the raw DataFrame (with whatever index/columns the API provided),
    or None when the ticker returns a 422 (not in stored result).
    """
    save_payload = {
        "result_id": result_id,
        "ticker": ticker,
        "dataset_name": f"dl_{ticker.replace('/', '_')}",
        "timeframe": "1d",
        "start_date": start_date,
        "end_date": end_date,
        "file_format": "parquet",
        "data_kind": "ohlcv",
    }
    resp = requests.post(
        f"{_DATALOADER_BASE}/save-dataframe",
        json=save_payload,
        headers=headers,
        timeout=_TIMEOUT,
    )
    if resp.status_code == 422:
        return None
    resp.raise_for_status()
    artifact_id = resp.json()["result"]["artifact_id"]

    resp2 = requests.get(
        f"{_DATALOADER_BASE}/artifacts/{artifact_id}/download",
        headers=headers,
        timeout=_TIMEOUT,
    )
    resp2.raise_for_status()
    download_url = resp2.json()["result"]["url"]

    for attempt in range(3):
        try:
            r = requests.get(download_url, timeout=_TIMEOUT)
            r.raise_for_status()
            return pd.read_parquet(io.BytesIO(r.content))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return None  # unreachable


def _extract_from_batch_parquet(raw_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract closing prices for each ticker from a full-batch parquet.

    The full-batch parquet has columns ``date``, ``{ticker}_open``, …,
    ``{ticker}_close``, … for every ticker that had data.  Date values are
    strings (YYYY-MM-DD) or already datetime objects.
    """
    dates = pd.to_datetime(raw_df["date"], utc=True).dt.tz_convert(None)
    result_dfs: list[pd.DataFrame] = []
    for ticker in tickers:
        close_col = f"{ticker.lower()}_close"
        if close_col in raw_df.columns:
            s = pd.Series(raw_df[close_col].values, index=dates, name=ticker, dtype=float)
            result_dfs.append(s.to_frame())
        else:
            logger.warning("Ticker %s: '%s' not found in batch parquet — skipped.", ticker, close_col)
    if not result_dfs:
        raise RuntimeError("No tickers found in full-batch parquet.")
    combined = pd.concat(result_dfs, axis=1)
    combined.index.name = "date"
    return combined


def _fetch_one_stored(
    result_id: str,
    ticker: str,
    start_date: str,
    end_date: str,
    headers: dict,
) -> pd.DataFrame | None:
    """Save and download one ticker from a stored Dataloader result."""
    # Step 1: save-dataframe
    save_payload = {
        "result_id": result_id,
        "ticker": ticker,
        "dataset_name": f"dl_{ticker}",
        "timeframe": "1d",
        "start_date": start_date,
        "end_date": end_date,
        "file_format": "parquet",
        "data_kind": "ohlcv",
    }
    resp = requests.post(
        f"{_DATALOADER_BASE}/save-dataframe",
        json=save_payload,
        headers=headers,
        timeout=_TIMEOUT,
    )
    if resp.status_code == 422:
        logger.warning("Ticker %s not in stored result (422): %s", ticker, resp.text[:200])
        return None
    resp.raise_for_status()
    artifact_id = resp.json()["result"]["artifact_id"]

    # Step 2: get download URL
    resp2 = requests.get(
        f"{_DATALOADER_BASE}/artifacts/{artifact_id}/download",
        headers=headers,
        timeout=_TIMEOUT,
    )
    resp2.raise_for_status()
    download_url = resp2.json()["result"]["url"]

    # Step 3: download parquet (no auth header on the pre-signed URL)
    for attempt in range(3):
        try:
            r = requests.get(download_url, timeout=_TIMEOUT)
            r.raise_for_status()
            buf = io.BytesIO(r.content)
            df = pd.read_parquet(buf)
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    # Extract close column — stored response may use ticker-prefixed names (e.g. "crm_close")
    if "close" not in df.columns:
        prefixed = f"{ticker.lower()}_close"
        if prefixed in df.columns:
            df = df.rename(columns={prefixed: "close"})
        else:
            logger.warning(
                "Ticker %s: 'close' column missing, available: %s", ticker, list(df.columns)
            )
            return None

    # Build date index — normalize to tz-naive UTC so concat aligns with inline path
    date_col = next((c for c in ("date", "Date", "timestamp") if c in df.columns), None)
    if date_col:
        df = df.set_index(date_col)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)  # tz-naive, matching inline path
    return df[["close"]].rename(columns={"close": ticker})


def _parse_ohlcv_df(df_raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """
    Parse the inline ohlcv_df from Dataloader response into a closing-prices DataFrame.

    Handles two response layouts:
    - Wide format: columns ['Date', 'AAPL_close', 'MSFT_close', ...] — one row per date.
    - Tall format: columns ['date', 'ticker', 'close', ...] — one row per (date, ticker).
    """
    cols_lower = {c.lower(): c for c in df_raw.columns}

    # Determine date column (case-insensitive)
    date_col_original = cols_lower.get("date")
    if date_col_original is None:
        raise RuntimeError(f"No 'Date' column found in ohlcv_df. Columns: {list(df_raw.columns)}")
    df_raw = df_raw.rename(columns={date_col_original: "_date"})
    df_raw["_date"] = pd.to_datetime(df_raw["_date"], utc=True).dt.tz_localize(None)

    price_dfs = []

    # Wide format: 'TICKER_close' columns (Dataloader multi-ticker response)
    if "ticker" not in cols_lower:
        for ticker in tickers:
            close_col = f"{ticker}_close"
            if close_col not in df_raw.columns:
                logger.warning("Ticker %s: column '%s' missing in wide-format response", ticker, close_col)
                continue
            sub = df_raw[["_date", close_col]].set_index("_date")
            sub.index.name = "date"
            price_dfs.append(sub.rename(columns={close_col: ticker}))
    else:
        # Tall format: 'ticker' + 'close' columns
        ticker_col = cols_lower["ticker"]
        close_col = cols_lower.get("close", "close")
        df_raw = df_raw.rename(columns={ticker_col: "_ticker", close_col: "_close"})
        for ticker in tickers:
            sub = df_raw[df_raw["_ticker"] == ticker].set_index("_date")[["_close"]]
            sub.index.name = "date"
            if sub.empty:
                logger.warning("Ticker %s: no rows in tall-format response", ticker)
                continue
            price_dfs.append(sub.rename(columns={"_close": ticker}))

    if not price_dfs:
        raise RuntimeError("No valid price series found in inline ohlcv_df.")

    prices = pd.concat(price_dfs, axis=1).sort_index()
    logger.info(
        "Loaded %d tickers × %d rows (%s to %s)",
        prices.shape[1],
        prices.shape[0],
        prices.index[0].date(),
        prices.index[-1].date(),
    )
    return prices


def load_prices_batch(
    tickers: list[str],
    start_date: str,
    end_date: str,
    batch_size: int = 50,
    adjust_dividends: bool = True,
    identity_date: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load prices for a large universe in batches.

    Parameters
    ----------
    tickers : list[str]
    start_date, end_date : str
    batch_size : int
        Tickers per batch (avoid oversized payloads).
    adjust_dividends : bool
    identity_date : str or None

    Returns
    -------
    prices : pd.DataFrame
        Aligned adjusted closing prices.
    all_ticker_details : dict
    """
    all_dfs: list[pd.DataFrame] = []
    all_details: dict = {}
    n = len(tickers)
    batches = [tickers[i : i + batch_size] for i in range(0, n, batch_size)]
    logger.info("Loading %d tickers in %d batches of up to %d", n, len(batches), batch_size)

    for b_idx, batch in enumerate(batches):
        logger.info("Batch %d/%d: %s ... (%d tickers)", b_idx + 1, len(batches), batch[0], len(batch))
        try:
            df_batch, td_batch = fetch_ohlcv(
                batch, start_date, end_date,
                adjust_dividends=adjust_dividends,
                identity_date=identity_date,
            )
            all_dfs.append(df_batch)
            all_details.update(td_batch)
        except Exception as exc:
            logger.error("Batch %d failed: %s", b_idx + 1, exc)
            # Continue with remaining batches; failed tickers are absent from result

    if not all_dfs:
        raise RuntimeError("All batches failed — no price data loaded.")

    prices = pd.concat(all_dfs, axis=1)
    prices = prices.sort_index()
    logger.info("Combined price matrix: %d dates × %d tickers", *prices.shape)
    return prices, all_details


def compute_returns(prices: pd.DataFrame, min_obs: int = 0) -> pd.DataFrame:
    """
    Compute simple daily returns y_i(t) = (S_i(t+1) - S_i(t)) / S_i(t)
    from adjusted closing prices (Section 2).

    Parameters
    ----------
    prices : pd.DataFrame
        Closing prices.
    min_obs : int
        Drop tickers with fewer than this many valid observations.

    Returns
    -------
    returns : pd.DataFrame
        Daily simple returns (one fewer row than prices).
    """
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if min_obs > 0:
        valid = returns.count() >= min_obs
        dropped = valid[~valid].index.tolist()
        if dropped:
            logger.warning("Dropping %d tickers with < %d observations: %s", len(dropped), min_obs, dropped[:10])
        returns = returns.loc[:, valid]
    return returns


def drop_missing_dates(
    returns: pd.DataFrame,
    max_missing_frac: float = 0.0,
) -> pd.DataFrame:
    """
    Remove dates where any ticker has a missing return.

    Parameters
    ----------
    returns : pd.DataFrame
    max_missing_frac : float
        If 0.0, keep only dates with no missing values.

    Returns
    -------
    clean : pd.DataFrame
    """
    if max_missing_frac == 0.0:
        clean = returns.dropna()
    else:
        thresh = int(np.ceil(returns.shape[1] * (1.0 - max_missing_frac)))
        clean = returns.dropna(thresh=thresh)
    n_dropped = len(returns) - len(clean)
    if n_dropped:
        logger.info("Dropped %d dates with missing values", n_dropped)
    return clean
