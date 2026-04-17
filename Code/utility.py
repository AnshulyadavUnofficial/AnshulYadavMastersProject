import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone, date
from enum import Enum
from typing import List
from pathlib import Path
from functools import total_ordering
from collections import OrderedDict

# ============================
# Hyperparameters
# ============================

# --- Rolling Window Setup ---
train_start_date = datetime(2025, 1, 1)     # Global anchor for the first fold
train_days = 60                          # Fixed lookback for training (regime window)
step_days  = 10                           # The "Atomic" unit: both val and test are these many days.

# Sampling horizon and offset
Horizon = 30 # interval length in minutes; possible values: 30, 39, 65, 78, 130, 195, 390
Offset  = 0   # offset in minutes from market start

# Windowing parameters
WindowSize = 7   # lookback length (W) along with horizon span more than 1 day, W*H > 2, to capture day volatility movement. 
PredStep   = 1   # prediction step ahead, in terms of number of horizons.

# Model training parameters
dropout   = 0.3
batchsize = 32
eval_batchsize = 32
num_layers = 2
layer_dim = 64
l2_norm = 5e-3          # strong regularization since snr is low
device = "cuda"
epochs = 500
iters = 200
patience = int(0.02 * epochs) # patience for early stopping
burn_in_epochs = int(0.1*epochs) # to prevent the first fold from terminating too early because of early stopping
lr_start = 1e-3
lr_end = 8e-5
markowitz_reg = 0.01

# loss coefficients
drift_loss_coeff     = 1
return_loss_coeff   = 1000        # return loss is much greater than ic loss so use a smaller value

# Index constants
IDX_LOGRET_CLOSE = 0
IDX_DAY_COUNTER = 6

feature_indicies = {
    'logret_close':IDX_LOGRET_CLOSE,
    'logret_high':1,
    'logret_open':2,
    'logret_low':3,
}

# Input to the backbone network 
class BackboneInputType(Enum):
    revol_only = 1
    bypass_only = 2
    hybrid = 3
# Model Output Type
class ModelOutputType(Enum):
    logreturn = 1
    rank = 2
    effr_vol = 3

backbone_input_type = BackboneInputType.revol_only
model_output_type = ModelOutputType.logreturn

# =============================================================
# Globals 
# These values are related to data collection and stock and etfs
# they are not changed. 
# =============================================================

DATA_ROOT = Path(__file__).parent.parent / 'Raw_Data'
DATA_COLLECTION_START_DATE = datetime(2025, 1, 1).date()  # January 1 2025
date_format = "%Y-%m-%d"
time_format = '%H:%M'
datetime_format = f'{date_format} {time_format}'
yf_intervals = ['1m','5m','15m']
date_file = 'date_file.txt'
EPS = 1e-6              # a small +ve constant to prevent divison by 0

DJ30_TICKERS = ['AMZN','AAPL','AXP','AMGN','BA','CAT','CSCO', 'CVX', 'GS', 'HD', 
    'HON', 'IBM', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 
    'NKE', 'NVDA', 'PG', 'CRM', 'SHW', 'TRV', 'UNH', 'V', 'VZ', 'WMT', 'DIS'
]
ETF_TICKERS = ['SPY', 'QQQ', 'DIA', 'IWM', 'XLK', 'XLF', 'XLV', 'XLI', 'VIXY', 'IEF', 'GLD']
ALL_TICKERS = DJ30_TICKERS + ETF_TICKERS
NUM_STOCKS = len(DJ30_TICKERS)


class Market(Enum):
    US = "US_Intraday_Data"
    # future extensions
    Crypto = None
    UK = None
    AU = None
    Forex = None
    

# Holiday sets
us_holidays = {
    datetime(2025, 1, 1).date(), datetime(2025, 1, 9).date(), datetime(2025, 1, 20).date(),
    datetime(2025, 2, 17).date(), datetime(2025, 4, 18).date(),
    datetime(2025, 5, 26).date(), datetime(2025, 6, 19).date(),
    datetime(2025, 7, 4).date(), datetime(2025, 9, 1).date(),
    datetime(2025, 11, 27).date(), datetime(2025, 12, 25).date(),
    datetime(2025, 7, 3).date(), datetime(2025, 11, 28).date(),
    datetime(2025, 12, 24).date(),

    datetime(2026, 1, 1).date(), datetime(2026, 1, 19).date(), datetime(2026, 2, 2).date(), # Feb 02 was a holiday because yahoo's server did not return data
    datetime(2026, 2, 16).date(), datetime(2026, 4, 3).date(),
    datetime(2026, 5, 25).date(), datetime(2026, 6, 19).date(),
    datetime(2026, 7, 3).date(), datetime(2026, 9, 7).date(),
    datetime(2026, 11, 26).date(), datetime(2026, 11, 27).date(),
    datetime(2026, 12, 24).date(), datetime(2026, 12, 25).date(),
}

holiday_map= {Market.US: us_holidays}

market_times_map = {
    Market.US:     {'open': '09:30', 'close': '15:59', 'tz': 'America/New_York', 'mins': 390},
    Market.UK:     {'open': '08:00', 'close': '16:29', 'tz': 'Europe/London', 'mins': 510},
    Market.AU:     {'open': '10:10', 'close': '15:59', 'tz': 'Australia/Sydney', 'mins': 350},
    Market.Crypto: {'open': '00:00', 'close': '23:59', 'tz': 'UTC', 'mins': 1440},
    Market.Forex : {'open': '00:00', 'close': '23:59', 'tz': 'UTC', 'mins': 1440},
}

@total_ordering
class IntervalDirs(Enum):
    ONE_MIN     = "1m"
    FIVE_MIN    = "5m"
    FIFTEEN_MIN = "15m"
    ONE_HOUR    = "1h"

    @property
    def minutes(self):
        mapping = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
        return mapping[self.value]

    def __lt__(self, other):
        if isinstance(other, IntervalDirs):
            return self.minutes < other.minutes
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, IntervalDirs):
            return self.minutes == other.minutes
        return NotImplemented
    
    @classmethod
    def from_minutes(cls, mins: int):
        """Return the IntervalDirs member corresponding to a minute count."""
        for iv in cls:
            if iv.minutes == mins:
                return iv
        raise ValueError(f"No IntervalDirs member for {mins} minutes")


def get_valid_dates(market: Market) -> List[date]:
    """
    Return all valid trading dates for the given market.

    - Start: Jan 1, 2025
    - End: 2 days before today (inclusive)
    - Exclude weekends and holidays (except Crypto, which has all days)
    - Dates returned as a list of `datetime.date` objects in UTC

    Parameters
    ----------
    market : Market
        Enum for the market (US, UK, AU, Crypto).

    Returns
    -------
    List[date]
        A list of valid trading dates as `datetime.date` objects.
    """
    global DATA_COLLECTION_START_DATE, holiday_map
    end_date = datetime(year=2026, month=4, day=6, hour=0, minute=0).date()

    holidays = holiday_map[market]
    valid_dates: List[date] = []

    current_date = DATA_COLLECTION_START_DATE
    while current_date <= end_date:
        if market == Market.Crypto:
            valid_dates.append(current_date)
        else:
            if current_date.weekday() < 5 and current_date not in holidays:
                valid_dates.append(current_date)
        current_date += timedelta(days=1)

    return valid_dates

def _generate_canonical_intervals(market: Market):
    """Generate canonical 1m intervals (start,end) as datetime objects for a given market."""
    global time_format, market_times_map
    times = market_times_map[market]
    open_time = datetime.strptime(times['open'], time_format)
    close_time = datetime.strptime(times['close'], time_format)

    intervals = []
    current = open_time
    while current <= close_time:
        start = current
        end = current + timedelta(minutes=1)
        intervals.append((start, end)) 
        current += timedelta(minutes=1)
    
    assert len(intervals) == market_times_map[market]['mins']
    return intervals

def _normalize_intervals(df: pd.DataFrame, minutes: int):
    """
    Given a dataframe with a 'Time' column (HH:MM),
    return a set of (start,end) datetime intervals of length `minutes`.
    """
    global time_format
    normalized = set()
    for t in df['Time'].astype(str):
        start = datetime.strptime(t, time_format)
        end = start + timedelta(minutes=minutes)
        normalized.add((start, end))
    return normalized

def check_interval_coverage(market: Market):
    """
    For each ticker in the given market, check if every canonical 1m interval
    is covered by either a 1m bar, a 5m bar, or higher.
    """
    global DATA_ROOT, IntervalDirs
    valid_dates = get_valid_dates(market)
    canonical_intervals = _generate_canonical_intervals(market)

    market_dir = market.value
    market_path = DATA_ROOT / market_dir

    interval_dirs = [
        (IntervalDirs.ONE_MIN, 1),
        (IntervalDirs.FIVE_MIN, 5),
        # (IntervalDirs.ONE_HOUR, 60),
        # Future extension: (IntervalDirs.FIFTEEN_MIN, 15), etc.
    ]

    for ticker_dir in (market_path / IntervalDirs.ONE_MIN.value).glob('*'):
        ticker = ticker_dir.name
        for date in valid_dates:

            # Collect intervals from all granularities
            intervals_by_granularity = {}

            for interval_enum, minutes in interval_dirs:
                file_path = market_path / interval_enum.value / ticker / f"{date}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        intervals_by_granularity[minutes] = _normalize_intervals(df, minutes)

            # Check coverage
            missing = []
            for c_start, c_end in canonical_intervals:
                covered = False
                for minutes, intervals in intervals_by_granularity.items():
                    # containment works for 1m, 5m, 15m, etc.
                    if any(s <= c_start and e >= c_end for (s, e) in intervals):
                        covered = True
                        break
                if not covered:
                    missing.append((c_start, c_end))

            if missing:
                # print(f"{market.name}/{ticker}: Missing {[c_start.strftime('%H-%M') for c_start, c_end in missing]} intervals on {date}")
                print(f"{market.name}/{ticker}/{date}: {len(missing)}")

def check_database_integrity():
    """
    Walk through the entire Raw_Data database.
    For each CSV file, check if it is empty or contains NaNs.
    Print the file path and the condition if any issue is found.
    """
    global DATA_ROOT
    for csv_file in DATA_ROOT.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                print(f"{csv_file}: EMPTY file")
            elif df.isnull().values.any():
                print(f"{csv_file}: Contains NaNs")
        except Exception as e:
            print(f"{csv_file}: ERROR reading file ({e})")

def check_datefile_consistency():
    """
    For each ticker folder in the database:
    - Ensure all .csv files are listed in date_file.txt
    - Ensure date_file.txt does not list missing files
    - Ensure dates in date_file.txt are in ascending order
    Print any inconsistencies found.
    """
    global DATA_ROOT, date_file
    for datefile in DATA_ROOT.rglob(date_file):
        ticker_dir = datefile.parent
        ticker = ticker_dir.name

        # Collect actual CSV dates
        csv_dates = sorted([
            f.stem for f in ticker_dir.glob("*.csv")
            if f.name.endswith(".csv")
        ])

        # Read date_file.txt
        try:
            with open(datefile, "r") as f:
                listed_dates = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"{ticker_dir}: ERROR reading {datefile} ({e})")
            continue

        # Compare sets
        csv_set = set(csv_dates)
        listed_set = set(listed_dates)

        missing_in_txt = csv_set - listed_set
        missing_in_dir = listed_set - csv_set

        if missing_in_txt:
            print(f"{ticker_dir}: Dates missing in {datefile} → {sorted(missing_in_txt)}")
        if missing_in_dir:
            print(f"{ticker_dir}: Dates listed in {date_file} but no file exists → {sorted(missing_in_dir)}")

        # Check ordering
        if listed_dates != sorted(listed_dates):
            print(f"{ticker_dir}: Dates in {date_file} are not in ascending order")

""" Downloading Functions """

def _localize_and_trim(df: pd.DataFrame, market: Market) -> pd.DataFrame:
    """Localize to UTC, convert to market timezone, and trim to trading hours."""
    global market_times_map, time_format
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    tz = market_times_map[market]['tz']
    idx = idx.tz_convert(tz)
    df.index = idx

    open_t = market_times_map[market]['open']
    close_t = market_times_map[market]['close']
    df = df.between_time(open_t, close_t, inclusive="both")

    df.index = df.index.strftime(time_format)
    df.index.name = "Time"
    return df

def _update_datefile(ticker_dir: Path, date_str: str):
    """Update date_file.txt with the given date in sorted order.
    If the date_file.txt does not exist, create it and add the date.
    """
    global date_format, date_file
    datefile_path = ticker_dir / date_file
    curr_date = datetime.strptime(date_str, date_format)

    existing_dates = []
    if datefile_path.exists():
        with open(datefile_path, "r") as f:
            existing_dates = [datetime.strptime(line.strip(), date_format) for line in f if line.strip()]

        if curr_date not in existing_dates:
            existing_dates.append(curr_date)
            existing_dates.sort()
            with open(datefile_path, "w") as f:
                for d in existing_dates:
                    f.write(d.strftime(date_format) + "\n")
    else:
        # File does not exist → create and write the current date
        with open(datefile_path, "w") as f:
            f.write(curr_date.strftime(date_format) + "\n")

def download_and_store(ticker: str, date_str: str, market: Market):
    """
    Download intraday data for one ticker/date across all intervals.
    Store raw CSVs under root/market/interval/ticker, update per-interval date_file.txt,
    and print diagnostics (NaNs and canonical 1m coverage).
    """
    global yf_intervals, DATA_ROOT, date_format
    canonical_1m = _generate_canonical_intervals(market)

    # --- Adjust ticker only for yfinance call ---
    yf_ticker = ticker
    if market == Market.AU:
        yf_ticker = f"{ticker}.AX"
    elif market == Market.UK:
        yf_ticker = f"{ticker}.L"
    elif market == Market.Forex:
        yf_ticker = f"{ticker}=X"

    for interval in yf_intervals:
        try:
            # Directory: root/market/interval/ticker (raw ticker, not yf_ticker)
            ticker_dir = DATA_ROOT / market.value / interval / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)

            df = yf.download(
                yf_ticker,
                start=date_str,
                end=(datetime.strptime(date_str, date_format) + timedelta(days=1)).strftime(date_format),
                interval=interval,
                auto_adjust=True,
                prepost=True,
                progress=False
            )

            if df.empty:
                print(f"{market.name}/{interval}/{ticker}/{date_str}: DOWNLOAD FAILED (empty)")
                continue  # <-- skip this interval, try the next one

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = _localize_and_trim(df, market)

            # Save raw CSV for this interval
            file_path = ticker_dir / f"{date_str}.csv"
            df.to_csv(file_path, mode="w", header=True, index_label="Time")

            # Coverage check
            if interval.endswith("m"):
                minutes = int(interval[:-1])
            elif interval.endswith("h"):
                minutes = int(interval[:-1]) * 60
            else:
                raise ValueError(f'interval must end with m or h, but instead is {interval}')

            normalized = _normalize_intervals(df.reset_index(), minutes)
            missing = [c for c in canonical_1m if not any(s <= c[0] and e >= c[1] for (s, e) in normalized)]
            if missing:
                print(f"{market.name}/{interval}/{ticker}/{date_str}: Missing {len(missing)} canonical 1m slots")
            else:
                print(f"{market.name}/{interval}/{ticker}/{date_str}: FULL coverage")

            _update_datefile(ticker_dir, date_str)

        except Exception as e:
            print(f"{market.name}/{interval}/{ticker}/{date_str}: ERROR {e}")
            continue  # <-- also skip just this interval on error


""" DataSet Creating Functions"""

def _align_to_canonical(df, market: Market):
    global market_times_map, time_format
    times = market_times_map[market]

    # canonical bounds as strings
    open_dt  = datetime.strptime(times['open'],  time_format)
    close_dt = datetime.strptime(times['close'], time_format)
    extra_open_str  = (open_dt - timedelta(minutes=1)).strftime("%H:%M")
    extra_close_str = (close_dt + timedelta(minutes=1)).strftime("%H:%M")

    open_str  = open_dt.strftime("%H:%M")
    close_str = close_dt.strftime("%H:%M")

    # index is assumed to be strings like "HH:MM"
    hhmm = df.index.astype(str)

    # handle extra close bar
    if (close_str in hhmm) and (extra_close_str in hhmm):
        last_bar = df.loc[hhmm == close_str]
        extra_bar = df.loc[hhmm == extra_close_str]

        close_val = extra_bar["Close"].iloc[0]
        high_val  = max(last_bar["High"].iloc[0], extra_bar["High"].iloc[0])
        low_val   = min(last_bar["Low"].iloc[0],  extra_bar["Low"].iloc[0])
        vol_add   = float(extra_bar["Volume"].fillna(0.0).iloc[0])

        df.loc[last_bar.index, "Close"]  = close_val
        df.loc[last_bar.index, "High"]   = high_val
        df.loc[last_bar.index, "Low"]    = low_val
        df.loc[last_bar.index, "Volume"] = df.loc[last_bar.index, "Volume"] + vol_add

    # handle extra open bar
    if (open_str in hhmm) and (extra_open_str in hhmm):
        first_bar = df.loc[hhmm == open_str]
        extra_bar = df.loc[hhmm == extra_open_str]

        open_val  = extra_bar["Open"].iloc[0]
        high_val  = max(first_bar["High"].iloc[0], extra_bar["High"].iloc[0])
        low_val   = min(first_bar["Low"].iloc[0],  extra_bar["Low"].iloc[0])
        vol_add   = float(extra_bar["Volume"].fillna(0.0).iloc[0])

        df.loc[first_bar.index, "Open"]   = open_val
        df.loc[first_bar.index, "High"]   = high_val
        df.loc[first_bar.index, "Low"]    = low_val
        df.loc[first_bar.index, "Volume"] = df.loc[first_bar.index, "Volume"] + vol_add

    # restrict explicitly to canonical window
    df = df.loc[(hhmm >= open_str) & (hhmm <= close_str)]
    return df

def _build_canonical_horizons(dfs_for_date: OrderedDict, market: Market, horizon: int, offset: int):
    global market_times_map
    
    times = market_times_map[market]
    open_dt  = datetime.strptime(times['open'], time_format)
    close_dt = datetime.strptime(times['close'], time_format)
    total_mins = times['mins']

    ch_dict = OrderedDict()

    # --- Pre-offset CH (open → open+offset) ---
    if offset > 0:
        pre_close = open_dt + timedelta(minutes=offset)
        ch_key = (open_dt.strftime("%H:%M"), pre_close.strftime("%H:%M"))
        ch_dict[ch_key] = []

    # --- Regular CHs ---
    num_slots = (total_mins - offset) // horizon
    for k in range(num_slots):
        ch_start_dt = open_dt + timedelta(minutes=offset + k * horizon)
        ch_close_dt = ch_start_dt + timedelta(minutes=horizon)
        ch_key = (ch_start_dt.strftime("%H:%M"), ch_close_dt.strftime("%H:%M"))
        ch_dict[ch_key] = []

    # --- Post-last CH (last close → market close) ---
    last_close = open_dt + timedelta(minutes=offset + num_slots * horizon)
    if last_close < close_dt:
        ch_key = (last_close.strftime("%H:%M"), close_dt.strftime("%H:%M"))
        ch_dict[ch_key] = []

    # --- Canonical 1m intervals (gaps) ---
    c1m_intervals = _generate_canonical_intervals(market)
    c1m_intervals = [(s.strftime("%H:%M"), e.strftime("%H:%M")) for s, e in c1m_intervals]

    # --- Convert dfs_for_date into (iv_minutes, [(start, close), ...]) ---
    dfs_times = []
    for iv, df in dfs_for_date.items():
        iv_minutes = IntervalDirs(iv).minutes
        idx_str = df.index.sort_values().astype(str)
        intervals = []
        for t in idx_str:
            start_dt = datetime.strptime(t, "%H:%M")
            close_dt = start_dt + timedelta(minutes=iv_minutes)
            intervals.append((start_dt.strftime("%H:%M"), close_dt.strftime("%H:%M")))
        dfs_times.append((iv_minutes, intervals))

    # --- Assign intervals to CHs via C1m gaps ---
    for ch_key in ch_dict.keys():
        ch_start, ch_close = ch_key
        gaps = [(s, e) for (s, e) in c1m_intervals if s >= ch_start and e <= ch_close]
        assigned = []

        for iv_minutes, intervals in dfs_times:
            if iv_minutes > horizon:
                break

            for (start, close) in intervals:
                covered = [(s, e) for (s, e) in gaps if s >= start and e <= close]
                if covered:
                    assigned.append((start, close, iv_minutes))
                    gaps = [g for g in gaps if g not in covered]

        # resolve overlaps: prefer larger intervals but keep maximum coverage
        if assigned:
            assigned.sort(key=lambda x: x[2], reverse=True)
            kept = []
            covered_gaps = set()

            for (s, c, iv_minutes) in assigned:
                interval_gaps = [(gs, ge) for (gs, ge) in c1m_intervals if gs >= s and ge <= c]
                new_gaps = [g for g in interval_gaps if g not in covered_gaps]
                if new_gaps:
                    iv_enum = IntervalDirs.from_minutes(iv_minutes)
                    kept.append((s, c, iv_enum.value))  # store "1m", "5m", etc.
                    covered_gaps.update(new_gaps)

            # order kept intervals by opening time
            kept.sort(key=lambda x: x[0])
            ch_dict[ch_key] = kept

    return ch_dict

def rs_vol_from_ohlc(o, h, l, c):
    """Rogers–Satchell variance for one bar (no sqrt)."""
    global EPS
    return max(np.sqrt(np.log(h / o) * np.log(h / c) + np.log(l / o) *np.log(l / c)), EPS)

def build_stock_dataset(stock, market: Market, horizon: int, offset: int):
    global DATA_ROOT, date_file, date_format, datetime_format, market_times_map

    valid_dates = get_valid_dates(market)

    # read all the datefiles corresponding to the intervals for this ticker
    market_path = DATA_ROOT / market.value
    datefiles = {}
    for iv in IntervalDirs:
        path = market_path / iv.value / stock / date_file
        if path.exists():
            with open(path, 'r') as file:
                dates = [line.strip() for line in file]
            datefiles[iv.value] = dates

    # invert mapping
    date_to_intervals = {}
    for interval, dates in datefiles.items():
        for d in dates:
            date_to_intervals.setdefault(d, []).append(interval)

    # enforce canonical order for intervals
    for d, iv_list in date_to_intervals.items():
        date_to_intervals[d] = sorted(iv_list, key=lambda iv: IntervalDirs(iv))

    # process dates
    dates_to_process = [
        d.strftime(date_format)
        for d in valid_dates
        if d.strftime(date_format) in date_to_intervals
    ]

    rows = []
    prev = None
    market_open_dt = datetime.strptime(market_times_map[market]['open'], time_format)
    total_market_minutes = market_times_map[market]['mins']

    prev_date = None

    for d in dates_to_process:
        iv_list = date_to_intervals[d]
        dfs_for_date = OrderedDict()
        for iv in iv_list:
            csv_path = market_path / iv / stock / f"{d}.csv"
            df = pd.read_csv(csv_path, index_col=0, header=0)

            # cleaning
            df = df.dropna(how="all")
            ohlc_present = df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
            vol_missing = df["Volume"].isna()
            df.loc[ohlc_present & vol_missing, "Volume"] = 0.0
            if df.isna().any().any():
                print(f"NaNs remain in file: {csv_path}")
            df = _align_to_canonical(df, market)
            if len(df) == 0:
                print(f"{csv_path.relative_to(market_path).stem}[{iv}]", end=',')
                continue

            dfs_for_date[iv] = df

        # --- Build canonical horizons for this date ---
        ch_dict = _build_canonical_horizons(dfs_for_date, market, horizon, offset)
        num_intervals = len(ch_dict)
        for i, (ch_key, intervals) in enumerate(ch_dict.items()):
            if not intervals:
                continue

            slices = [dfs_for_date[iv_str].loc[s] for (s, c, iv_str) in intervals]
            window = pd.DataFrame(slices)
            t0_ts = datetime.strptime(f"{d} {ch_key[0]}", datetime_format)

            # convert ch_start string to minutes since market open
            ch_start_dt = datetime.strptime(ch_key[0], "%H:%M")
            ch_start_minutes = int((ch_start_dt - market_open_dt).total_seconds() // 60)

            if i == 0 and offset != 0:
                if prev is None:
                    continue
                O = prev['open']
                C = window.iloc[-1]['Close']
                H = max(prev['high'], window['High'].max())
                L = min(prev['low'], window['Low'].min())

                cumul_change = np.log(C / O)
                rs_vol_across_days = rs_vol_from_ohlc(O, H, L, C)
                park_vol = np.log(H/L)/np.sqrt(4*np.log(2))

                merged_row = {
                    't0_ts': prev['t0_ts'],
                    'ch_start': prev['ch_start'] / total_market_minutes,
                    'coverage': (prev['interval_len'] + sum(IntervalDirs(iv_str).minutes for (_, _, iv_str) in intervals))/horizon,
                    'day_flag':1,       # next day
                    # 'horizon_norm': horizon / total_market_minutes,

                    
                    'open': O,
                    'close': C,
                    'high': H,
                    'low': L,
                    'volume': prev['volume'] + window['Volume'].sum(),
                    
                    # 'raw_snr': cumul_change / rs_vol_across_days,
                    # 'raw_effr': cumul_change/park_vol,
                    'rs_vol':rs_vol_across_days,
                    'park_vol':park_vol,

                    'prev_date': prev_date,
                }
                rows.append(merged_row)
            else:
                O = window.iloc[0]['Open']
                C = window.iloc[-1]['Close']
                H = window['High'].max()
                L = window['Low'].min()

                cumul_change = np.log(C / O)
                rs_vol_intra = rs_vol_from_ohlc(O, H, L, C)
                park_vol = np.log(H/L)/np.sqrt(4*np.log(2))
                row = {
                    # internal field
                    'interval_len':  sum(IntervalDirs(iv_str).minutes for (_, _, iv_str) in intervals),

                    # features
                    't0_ts': t0_ts,
                    'ch_start': ch_start_minutes / total_market_minutes,
                    'coverage': sum(IntervalDirs(iv_str).minutes for (_, _, iv_str) in intervals)/horizon,
                    'day_flag':0,       # same day
                    # 'horizon_norm': horizon / total_market_minutes,
                    
                    'open': O,
                    'close': C,
                    'high': H,
                    'low': L,
                    'volume': window['Volume'].sum(),

                    # 'raw_snr': cumul_change / rs_vol_intra,
                    # 'raw_effr': cumul_change/park_vol,
                    'rs_vol':rs_vol_intra,
                    'park_vol':park_vol,

                    # previous date
                    'prev_date': prev_date,
                }

                if offset != 0 and i == num_intervals - 1:
                    prev = row
                else:
                    prev = row
                    
                    # drop interval feature before appending
                    row_for_append = dict(row)   # make a shallow copy
                    row_for_append.pop('interval_len', None)
                    rows.append(row_for_append)
        prev_date = datetime.strptime(d, date_format).date()


    df_stock = pd.DataFrame(rows).set_index('t0_ts').sort_index()

    # 1. Create the Daily Ritual Anchor table
    # We use the current date to define what 'High/Low/Close' were for that specific day
    df_stock['current_date'] = df_stock.index.date
    daily_stats = df_stock.groupby('current_date').agg(
        prev_day_high=('high', 'max'),
        prev_day_low=('low', 'min'),
        prev_day_close=('close', 'last')
    )

    # 2. Map the 'Social Ritual' back to the main DataFrame
    # We merge daily_stats into df_stock using the 'prev_date' column
    # This effectively "shifts" the data to the correct next trading session
    df_stock = df_stock.merge(
        daily_stats, 
        left_on='prev_date', 
        right_index=True, 
        how='left'
    )
    df_stock['day_counter'] = pd.factorize(df_stock['current_date'])[0]

    # 3. Cleanup temp columns
    df_stock = df_stock.drop(columns=['current_date', 'prev_date'])


    if offset == 0: 
        # when offset = 0, day flag is not useful, since all horizons start and end on the same day
        df_stock = df_stock.drop(columns=['day_flag'])

    if market == Market.US:
        # for the US market coverage is close to 100%, so the feature is not useful. 
        df_stock = df_stock.drop(columns=['coverage'])

    return df_stock

def compute_intraday_features(raw_dfs):
    """
    Returns:
        updated_dfs: Scale-invariant features for model training.
        price_dfs: Raw close prices for PNL and weight calculation.
    """
    global WindowSize
    updated_dfs = {}
    price_dfs = {}
    epsilon = 1e-9

    for sym, df in raw_dfs.items():
        df_copy = df.copy()

        # 1. Store raw prices separately
        price_dfs[sym] = df_copy[["close"]].copy()

        # 2. Extract price components for calculation
        close  = df_copy["close"]
        open_  = df_copy["open"]
        high   = df_copy["high"]
        low    = df_copy["low"]
        volume = df_copy['volume']

        pd_h = df_copy['prev_day_high']
        pd_l = df_copy['prev_day_low']
        pd_c = df_copy['prev_day_close']

        # --- Volatility ---
        df_copy[f"log_rs_vol"] = np.log(df_copy[f"rs_vol"])
        df_copy[f"log_park_vol"] = np.log(df_copy[f"park_vol"])

        df_copy = df_copy.drop(columns = ['rs_vol', 'park_vol'])

        # --- Log Returns (Scale Invariant) ---
        df_copy["logret_close"] = np.log(close / close.shift(1))
        df_copy["logret_open"]  = np.log(open_ / close.shift(1))
        df_copy["logret_high"]  = np.log(high  / close.shift(1))
        df_copy["logret_low"]   = np.log(low   / close.shift(1))

        # --- Cyclical Time Encoding ---
        # Convert ch_start (0 to 1) into a continuous circle
        angle = 2 * np.pi * df_copy['ch_start']
        df_copy['time_sin'] = np.sin(angle)
        df_copy['time_cos'] = np.cos(angle)

        # --- Multiple Horizon Based Technical Indicators ---

        # 1. RSI 14 with Wilder's Smoothing normalized to [0,1]
        num_rsi_horizons = 14
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/num_rsi_horizons, min_periods=num_rsi_horizons, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/num_rsi_horizons, min_periods=num_rsi_horizons, adjust=False).mean()
        rs = avg_gain / (avg_loss + epsilon)
        df_copy[f"rsi_{num_rsi_horizons}"] = (100 - (100 / (1 + rs))) / 100

        # 2.  EMA 9 Log-distance from the close line
        num_ema_horizons = 9
        ema_9 = close.ewm(span=num_ema_horizons, adjust=False).mean()
        df_copy[f"log_ema_{num_ema_horizons}"] = np.log(close / ema_9)

        # 3. MACD 12, 26, 9 
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line_log = np.log(ema_12 / ema_26)
        signal_line = macd_line_log.ewm(span=9, adjust=False).mean()
        macd_hist_log = macd_line_log - signal_line

        df_copy["macd_line"] = macd_line_log
        df_copy["macd_hist"] = macd_hist_log

        # 4. Bollinger Bands (20, 2) 
        num_bb_horizons = 20
        std_dev_multiplier = 2
        sma_20 = close.rolling(window=num_bb_horizons).mean()
        std_20 = close.rolling(window=num_bb_horizons).std()

        upper_band = sma_20 + (std_dev_multiplier * std_20)
        lower_band = sma_20 - (std_dev_multiplier * std_20)

        df_copy["bb_pct"] = (close - lower_band) / (upper_band - lower_band + epsilon)
        df_copy["bb_width"] = np.log((upper_band - lower_band) / sma_20)

        # --- 6. VWAP (Institutional Anchor - Typical Price) ---
        typical_p = (high + low + close) / 3
        df_copy['pv'] = typical_p * volume
        daily_groups = df_copy.groupby(df_copy.index.date)
        vwap = daily_groups['pv'].cumsum() / (daily_groups['volume'].cumsum())
        df_copy["log_vwap_dist"] = np.log(close / vwap)
        df_copy = df_copy.drop(columns = ['pv']) 

        # --- Classic Pivot Points  ---
        P = (pd_h + pd_l + pd_c) / 3
        R1 = (2 * P) - pd_l
        S1 = (2 * P) - pd_h
        
        df_copy['log_classic_p_dist']  = np.log(close / P)
        df_copy['log_classic_r1_dist'] = np.log(close / R1)
        df_copy['log_classic_s1_dist'] = np.log(close / S1)

        # --- Camarilla Pivots (Intraday Mean Reversion/Breakout) ---
        pd_range = pd_h - pd_l
        df_copy['log_h3_dist'] = np.log(close / (pd_c + pd_range * 1.1 / 4))
        df_copy['log_h4_dist'] = np.log(close / (pd_c + pd_range * 1.1 / 2))
        df_copy['log_l3_dist'] = np.log(close / (pd_c - pd_range * 1.1 / 4))
        df_copy['log_l4_dist'] = np.log(close / (pd_c - pd_range * 1.1 / 2))

        # 3. DROP NON-SCALE INVARIANT & ANCHOR COLUMNS
        df_copy = df_copy.drop(columns=[
            "open", "high", "low", "close", "volume",
            "prev_day_high", "prev_day_low", "prev_day_close",
            "ch_start" # explicitly dropping the linear time feature now
        ])

        # 4. Final Reordering
        cols_first = [
            "logret_close", "logret_high", "logret_open", "logret_low", 
            "time_sin", "time_cos", "day_counter"
        ]
        cols_middle = [c for c in df_copy.columns if c not in cols_first]
        
        new_order = cols_first + cols_middle 
        df_copy = df_copy[new_order]

        updated_dfs[sym] = df_copy.add_prefix(f"{sym}_")

    return updated_dfs, price_dfs


""" Covariance Matrix Functions """


def interpolate_window_rs_bridge(window_df, num_sims=50, seed=10):
    rng = np.random.default_rng(seed)
    results = []

    for _, row in window_df.iterrows():
        O, H, L, C = row["open"], row["high"], row["low"], row["close"]
        T = int(row["minutes"])
        start_dt = datetime.strptime(row["start"], "%H:%M")
        
        # 1. Local Rogers-Satchell scaling
        vol_total = rs_vol_from_ohlc(O, H, L, C)
        sigma_1m = vol_total / np.sqrt(T) if T > 0 else 0
        L_O, L_C = np.log(O), np.log(C)
        target_ret = L_C - L_O

        if T == 1:
            best_returns = np.array([target_ret])
        else:
            best_error = float('inf')
            best_returns = None
            
            for _ in range(num_sims):
                dW = rng.normal(0, sigma_1m, T)
                W = np.cumsum(dW)
                t_idx = np.arange(1, T + 1) # T+1 since indexing is exclusive. 
                # Bridge: force the path to hit target_ret at index T
                bridge_log_path = W - (t_idx / T) * W[-1] + (t_idx / T) * target_ret
                
                # Selection logic: match observed H/L
                full_path_prices = np.exp(np.insert(bridge_log_path, 0, 0) + L_O)
                error = abs(np.max(full_path_prices) - H) + abs(np.min(full_path_prices) - L)
                
                if error < best_error:
                    best_error = error
                    best_returns = np.diff(np.insert(bridge_log_path, 0, 0))

        # 2. Map to 1-minute timestamps
        for m in range(T):
            current_time = (start_dt + timedelta(minutes=m)).strftime("%H:%M")
            results.append({
                "Time": current_time,
                "log_return": best_returns[m]
            })

    # Return as a clean DataFrame indexed by Time
    return pd.DataFrame(results).set_index("Time")

def build_covariance_matrix(stocks: List[str], market: Market, horizon: int, offset: int):
    global DATA_ROOT, date_file, date_format, datetime_format, market_times_map

    valid_dates = get_valid_dates(market)
    market_path = DATA_ROOT / market.value

    # --- Collect datefiles for each stock ---
    stock_datefiles = {}
    for stock in stocks:
        datefiles = {}
        for iv in IntervalDirs:
            path = market_path / iv.value / stock / date_file
            if path.exists():
                with open(path, 'r') as file:
                    dates = [line.strip() for line in file]
                datefiles[iv.value] = dates
        stock_datefiles[stock] = datefiles

    # --- Invert mapping: date -> intervals per stock ---
    stock_date_to_intervals = {}
    for stock, datefiles in stock_datefiles.items():
        date_to_intervals = {}
        for interval, dates in datefiles.items():
            for d in dates:
                date_to_intervals.setdefault(d, []).append(interval)
        # enforce canonical order
        for d, iv_list in date_to_intervals.items():
            date_to_intervals[d] = sorted(iv_list, key=lambda iv: IntervalDirs(iv))
        stock_date_to_intervals[stock] = date_to_intervals

    prev_day_last_row = None
    all_results = []
    # --- Iterate directly over valid dates ---
    for d in valid_dates:
        d_str = d.strftime(date_format)

        # For each stock, build dfs_for_date
        stock_dfs_for_date = {}
        for stock in stocks:
            dfs_for_date = OrderedDict()
            iv_list = stock_date_to_intervals[stock][d_str]
            for iv in iv_list:
                csv_path = market_path / iv / stock / f"{d_str}.csv"
                df = pd.read_csv(csv_path, index_col=0, header=0)

                # cleaning
                df = df.dropna(how="all")
                ohlc_present = df[["Open", "High", "Low", "Close"]].notna().all(axis=1)
                vol_missing = df["Volume"].isna()
                df.loc[ohlc_present & vol_missing, "Volume"] = 0.0
                if df.isna().any().any():
                    print(f"NaNs remain in file: {csv_path}")
                df = _align_to_canonical(df, market)
                if len(df) == 0:
                    print(f"{csv_path.relative_to(market_path).stem}", end=',')
                    continue

                dfs_for_date[iv] = df

            stock_dfs_for_date[stock] = dfs_for_date

        # --- Build canonical horizons for each stock ---
        ch_dicts = {}
        for stock, dfs_for_date in stock_dfs_for_date.items():
            if dfs_for_date:
                ch_dicts[stock] = _build_canonical_horizons(dfs_for_date, market, horizon, offset)

        # --- Combine into a DataFrame: rows = FULL datetime (t0_ts), cols = stocks ---
        # Generate the datetime index upfront to incorporate offset/date logic
        all_t0_ts = [datetime.strptime(f"{d_str} {ck[0]}", datetime_format) for ck in ch_dicts[stocks[0]].keys()]
        df_day = pd.DataFrame(index=all_t0_ts, columns=stocks)
        df_day.index.name = "t0_ts"

        for stock, ch_dict in ch_dicts.items():
            dfs_for_date = stock_dfs_for_date[stock]
            for ch_key, intervals in ch_dict.items():
                if not intervals:
                    continue

                slice_records = []
                for (s, c, iv_str) in intervals:
                    row = dfs_for_date[iv_str].loc[s]
                    slice_records.append({
                        "start": s,
                        "open": row["Open"],
                        "high": row["High"],
                        "low": row["Low"],
                        "close": row["Close"],
                        "minutes": IntervalDirs(iv_str).minutes
                    })

                window_df = pd.DataFrame(slice_records)
                interpolated_df = interpolate_window_rs_bridge(window_df)

                # Map the horizon start string back to the datetime index
                current_t0 = datetime.strptime(f"{d_str} {ch_key[0]}", datetime_format)
                df_day.at[current_t0, stock] = interpolated_df

        # --- OFFSET STRADDLE LOGIC ---
        if offset != 0:
            today_first_ts = df_day.index[0]

            if prev_day_last_row is None:
                # No "yesterday" tail to complete this horizon; discard it
                df_day = df_day.drop(index=today_first_ts)
            else:
                #  Merge the dataframes for each stock
                for stock in stocks:
                    yesterday_part = prev_day_last_row[stock]
                    today_part = df_day.at[today_first_ts, stock]
                    merged_df = pd.concat([yesterday_part, today_part])
                    df_day.at[today_first_ts, stock] = merged_df

                # Update the index of df_day to reflect the true start time (yesterday's t0)
                df_day = df_day.rename(index={today_first_ts: prev_day_last_row.name })

            # Store today's last row (the incomplete head) as the buffer for tomorrows
            prev_day_last_row = df_day.loc[df_day.index[-1]].copy()
            
            # Remove today's last row from current daily processing
            df_day = df_day.drop(index=df_day.index[-1])

        # ---  compute covariance ---
        day_results = []
        for t0_ts, row in df_day.iterrows():
            # Align the 1m log_returns for this timestamped row
            valid_dfs = [row[s]["log_return"].rename(s) for s in stocks if isinstance(row[s], pd.DataFrame)]
            horizon_returns = pd.concat(valid_dfs, axis=1, join="outer").reindex(columns=stocks)

            # Compute pairwise covariance and enforce stock ordering
            sigma_matrix = horizon_returns.cov(min_periods=2)

            # Convert variance estimate from 1 min to Horizon mins
            sigma_matrix *= horizon 
            day_results.append({"t0_ts": t0_ts,"sigma": sigma_matrix})

        all_results.extend(day_results)
        print(f'{d_str} completed')
    
    return pd.DataFrame(all_results).set_index("t0_ts")

def calculate_covariance_error(sigma_true, sigma_pred, k=5, tau=0.2):
    """
    Calculates a structural similarity score between two covariance matrices
    based on subspace alignment and variance magnitude.
    
    The metric decomposes the error into three main components:
    1. Subspace Alignment: How well the top 'k' eigenvectors of the prediction 
       align with the ground truth.
    2. Absolute Magnitude: How well the total variance of those top 'k' 
       components matches the truth.
    3. Concentration: Whether the model correctly captures the 'sharpness' 
       (eigenvalue decay) of the risk distribution.

    Args:
        sigma_true (ndarray): Ground truth covariance matrix (NxN).
        sigma_pred (ndarray): Predicted covariance matrix (NxN).
        k (int): Number of top principal components to evaluate. 
                 Default is 5.
        tau (float): Sensitivity parameter for the magnitude penalty. 
                     Lower values penalize scale mismatches more harshly.

    Returns:
        tuple: (final_score, alignment, p_mag, p_mag_norm, var_covered_t)
            - final_score: Alignment scaled by absolute magnitude penalty [0, 1].
            - alignment: Weighted cosine similarity of the subspaces.
            - p_mag: Penalty based on absolute variance scale.
            - p_mag_norm: Penalty based on relative concentration of variance.
            - var_covered_t: Percentage of total variance captured by top k in truth.
    """
    # 1. Eigen-decomposition: Extracting the principal axes of risk
    vals_t, vecs_t = np.linalg.eigh(sigma_true)
    vals_p, vecs_p = np.linalg.eigh(sigma_pred)
    
    # Sort descending to focus on the largest sources of variance
    idx_t = np.argsort(vals_t)[::-1]
    vals_t, vecs_t = vals_t[idx_t], vecs_t[:, idx_t]
    
    idx_p = np.argsort(vals_p)[::-1]
    vals_p, vecs_p = vals_p[idx_p], vecs_p[:, idx_p]
    
    k = min(k, len(vals_t))
    Vk, Vk_hat = vecs_t[:, :k], vecs_p[:, :k]
    
    # 2. Variance Coverage Logic: Assessing 'Truth' vs 'Predicted' concentration
    sum_k_t = np.sum(vals_t[:k])
    total_var_t = np.sum(vals_t)
    var_covered_t = sum_k_t / total_var_t # How dominant are the top k factors?
    
    sum_k_p = np.sum(vals_p[:k])
    total_var_p = np.sum(vals_p)
    var_covered_p = sum_k_p / total_var_p 
    
    # 3. Subspace Alignment: Weighted overlap of the top k eigenvectors
    # Uses SVD on the projection matrix to find the principal angles
    M = Vk.T @ Vk_hat
    _, sigmas, _ = np.linalg.svd(np.clip(M, -1.0, 1.0))
    # Weight the alignment by the importance (eigenvalues) of the truth components
    s_sub = np.sum(vals_t[:k] * sigmas[:k]) / sum_k_t
    
    # 4. Magnitude Penalty (Absolute Scale): Penalizes over/under estimation of risk
    rk = sum_k_p / sum_k_t
    p_mag = np.exp(-(np.log(rk)**2) / (2 * tau**2))
    
    # 5. Magnitude Penalty (Normalized Concentration): 
    # Measures if the 'ratio' of systemic risk to idiosyncratic risk is correct
    rk_norm = var_covered_p / var_covered_t
    p_mag_norm = np.exp(-(np.log(rk_norm)**2) / (2 * tau**2))
    
    # Returns: Score (Integrated), Alignment, Abs Magnitude, Concentration Magnitude, Coverage
    return s_sub * p_mag, s_sub, p_mag, p_mag_norm, var_covered_t
