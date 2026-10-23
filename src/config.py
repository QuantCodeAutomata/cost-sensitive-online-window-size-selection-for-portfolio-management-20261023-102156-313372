"""
Shared configuration constants derived from the paper.

Paper: Cost-Sensitive Online Window Size Selection for Portfolio Management
Authors: Yi-Chen Liu, Chung-Han Hsieh (arXiv:2609.29887v2)
"""

# Transaction cost rate (Section 4.1 and Appendix B.4/B.5)
TRANSACTION_COST: float = 0.001  # 10 bps

# Candidate window sets (Section 4.1, Appendix B)
WINDOWS_SMALL: list[int] = [5, 10, 21, 63]           # Main synthetic experiment (Section 4.1)
WINDOWS_LARGE: list[int] = [5, 10, 21, 30, 41, 50, 63, 126]  # Empirical experiments (Appendix B.4/B.5)

# Synthetic data generation parameters (Section 4.1; numeric values assumed; paper gives structure only)
SYN_N_STOCKS: int = 32
SYN_T: int = 2000
SYN_REGIME_CHANGE: int = 1000  # day after which j_t switches from 63 to 5
SYN_J1: int = 63               # window governing first regime
SYN_J2: int = 5                # window governing second regime
SYN_START_DATE: str = "2020-01-01"

# DGP parameters (assumed; paper specifies structure but not numerical values)
SYN_MU_DAILY: float = 0.0005      # baseline daily expected return per stock
SYN_ALPHA: float = 0.5            # signal coefficient in (0,1)
SYN_SIGMA: float = 0.02           # Gaussian noise std per stock per day (i.i.d.)
SYN_SEED: int = 42

# Loss bounds (assumed conservatively; paper requires bounded losses for regret formula)
LOSS_A: float = -0.15   # lower bound on prediction loss l(w,y) = -w'y
LOSS_B: float = 0.15    # upper bound on prediction loss

# Empirical backtest periods (Appendix B.4, B.5; Table 1)
DJIA_START: str = "2021-03-04"
DJIA_END: str = "2024-02-25"
SP500_START: str = "2020-01-02"
SP500_END: str = "2026-08-01"

# Switching budget formula parameter (assumed; paper uses K(T)=floor(T/(log(T)*(log(log(T)))^(1+eps))))
K_EPSILON: float = 0.1

# DJIA tickers (Appendix C)
DJIA_TICKERS: list[str] = [
    "WBA", "CRM", "HON", "AMGN", "DOW", "AAPL", "GS", "V", "NKE", "UNH",
    "TRV", "CSCO", "CVX", "VZ", "MSFT", "HD", "INTC", "JNJ", "WMT", "CAT",
    "JPM", "DIS", "BA", "KO", "MCD", "AXP", "IBM", "MRK", "MMM", "PG",
    "BIL",
]
DJIA_INDEX_TICKER: str = "I:DJI"

# S&P 500 tickers (Appendix C) — 481 stocks + BIL risk-free
SP500_TICKERS: list[str] = [
    "A", "AAPL", "ABBV", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP",
    "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB",
    "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT",
    "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APTV", "ARE",
    "ARES", "ATO", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BAC",
    "BALL", "BAX", "BBY", "BDX", "BEN", "BG", "BIIB", "BKNG", "BKR", "BLDR",
    "BLK", "BMY", "BNY", "BR", "BRK-B", "BRO", "BSX", "BWA", "BX", "BXP",
    "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL",
    "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI",
    "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC",
    "CNP", "COF", "COO", "COP", "COR", "COST", "CPB", "CPRT", "CPT", "CRWD",
    "CSCO", "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "CVS", "CVX",
    "CZR", "D", "DAL", "DD", "DE", "DFS", "DG", "DGX", "DHI", "DHR", "DIS",
    "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM",
    "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR",
    "ENPH", "EOG", "EPAM", "EQT", "ERIE", "ES", "ESS", "ETN", "ETR", "EVRG",
    "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS",
    "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FLT", "FMC", "FOX",
    "FOXA", "FRT", "FSLR", "FTNT", "GE", "GEHC", "GEV", "GILD", "GIS", "GL",
    "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW",
    "HAL", "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX",
    "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM",
    "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", "INTU", "IP",
    "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT",
    "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC",
    "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "L", "LDOS", "LEN",
    "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LUV",
    "LVS", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK",
    "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM",
    "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRO",
    "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ",
    "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC",
    "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL",
    "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA",
    "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PGR", "PH",
    "PHM", "PKG", "PLD", "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG",
    "PPL", "PRU", "PSA", "PTC", "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG",
    "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST",
    "RSG", "RTX", "RVTY", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SNA", "SNPS",
    "SO", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK",
    "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL",
    "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRMB", "TROW",
    "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL",
    "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V",
    "VLO", "VLTO", "VMC", "VRSK", "VRSN", "VRTX", "VTR", "VZ", "WAB", "WAT",
    "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WHR", "WM", "WMB", "WMT",
    "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH",
    "ZBRA", "ZTS", "BIL",
]
SP500_INDEX_TICKER: str = "I:SPX"

# Annualization factor for Sharpe ratio (trading days per year)
ANNUALIZATION: int = 252
