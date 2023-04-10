import pandas as pd
import yfinance as yf
import os

# NASDAQ-100
nasdaq100 = [
    'AAPL', 'ABNB', 'ADBE', 'ADI', 'ADP', 'ADSK', 'AEP', 'ALGN', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'ANSS', 'ASML', 'ATVI',
    'AVGO', 'AZN', 'BIDU', 'BIIB', 'BKNG', 'CDNS', 'CEG', 'CHTR', 'CMCSA', 'COST', 'CPRT', 'CRWD', 'CSCO', 'CSX', 'CTAS',
    'CTSH', 'DDOG', 'DLTR', 'DOCU', 'DXCM', 'EA', 'EBAY', 'EXC', 'FAST', 'FB', 'FISV', 'FTNT', 'GILD', 'GOOG', 'GOOGL',
    'HON', 'IDXX', 'ILMN', 'INTC', 'INTU', 'ISRG', 'JD', 'KDP', 'KHC', 'KLAC', 'LCID', 'LRCX', 'LULU', 'MAR', 'MCHP',
    'MDLZ', 'MELI', 'MNST', 'MRNA', 'MRVL', 'MSFT', 'MTCH', 'MU', 'NFLX', 'NTES', 'NVDA', 'NXPI', 'ODFL', 'OKTA',
    'ORLY', 'PANW', 'PAYX', 'PCAR', 'PDD', 'PEP', 'PYPL', 'QCOM', 'REGN', 'ROST', 'SBUX', 'SGEN', 'SIRI', 'SNPS',
    'SPLK', 'SWKS', 'TEAM', 'TMUS', 'TSLA', 'TXN', 'VRSK', 'VRSN', 'VRTX', 'WBA', 'WDAY', 'XEL', 'ZM', 'ZS']
nasdaq100.remove('CEG') # no data from 2018.1.1 to 2021.12.31

# S&P-500
snp500 = [
    'AAPL', 'MSFT', 'AMZN', 'TSLA', 'GOOGL', 'GOOG', 'BRK-B', 'UNH', 'NVDA', 'JNJ', 'FB', 'PG', 'JPM', 'XOM', 'V', 'HD',
    'CVX', 'MA', 'ABBV', 'PFE', 'BAC', 'KO', 'COST', 'AVGO', 'PEP', 'WMT', 'LLY', 'TMO', 'VZ', 'CSCO', 'DIS', 'MRK',
    'ABT', 'CMCSA', 'ACN', 'ADBE', 'INTC', 'MCD', 'WFC', 'CRM', 'DHR', 'BMY', 'NKE', 'TXN', 'PM', 'LIN', 'RTX', 'QCOM',
    'UNP', 'NEE', 'MDT', 'AMD', 'AMGN', 'T', 'UPS', 'SPGI', 'CVS', 'LOW', 'HON', 'INTU', 'COP', 'PLD', 'IBM', 'ANTM',
    'MS', 'ORCL', 'AMT', 'CAT', 'TGT', 'DE', 'AXP', 'GS', 'LMT', 'SCHW', 'C', 'MO', 'PYPL', 'AMAT', 'GE', 'BA', 'NFLX',
    'BLK', 'NOW', 'ADP', 'BKNG', 'MDLZ', 'ISRG', 'SBUX', 'CB', 'DUK', 'MMC', 'ZTS', 'MMM', 'CCI', 'SYK', 'CI', 'ADI',
    'SO', 'CME', 'GILD', 'MU', 'CSX', 'TMUS', 'TJX', 'EW', 'REGN', 'PNC', 'BDX', 'AON', 'D', 'USB', 'VRTX', 'CL', 'EOG',
    'TFC', 'EQIX', 'ICE', 'NOC', 'LRCX', 'PGR', 'BSX', 'NSC', 'EL', 'FCX', 'ATVI', 'PSA', 'CHTR', 'FIS', 'WM', 'F',
    'NEM', 'SHW', 'SLB', 'ITW', 'ETN', 'DG', 'FISV', 'GM', 'HUM', 'COF', 'EMR', 'GD', 'SRE', 'APD', 'PXD', 'MCO', 'ADM',
    'DOW', 'AEP', 'MPC', 'ILMN', 'HCA', 'AIG', 'FDX', 'OXY', 'MRNA', 'KLAC', 'CNC', 'MAR', 'MET', 'LHX', 'ROP', 'MCK',
    'ORLY', 'EXC', 'KMB', 'NXPI', 'WBD', 'SYY', 'AZO', 'JCI', 'NUE', 'GIS', 'SNPS', 'PRU', 'IQV', 'CTSH', 'ECL', 'HLT',
    'DLR', 'WMB', 'DXCM', 'CTVA', 'VLO', 'PAYX', 'TRV', 'O', 'APH', 'CMG', 'WELL', 'SPG', 'STZ', 'FTNT', 'ADSK', 'CDNS',
    'TEL', 'IDXX', 'SBAC', 'XEL', 'HPQ', 'PSX', 'TWTR', 'GPN', 'KR', 'AFL', 'MSI', 'DLTR', 'PEG', 'KMI', 'AJG', 'ALL',
    'MSCI', 'ROST', 'A', 'MCHP', 'DVN', 'BAX', 'EA', 'CTAS', 'CARR', 'PH', 'YUM', 'AVB', 'DD', 'TT', 'ED', 'HAL',
    'VRSK', 'RMD', 'EBAY', 'TDG', 'BK', 'WEC', 'FAST', 'WBA', 'HSY', 'DFS', 'MNST', 'IFF', 'SIVB', 'ES', 'PPG', 'AMP',
    'OTIS', 'WY', 'EQR', 'MTB', 'BIIB', 'TROW', 'OKE', 'KHC', 'ROK', 'AWK', 'PCAR', 'MTD', 'AME', 'HES', 'BKR', 'WTW',
    'APTV', 'CMI', 'ARE', 'EXR', 'CBRE', 'BLL', 'FRC', 'LYB', 'TSN', 'DAL', 'RSG', 'LUV', 'CERN', 'EXPE', 'EIX', 'KEYS',
    'DTE', 'ALGN', 'ANET', 'FE', 'FITB', 'ZBH', 'STT', 'WST', 'MKC', 'GLW', 'CHD', 'ODFL', 'LH', 'AEE', 'CPRT', 'EFX',
    'ETR', 'MOS', 'IT', 'ANSS', 'HIG', 'ABC', 'MAA', 'TSCO', 'CTRA', 'ALB', 'STE', 'VTR', 'DHI', 'CDW', 'SWK', 'DRE',
    'ESS', 'URI', 'PPL', 'VMC', 'ULTA', 'FANG', 'MLM', 'NTRS', 'MTCH', 'TDY', 'GWW', 'CF', 'CMS', 'CFG', 'FTV', 'ENPH',
    'DOV', 'HPE', 'FLT', 'CINF', 'RF', 'CEG', 'LEN', 'ZBRA', 'CNP', 'VRSN', 'HBAN', 'SYF', 'BBY', 'MRO', 'NDAQ', 'KEY',
    'PKI', 'COO', 'RJF', 'GPC', 'MOH', 'AKAM', 'SWKS', 'HOLX', 'PARA', 'IP', 'IR', 'PEAK', 'J', 'CLX', 'RCL', 'WAT',
    'AMCR', 'BXP', 'TER', 'VFC', 'PFG', 'K', 'MPWR', 'UDR', 'DRI', 'CPT', 'CAH', 'CAG', 'PWR', 'BR', 'FMC', 'NTAP',
    'EXPD', 'WAB', 'TRMB', 'POOL', 'GRMN', 'OMC', 'STX', 'UAL', 'IRM', 'DGX', 'SBNY', 'EVRG', 'CTLT', 'FDS', 'ATO',
    'TTWO', 'BRO', 'LNT', 'TYL', 'KIM', 'EPAM', 'TECH', 'CE', 'WDC', 'MGM', 'SJM', 'PKG', 'XYL', 'LDOS', 'HRL', 'CCL',
    'GNRC', 'TFX', 'AES', 'TXT', 'NLOK', 'APA', 'KMX', 'JKHY', 'HST', 'IEX', 'INCY', 'LYV', 'WRB', 'CZR', 'JBHT',
    'PAYC', 'BBWI', 'NVR', 'DPZ', 'AVY', 'IPG', 'EMN', 'CRL', 'AAP', 'HWM', 'CHRW', 'ABMD', 'LKQ', 'WRK', 'SEDG', 'AAL',
    'L', 'RHI', 'CTXS', 'LVS', 'ETSY', 'VTRS', 'FOXA', 'MAS', 'BF-B', 'HSIC', 'CBOE', 'QRVO', 'FFIV', 'NI', 'NDSN',
    'SNA', 'BIO', 'JNPR', 'RE', 'HAS', 'LNC', 'CMA', 'REG', 'AIZ', 'PHM', 'WHR', 'PTC', 'ALLE', 'TAP', 'MKTX', 'LUMN',
    'LW', 'SEE', 'UHS', 'FBHS', 'GL', 'NLSN', 'CPB', 'NRG', 'ZION', 'BWA', 'XRAY', 'HII', 'NCLH', 'PNW', 'PNR', 'TPR',
    'NWL', 'AOS', 'FRT', 'OGN', 'NWSA', 'WYNN', 'CDAY', 'DISH', 'ROL', 'DXC', 'BEN', 'ALK', 'IVZ', 'MHK', 'DVA', 'VNO',
    'PENN', 'PVH', 'RL', 'FOX', 'IPGP', 'UAA', 'UA', 'NWS', 'EMBC']
# no data/missing data from 2018.1.1 to 2021.12.31
snp500.remove('EMBC')
snp500.remove('CEG')
snp500.remove('CARR')
snp500.remove('CDAY')
snp500.remove('CTVA')
snp500.remove('DOW')
snp500.remove('FOX')
snp500.remove('FOXA')
snp500.remove('MRNA')
snp500.remove('OGN')
snp500.remove('OTIS')

snp500index = ['^GSPC']


class PortDataset():
    def __init__(self, asset_name, history_length, future_length, train_test_split=0.5,
                 start_date = '2018-01-01', end_date = '2021-12-30'):
        self.asset_name = asset_name
        self.history_length = history_length
        self.future_length = future_length

        # Prepare data
        self.assets = eval(asset_name)
        self.assets.sort()

        # Download or load data
        csv_path = f'data/{asset_name}.csv'
        if os.path.exists(csv_path):
            data = pd.read_csv(f'data/{asset_name}.csv')
        else:
            data = yf.download(self.assets, start=start_date, end=end_date)
            data = data.loc[:, ('Adj Close')].to_frame()
            #data = data.loc[:,('Adj Close', slice(None))]
            data.columns = self.assets

        # Calculate returns
        returns = data[self.assets].pct_change().dropna()
        self.num_assets = returns.shape[1]
        self.num_data_points = returns.shape[0]
        print(f'Fetched dataset with {self.num_assets} assets and {self.num_data_points} data points.')

        # Prepare train/test datasets
        self.train_set = []
        self.test_set = []
        i = 0
        cur_set = self.train_set
        while i + history_length + future_length < self.num_data_points:
            # compatibility for newer/older yfinance API
            if 'Date' in data:
                real_date = data['Date'].iloc[i+history_length]
            else:
                real_date = data.index[i+history_length]

            cur_set.append({
                'date_index': i,
                'history': returns.iloc[i:i+history_length, :],
                'future': returns.iloc[i+history_length:i+history_length+future_length, :],
                'real_date': real_date # the starting date of 'future'
            })
            if i + history_length + future_length <= self.num_data_points * train_test_split <= i + history_length + future_length + 1:
                i = int(self.num_data_points * train_test_split - history_length)
                cur_set = self.test_set
            i += 1


if __name__ == '__main__':
    # Below is a toy example of portfolio optimization using off-the-shelf riskfolio library

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import riskfolio as rp
    import warnings
    from portfolio_opt_methods import *
    import torch

    warnings.filterwarnings("ignore")
    pd.options.display.float_format = '{:.4%}'.format

    dataset = PortDataset('snp500index', 120, 120, train_test_split=0.75)

    # Building the portfolio object
    port = rp.Portfolio(returns=dataset.test_set[0]['future'])

    # Calculating optimal portfolio

    # Select method and estimate input parameters:

    method_mu='hist' # Method to estimate expected returns based on historical data.
    method_cov='hist' # Method to estimate covariance matrix based on historical data.

    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

    # Call gurobi to solve it
    risk_free_return = 0.03
    t_mu = torch.tensor(port.mu.values).squeeze(0) * 252
    t_cov = torch.tensor(port.cov.values) * 252
    assert is_psd(t_cov)
    risk, weight, _ = gurobi_portfolio_opt(t_mu, t_cov, rf=risk_free_return, obj='Sharpe', card_constr=5, linear_relaxation=False, timeout_sec=-1)
    print('** constrained return:', (weight * t_mu).sum())
    print('** constrained risk:', risk ** 0.5)
    weight = pd.DataFrame(weight.unsqueeze(0).cpu().numpy(), columns=dataset.assets)

    #r1 = rp.Sharpe_Risk(weight.T, port.cov, port.returns)

    non_constr_risk, non_constr_weight, _ = gurobi_portfolio_opt(t_mu, t_cov, rf=risk_free_return, obj='Sharpe')
    print('** non-constrained return:', (non_constr_weight * t_mu).sum())
    print('** non-constrained risk:', non_constr_risk ** 0.5)
    non_constr_weight = pd.DataFrame(non_constr_weight.unsqueeze(0).cpu().numpy(), columns=dataset.assets)

    #print('diff:', risk ** 0.5 - non_constr_risk ** 0.5)

    # the following result drops the w>=0 constraint
    z = torch.mm(torch.inverse(t_cov), (t_mu.unsqueeze(1) - risk_free_return))
    w = z / z.sum()

    # Estimate optimal portfolio:

    model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    rm = 'MV' # Risk measure used, this time will be variance
    obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = True # Use historical scenarios for risk measures that depend on scenarios
    rf = 0.03 / 252 # Risk free rate
    l = 0 # Risk aversion factor, only useful when obj is 'Utility'

    w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

    print(w.T)

    points = 50 # Number of points of the frontier

    frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)

    # Plotting the efficient frontier

    label = f'Sharpe Optimal Portfolio under Cardinality Constraint' # Title of point
    mu = port.mu # Expected returns
    cov = port.cov # Covariance matrix
    returns = port.returns # Returns of the assets

    ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                          rf=rf, alpha=0.05, cmap='viridis', w=weight.T, label=label,
                          marker='*', s=16, c='r', height=6, width=10, ax=None)
    plt.savefig('efficient_frontier.png')
