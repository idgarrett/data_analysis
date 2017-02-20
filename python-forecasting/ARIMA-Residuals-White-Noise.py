
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# 1.2. CSV data file import
# Dates must be in format yyyy-mm-dd
data = pd.read_csv('/Users/iangarrett/Apple Daily.csv', index_col='Date',
                   parse_dates=True)

# Imported data chart
data.plot(y=['Yt'])
plt.title('Daily Apple Stock Prices 2014-10-01 to 2015-10-30')
plt.legend(loc='upper left')
plt.axvspan('2015-10-01', '2015-10-30', color='green', alpha=0.25)
plt.show()

# 1.3. Delimit training and forecasting ranges
tdata = data['2014-10-01':'2015-09-30']
fdata = data['2015-10-01':'2015-10-30']

# 1.4. Create full range and training series
series = data['Yt']
tseries = tdata['Yt']

# 2. Random Walk with Drift ARIMA(0, 1, 0) model calculation for training range
RWD = ARIMA(series[:len(tseries) - 1], order=(0, 1, 0)).fit()

# 3. Random Walk with Drift ARIMA(0, 1, 0) model Residuals White Noise
RWDres = RWD.resid
tdata['RWDres'] = RWDres

# Training range chart
tdata.plot(y=['RWDres'])
plt.title('Random Walk with Drift ARIMA(0,1,0) Model Residuals')
plt.legend(loc='upper left')
plt.show()

# Auto-correlation Function ACF calculation and chart
RWDresACF = acf(RWDres)
plt.title('Autocorrelation Function ACF (Random Walk with Drift ARIMA(0,1,0) Model Residuals)')
plt.bar(range(len(RWDresACF)), RWDresACF, width=1/2)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.show()

# Partial Auto-correltaion Function PACF calculation and chart
RWDresPACF = pacf(RWDres)
plt.title('Partial Autocorrelation Function PACF (Random Walk with Drift ARIMA(0,1,0) Model Residuals)')
plt.bar(range(len(RWDresPACF)), RWDresPACF, width=1/2)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.show()

# Ljung-Box Auto-correlation Test

print("")
print("Ljung-Box Autocorrelation Test (Random Walk with Drift ARIMA(0,1,0) Model Residuals):")
print("")
RWDresLB = acorr_ljungbox(RWDres, lags= 10)
RWDresLBseries = pd.Series(RWDresLB, index=['Q-statistic:','p-value:'])
RWDresLBqstat = (RWDresLBseries['Q-statistic:'])
RWDresLBpval = (RWDresLBseries['p-value:'])
RWDresLBdata = [{'Q-statistic:': RWDresLBqstat[0], 'p-value:': RWDresLBpval[0]},
                {'Q-statistic:': RWDresLBqstat[1], 'p-value:': RWDresLBpval[1]},
                {'Q-statistic:': RWDresLBqstat[2], 'p-value:': RWDresLBpval[2]},
                {'Q-statistic:': RWDresLBqstat[3], 'p-value:': RWDresLBpval[3]},
                {'Q-statistic:': RWDresLBqstat[4], 'p-value:': RWDresLBpval[4]},
                {'Q-statistic:': RWDresLBqstat[5], 'p-value:': RWDresLBpval[5]},
                {'Q-statistic:': RWDresLBqstat[6], 'p-value:': RWDresLBpval[6]},
                {'Q-statistic:': RWDresLBqstat[7], 'p-value:': RWDresLBpval[7]},
                {'Q-statistic:': RWDresLBqstat[8], 'p-value:': RWDresLBpval[8]},
                {'Q-statistic:': RWDresLBqstat[9], 'p-value:': RWDresLBpval[9]}]
RWDresLBtable = pd.DataFrame(RWDresLBdata)
print(RWDresLBtable)
print("")


