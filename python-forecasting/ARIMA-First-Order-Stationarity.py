import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf

# 1.2. CSV data file import
# Dates must be in format yyyy-mm-dd
data = pd.read_csv('C:\\Users\\Diego\\Desktop\\Forecasting Models with Python\\Apple Daily.csv', index_col='Date',
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

# 1.4. Create training and forecasting series
# Create training series and forecasting series length
series = data['Yt']
tseries = tdata['Yt']
fserieslen = fdata['Yt']

# Add new forecasting series with last training series value and second to last for initial value
fdata['Ytf'] = tseries[-1]
fseries = fdata['Ytf']

# 2. First Order Stationarity Tests for Level training range

# Training range chart
tdata.plot(y=['Yt'])
plt.title('Daily Apple Stock Prices Level 2014-10-01 to 2015-09-30')
plt.legend(loc='upper left')
plt.show()

# 2.1. Autocorrelation Function ACF, Partial Autocorrelation Function PACF calculation for Level training range

# Auto-correlation Function ACF calculation and chart
tseriesACF = acf(tseries)
plt.title('Autocorrelation Function ACF (Level, Training Range)')
plt.bar(range(len(tseriesACF)), tseriesACF, width=1/2)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.show()

# Partial Auto-correlation Function PACF calculation and chart
tseriesPACF = pacf(tseries)
plt.title('Partial Autocorrelation Function PACF (Level, Training Range)')
plt.bar(range(len(tseriesPACF)), tseriesPACF, width=1/2)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.show()

# 2.2. Augmented Dickey Fuller Test ADF calculation for Level training range

print("Augmented Dickey-Fuller Test ADF (Level, Training Range):")
print("")
tseriesADF = adfuller(tseries)
tseriesADFresult = pd.Series(tseriesADF[0:2], index=['Test statistic:','p-value:'])
print(tseriesADFresult)

# 3. First Order Stationarity Tests for Differentiated training range

# Differentiated training range calculation and chart
difftseries = tseries - tseries.shift()
difftseries[0] = 0.00
tdata['DYt'] = difftseries
tdata.plot(y=['DYt'])
plt.title('Daily Apple Stock Price Differences 2014-10-01 to 2015-09-30')
plt.legend(loc='upper left')
plt.show()

# 3.1. Auto-correlation Function ACF, Partial Auto-correlation Function PACF calculation for Differentiated
# training range

# Auto-correlation Function ACF calculation and chart
difftseriesACF = acf(difftseries)
plt.title('Autocorrelation Function ACF (First Difference, Training Range)')
plt.bar(range(len(difftseriesACF)), difftseriesACF, width=1/2)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.show()

# Partial Auto-correlation Function PACF calculation and chart
difftseriesPACF = pacf(difftseries)
plt.title('Partial Autocorrelation Function PACF (First Difference, Training Range)')
plt.bar(range(len(difftseriesPACF)), difftseriesPACF, width=1/2)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(tseries)), linestyle='--', color='gray')
plt.show()

# 3.2. Augmented Dickey Fuller Test ADF calculation for Differentiated training range

print("")
print("Augmented Dickey-Fuller Test ADF (First Difference, Training Range):")
print("")
difftseriesADF = adfuller(difftseries)
difftseriesADFresult = pd.Series(difftseriesADF[0:2], index=['Test statistic:','p-value:'])
print(difftseriesADFresult)
