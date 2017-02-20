
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

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

# 2. Simple Moving Average SMA, Weighted Moving Average WMA, Exponentially Weighted Moving Average EWMA forecasting

# 2.1. Simple Moving Average SMA forecasting

# Simple Moving Average SMA calculation for training range
SMA = pd.rolling_mean(tseries, 2)

# Simple Moving Average SMA calculation for forecasting range with last calculation for training range
for i in range(1, len(fseries)):
    fdata['SMA'] = SMA[-1]
print(fdata)

# Chart Simple Moving Average SMA Forecast
fdata.plot(y=['Yt', 'SMA'])
plt.title('Simple Moving Average SMA Forecast')
plt.legend(loc='upper left')
plt.show()

# 2.2. Exponentially Weighted Moving Average EWMA forecasting

# Exponentially Weighted Moving Average EWMA  calculation for training range
EWMA = pd.ewma(tseries, span=2)

# Exponentially Weighted Moving Average EWMA calculation for forecasting range with last calculation for training range
for i in range(1, len(fseries)):
    fdata['EWMA'] = EWMA[-1]
print(fdata)

# Chart Exponentially Weighted Moving Average EWMA Forecast
fdata.plot(y=['Yt', 'EWMA'])
plt.title('Exponentially Weighted Moving Average EWMA Forecast')
plt.legend(loc='upper left')
plt.show()

# 2.3. Weighted Moving Average WMA forecasting

# 2.3.1. Weighted Moving Average WMA optimal weight1, weight2 parameters calculation for training range
def WMAopt(params):
    weight1 = params[0]
    weight2 = 1 - weight1
    wmalist = [tseries[0], tseries[1]]
    wma = 0.00
    sse = 0.00
    for j in range(2, len(tseries)):
        wma = tseries[j - 1] * weight1 + tseries[j - 2] * weight2
        wmalist.insert(j, wma)
        sse += (tseries[j] - wmalist[j]) ** 2
    return sse

paramsopt = opt.minimize(WMAopt, x0=[0.001], bounds=[(0, 1)])
weightsopt = paramsopt.x

# 2.3.2. Weighted Moving Average WMA calculation for forecasting range with last calculation for training range

# Calculate Weighted Moving Average WMA for training range
def WMA(tseries, weights):
    wmalist = [tseries[0], tseries[1]]
    wma = 0.00
    weight1 = weights[0]
    weight2 = 1 - weight1
    for i in range(2, len(tseries)):
        wma = tseries[i - 1] * weight1 + tseries[i - 2] * weight2
        wmalist.insert(i, wma)
    return wmalist

# Add new Weighted Moving Average WMA forecasting series with optimized weights
weights = weightsopt
for i in range(1, len(fseries)):
    fdata['WMA'] = WMA(tseries, weights)[-1]
print(fdata)

print("")
print("Optimal Weight (t-1):", weights[0])
print("Optimal Weight (t-2):", 1 - weights[0])
print("")

# Chart Weighted Moving Average WMA Forecast
fdata.plot(y=['Yt', 'WMA'])
plt.title('Weighted Moving Average WMA Forecast')
plt.legend(loc='upper left')
plt.show()

# 3. Simple Moving Average SMA, Weighted Moving Average WMA, Exponentially Weighted Moving Average EWMA
# forecasting accuracy

# 3.1. Calculate Scale-Dependant Mean Absolute Error MAE
smamaelist = []
wmamaelist = []
ewmamaelist = []
for i in range(0, len(fseries)):
    smamaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['SMA'][i]))
    wmamaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['WMA'][i]))
    ewmamaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['EWMA'][i]))
SMAmae = np.mean(smamaelist)
WMAmae = np.mean(wmamaelist)
EWMAmae = np.mean(ewmamaelist)

# 3.2. Calculate Scale-Independent Mean Absolute Percentage Error MAPE, Mean Absolute Scaled Error MASE
# using random walk as forecasting benchmark
smamapelist = []
wmamapelist = []
ewmamapelist = []
rndmapelist = []
for i in range(0, len(fseries)):
    smamapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['SMA'][i])) / fdata['Yt'][i])
    wmamapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['WMA'][i])) / fdata['Yt'][i])
    ewmamapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['EWMA'][i])) / fdata['Yt'][i])
    rndmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['Ytf'][i])) / fdata['Yt'][i])
SMAmape = np.mean(smamapelist) * 100
WMAmape = np.mean(wmamapelist) * 100
EWMAmape = np.mean(ewmamapelist) * 100
RNDmape = np.mean(rndmapelist) * 100

SMAmase = SMAmape / RNDmape
WMAmase = WMAmape / RNDmape
EWMAmase = EWMAmape / RNDmape

# Print forecasting accuracy comparison results
print("")
print("Mean Absolute Error MAE")
print("SMA:", SMAmae)
print("WMA:", WMAmae)
print("EWMA:", EWMAmae)

print("")
print("Mean Absolute Percentage Error MAPE")
print("SMA:", SMAmape)
print("WMA:", WMAmape)
print("EWMA:", EWMAmape)

print("")
print("Mean Absolute Scaled Error MASE")
print("SMA:", SMAmase)
print("WMA:", WMAmase)
print("EWMA:", EWMAmase)
