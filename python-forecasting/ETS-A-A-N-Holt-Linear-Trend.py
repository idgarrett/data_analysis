
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

# 2. Holt's Linear Trend HOLT alpha & beta smoothing parameter optimization and forecasting

# 2.1. Holt's Linear Trend HOLT  optimal alpha, beta smoothing parameters calculation for training range

def HOLTopt(params):
    alpha = params[0]
    beta = params[1]
    levellist = [tseries[0]]
    trendlist = [tseries[1] - tseries[0]]
    holtlist = [levellist[0] + trendlist[0]]
    sse = 0.00
    for j in range(1, len(tseries)):
        levellist.insert(j, alpha * tseries[j] + (1 - alpha) * (levellist[j - 1] + trendlist[j - 1]))
        trendlist.insert(j, beta*(levellist[j] - levellist[j -1]) + (1 - beta) * trendlist[j - 1])
        holtlist.insert(j, levellist[j] + trendlist[j])
        sse += (tseries[j] - holtlist[j - 1]) ** 2
    return sse

paramsopt = opt.minimize(HOLTopt, x0=[0.001, 0.001], bounds=[(0, 1), (0, 1)])
alphaopt = paramsopt.x[0]
betaopt = paramsopt.x[1]

# 2.2. Holt's Linear Trend HOLT calculation for forecasting range with initial values as final from training range

def HOLT(tseries, fseries, alpha, beta):
    # Calculate Holt's Linear Trend HOLT for training range
    tlevellist = [tseries[0]]
    ttrendlist = [tseries[1] - tseries[0]]
    tholtlist = [tlevellist[0] + ttrendlist[0]]
    for j in range(1, len(tseries)):
        tlevellist.insert(j, alpha * tseries[j] + (1 - alpha) * (tlevellist[j - 1] + ttrendlist[j - 1]))
        ttrendlist.insert(j, beta * (tlevellist[j] - tlevellist[j -1]) + (1 - beta) * ttrendlist[j - 1])
        tholtlist.insert(j, tlevellist[j] + ttrendlist[j])
    # Calculate Holt's Linear Trend HOLT for forecasting range using final level, trend values from training range
    flevellist = [alpha * fseries[0] + (1 - alpha) * (tlevellist[-1] + ttrendlist[-1])]
    ftrendlist = [beta * (flevellist[0] - tlevellist[-1]) + (1 - beta) * ttrendlist[-1]]
    fholtlist = [flevellist[0] + ftrendlist[0]]
    for k in range(1, len(fseries)):
        flevellist.insert(k, alpha * fseries[k] + (1 - alpha) * (flevellist[k - 1] + ftrendlist[k - 1]))
        ftrendlist.insert(k, beta * (flevellist[k] - flevellist[k -1]) + (1 - beta) * ftrendlist[k - 1])
        fholtlist.insert(k, flevellist[k] + ftrendlist[k])
    return fholtlist

# Add new Holt's Linear Trend HOLT forecasting series with optimized alpha, beta
alpha = alphaopt
beta = betaopt
fdata['HOLT'] = HOLT(tseries, fseries, alpha, beta)
print(fdata)

print("")
print("Optimized Alpha (Level):", alpha)
print("Optimized Beta (Trend):", beta)

# Chart Holt's Linear Trend HOLT Forecast with optimized alpha, beta
fdata.plot(y=['Yt', 'HOLT'])
plt.title('Holt Linear Trend ETS(A,A,N) Forecast')
plt.legend(loc='upper left')
plt.show()

# 3. Holt's Linear Trend HOLT forecasting accuracy

# 3.1. Calculate Scale-Dependant Mean Absolute Error MAE
holtmaelist = []
for i in range(0, len(fseries)):
    holtmaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['HOLT'][i - 1]))
HOLTmae = np.mean(holtmaelist)

# 3.2. Calculate Scale-Independent Mean Absolute Percentage Error MAPE, Mean Absolute Scaled Error MASE
# using random walk as forecasting benchmark
holtmapelist = []
rndmapelist = []
for i in range(0, len(fseries)):
    holtmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['HOLT'][i - 1])) / fdata['Yt'][i])
    rndmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['Ytf'][i - 1])) / fdata['Yt'][i])
HOLTmape = np.mean(holtmapelist) * 100
RNDmape = np.mean(rndmapelist) * 100

HOLTmase = HOLTmape / RNDmape

# Print forecasting accuracy comparison results
print("")
print("Holt Linear Trend ETS(A,A,N) Forecast Accuracy:")
print("")
print("Mean Absolute Error MAE:", HOLTmae)
print("Mean Absolute Percentage Error MAPE:", HOLTmape)
print("Mean Absolute Scaled Error MASE:", HOLTmase)
