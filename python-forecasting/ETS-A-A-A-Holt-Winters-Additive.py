
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

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

# 1.4. Create training and forecasting series
# Create training series and forecasting series length
series = data['Yt']
tseries = tdata['Yt']
fserieslen = fdata['Yt']

# Add new forecasting series with last training series value and second to last for initial value
fdata['Ytf'] = tseries[-1]
fseries = fdata['Ytf']

# 2. Holt-Winters Additive Seasonality HWA alpha, beta, gamma smoothing parameters optimization and forecasting

# 2.1. Holt-Winters Additive Seasonality HWA initial level, trend and seasonality

# 2.1.1. Initial level value calculation for training range
def ilevel(tseries, season):
    ilevelsum = 0.00
    for i in range(0, season):
        ilevelsum += tseries[i]
    ilevel = ilevelsum / season
    return ilevel

# 2.1.2. Initial trend value calculation for training range

def itrend(tseries, season):
    itrendsum = 0.00
    for i in range(0, season):
        itrendsum += (tseries[i + season] - tseries[i]) / season
    itrend = itrendsum / season
    return itrend

# 2.1.3. Initial season list calculation for training range
def iseason(tseries, season):
    iseason = []
    for i in range(0, season):
        iseason.insert(i, tseries[i] - ilevel(tseries, season))
    return iseason

# 2.2. Holt-Winters Additive Seasonality HWA  optimal alpha, beta, gamma smoothing parameters calculation
# for training range

def HWAopt(params):
    alpha = params[0]
    beta = params[1]
    gamma = params[2]
    season = 5
    levellist = [ilevel(tseries, season)]
    trendlist = [itrend(tseries, season)]
    seasonlist = iseason(tseries, season)
    hwalist = [levellist[0] + trendlist[0] + seasonlist[0]]
    sse = 0.00
    for j in range(1, len(tseries)):
        levellist.insert(j, alpha * (tseries[j] - seasonlist[j - 1]) + (1 - alpha) * (levellist[j - 1] +
                                                                                      trendlist[j - 1]))
        trendlist.insert(j, beta*(levellist[j] - levellist[j - 1]) + (1 - beta) * trendlist[j - 1])
        seasonlist.insert(j + season - 1, gamma * (tseries[j] - levellist[j - 1] - trendlist[j - 1]) +
                          (1 - gamma) * seasonlist[j - 1])
        hwalist.insert(j, levellist[j] + trendlist[j] + seasonlist[j])
        sse += (tseries[j] - hwalist[j - 1]) ** 2
    return sse

paramsopt = opt.minimize(HWAopt, x0=[0.001, 0.001, 0.001], bounds=[(0, 1), (0, 1), (0, 1)])
alphaopt = paramsopt.x[0]
betaopt = paramsopt.x[1]
gammaopt = paramsopt.x[2]

# 2.3. Holt-Winters Additive Seasonality HWA calculation for forecasting range with initial values as final from
# training range

def HWA(tseries, fseries, season, alpha, beta, gamma):
    # Calculate Holt-Winters Additive Seasonality HWA for training range
    tlevellist = [ilevel(tseries, season)]
    ttrendlist = [itrend(tseries, season)]
    tseasonlist = iseason(tseries, season)
    thwalist = [tlevellist[0] + ttrendlist[0] + tseasonlist[0]]
    for j in range(1, len(tseries)):
        tlevellist.insert(j, alpha * (tseries[j] - tseasonlist[j - 1]) + (1 - alpha) * (tlevellist[j - 1]
                                                                                        + ttrendlist[j - 1]))
        ttrendlist.insert(j, beta*(tlevellist[j] - tlevellist[j - 1]) + (1 - beta) * ttrendlist[j - 1])
        tseasonlist.insert(j + season, gamma * (tseries[j] - tlevellist[j - 1] - ttrendlist[j - 1])
                           + (1 - gamma) * tseasonlist[j - 1])
        thwalist.insert(j, tlevellist[j] + ttrendlist[j] + tseasonlist[j])
    # Calculate Holt-Winters Additive Seasonality HWA for forecasting range using final level, trend, seasonality values
    # from training range
    flevellist = [alpha * (tseries[j] - tseasonlist[j]) + (1 - alpha) * (tlevellist[j] + ttrendlist[j])]
    ftrendlist = [beta * (flevellist[0] - tlevellist[j]) + (1 - beta) * ttrendlist[j]]
    tseasonlist.insert(j + season + 1, gamma * (tseries[j] - tlevellist[j] - ttrendlist[j])
                       + (1 - gamma) * tseasonlist[j])
    fhwalist = [flevellist[0] + ftrendlist[0] + tseasonlist[j + 1]]
    for k in range(1, len(fseries)):
        flevellist.insert(k, alpha * (fseries[k] - tseasonlist[j + k]) + (1 - alpha) * (flevellist[k - 1]
                                                                                        + ftrendlist[k - 1]))
        ftrendlist.insert(k, beta*(flevellist[k] - flevellist[k - 1]) + (1 - beta) * ftrendlist[k - 1])
        tseasonlist.insert(j + season + 1 + k, gamma * (fseries[k] - flevellist[k - 1] - ftrendlist[k - 1])
                           + (1 - gamma) * tseasonlist[j + k])
        fhwalist.insert(k, flevellist[k] + ftrendlist[k] + tseasonlist[j + 1 + k])
    return fhwalist

# Add new Holt-Winters Additive Seasonality HWA forecasting series with optimized alpha, beta, gamma
season = 5
alpha = alphaopt
beta = betaopt
gamma = gammaopt
fdata['HWA'] = HWA(tseries, fseries, season, alpha, beta, gamma)
print(fdata)

print("")
print("Optimized Alpha (Level):", alpha)
print("Optimized Beta (Trend):", beta)
print("Optimized Gamma (Seasonality):", gamma)

# Chart Holt-Winters Additive Seasonality HWA Forecast with optimized alpha, beta, gamma
fdata.plot(y=['Yt', 'HWA'])
plt.title('Holt-Winters Additive Seasonality ETS(A,A,A) Forecast')
plt.legend(loc='upper left')
plt.show()

# 3. Holt-Winters Additive Seasonality HWA forecasting accuracy

# 3.1. Calculate Scale-Dependant Mean Absolute Error MAE
hwamaelist = []
for i in range(0, len(fseries)):
    hwamaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['HWA'][i - 1]))
HWAmae = np.mean(hwamaelist)

# 3.2. Calculate Scale-Independent Mean Absolute Percentage Error MAPE, Mean Absolute Scaled Error MASE
# using random walk as forecasting benchmark
hwamapelist = []
rndmapelist = []
for i in range(0, len(fseries)):
    hwamapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['HWA'][i - 1])) / fdata['Yt'][i])
    rndmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['Ytf'][i - 1])) / fdata['Yt'][i])
HWAmape = np.mean(hwamapelist) * 100
RNDmape = np.mean(rndmapelist) * 100

HWAmase = HWAmape / RNDmape

# Print forecasting accuracy comparison results
print("")
print("Holt-Winters Additive Seasonality ETS(A,A,A) Forecast Accuracy:")
print("")
print("Mean Absolute Error MAE:", HWAmae)
print("Mean Absolute Percentage Error MAPE:", HWAmape)
print("Mean Absolute Scaled Error MASE:", HWAmase)