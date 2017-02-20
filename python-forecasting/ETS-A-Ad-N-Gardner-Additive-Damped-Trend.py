
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

# 2. Gardner's Additive Damped Trend GARD alpha & beta smoothing, delta damping parameters optimization and forecasting

# 2.1. Gardner's Additive Damped Trend GARD  optimal alpha, beta smoothing, delta damping parameters calculation for
# training range

def GARDopt(params):
    alpha = params[0]
    beta = params[1]
    delta = params[2]
    levellist = [tseries[0]]
    trendlist = [tseries[1] - tseries[0]]
    gardlist = [levellist[0] + delta * trendlist[0]]
    sse = 0.00
    for j in range(1, len(tseries)):
        levellist.insert(j, alpha * tseries[j] + (1 - alpha) * (levellist[j - 1] + delta * trendlist[j - 1]))
        trendlist.insert(j, beta*(levellist[j] - levellist[j - 1]) + (1 - beta) * delta * trendlist[j - 1])
        gardlist.insert(j, levellist[j] + delta * trendlist[j])
        sse += (tseries[j] - gardlist[j - 1]) ** 2
    return sse

paramsopt = opt.minimize(GARDopt, x0=[0.001, 0.001, 0.001], bounds=[(0, 1), (0, 1), (0, 1)])
alphaopt = paramsopt.x[0]
betaopt = paramsopt.x[1]
deltaopt = paramsopt.x[2]

# 2.2. Gardner's Additive Damped Trend GARD calculation for forecasting range with initial values as final from
# training range

def GARD(tseries, fseries, alpha, beta, delta):
    # Calculate Gardner's Additive Damped Trend GARD for training range
    tlevellist = [tseries[0]]
    ttrendlist = [tseries[1] - tseries[0]]
    tgardlist = [tlevellist[0] + delta * ttrendlist[0]]
    for j in range(1, len(tseries)):
        tlevellist.insert(j, alpha * tseries[j] + (1 - alpha) * (tlevellist[j - 1] + delta * ttrendlist[j - 1]))
        ttrendlist.insert(j, beta * (tlevellist[j] - tlevellist[j - 1]) + (1 - beta) * delta * ttrendlist[j - 1])
        tgardlist.insert(j, tlevellist[j] + delta * ttrendlist[j])
    # Calculate Gardner's Additive Damped Trend GARD for forecasting range using final level, trend values from
    # training range
    flevellist = [alpha * fseries[0] + (1 - alpha) * (tlevellist[-1] + ttrendlist[-1])]
    ftrendlist = [beta * (flevellist[0] - tlevellist[-1]) + (1 - beta) * ttrendlist[-1]]
    fgardlist = [flevellist[0] + delta * ftrendlist[0]]
    for k in range(1, len(fseries)):
        flevellist.insert(k, alpha * fseries[k] + (1 - alpha) * (flevellist[k - 1] + delta * ftrendlist[k - 1]))
        ftrendlist.insert(k, beta * (flevellist[k] - flevellist[k - 1]) + (1 - beta) * delta * ftrendlist[k - 1])
        fgardlist.insert(k, flevellist[k] + delta * ftrendlist[k])
    return fgardlist

# Add new Gardner's Additive Damped Trend GARD forecasting series with optimized alpha, beta, delta
alpha = alphaopt
beta = betaopt
delta = deltaopt
fdata['GARD'] = GARD(tseries, fseries, alpha, beta, delta)
print(fdata)

print("")
print("Optimized Alpha (Level):", alpha)
print("Optimized Beta (Trend):", beta)
print("Optimized Delta (Trend Damping):", delta)

# Chart Gardner's Additive Damped Trend GARD Forecast with optimized alpha, beta, delta
fdata.plot(y=['Yt', 'GARD'])
plt.title('Gardner Additive Damped Trend ETS(A,Ad,N) Forecast')
plt.legend(loc='upper left')
plt.show()

# 3. Gardner's Additive Damped Trend GARD forecasting accuracy

# 3.1. Calculate Scale-Dependant Mean Absolute Error MAE
gardmaelist = []
for i in range(0, len(fseries)):
    gardmaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['GARD'][i - 1]))
GARDmae = np.mean(gardmaelist)

# 3.2. Calculate Scale-Independent Mean Absolute Percentage Error MAPE, Mean Absolute Scaled Error MASE
# using random walk as forecasting benchmark
gardmapelist = []
rndmapelist = []
for i in range(0, len(fseries)):
    gardmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['GARD'][i - 1])) / fdata['Yt'][i])
    rndmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['Ytf'][i - 1])) / fdata['Yt'][i])
GARDmape = np.mean(gardmapelist) * 100
RNDmape = np.mean(rndmapelist) * 100

GARDmase = GARDmape / RNDmape

# Print forecasting accuracy comparison results
print("")
print("Gardner Additive Damped Trend ETS(A,Ad,N) Forecast Accuracy:")
print("")
print("Mean Absolute Error MAE:", GARDmae)
print("Mean Absolute Percentage Error MAPE:", GARDmape)
print("Mean Absolute Scaled Error MASE:", GARDmase)
