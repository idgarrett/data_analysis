
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

# 2. Taylor's Multiplicative Damped Trend TAY alpha, beta smoothing, delta damping parameter optimization
# and forecasting

# 2.1. Taylor's Multiplicative Damped Trend TAY optimal alpha, beta smoothing, delta damping parameters calculation
# for training range

def TAYopt(params):
    alpha = params[0]
    beta = params[1]
    delta = params[2]
    levellist = [tseries[0]]
    trendlist = [tseries[1] / tseries[0]]
    taylist = [levellist[0] * (trendlist[0] ** delta)]
    sse = 0.00
    for j in range(1, len(tseries)):
        levellist.insert(j, alpha * tseries[j] + (1 - alpha) * (levellist[j - 1] * (trendlist[j - 1] ** delta)))
        trendlist.insert(j, beta*(levellist[j] / levellist[j - 1]) + (1 - beta) * (trendlist[j - 1] ** delta))
        taylist.insert(j, levellist[j] * (trendlist[j] ** delta))
        sse += (tseries[j] - taylist[j - 1]) ** 2
    return sse

paramsopt = opt.minimize(TAYopt, x0=[0.001, 0.001, 0.001], bounds=[(0, 1), (0, 1), (0, 1)])
alphaopt = paramsopt.x[0]
betaopt = paramsopt.x[1]
deltaopt = paramsopt.x[2]

# 2.2. Taylor's Multiplicative Damped Trend TAY calculation for forecasting range with initial values as final
# from training range

def TAY(tseries, fseries, alpha, beta, delta):
    # Calculate Taylor's Multiplicative Damped Trend TAY for training range
    tlevellist = [tseries[0]]
    ttrendlist = [tseries[1] / tseries[0]]
    ttaylist = [tlevellist[0] * (ttrendlist[0] ** delta)]
    for j in range(1, len(tseries)):
        tlevellist.insert(j, alpha * tseries[j] + (1 - alpha) * (tlevellist[j - 1] * (ttrendlist[j - 1] ** delta)))
        ttrendlist.insert(j, beta * (tlevellist[j] / tlevellist[j - 1]) + (1 - beta) * (ttrendlist[j - 1] ** delta))
        ttaylist.insert(j, tlevellist[j] * (ttrendlist[j] ** delta))
    # Calculate Taylor's Multiplicative Damped Trend TAY for forecasting range using final level, trend values
    # from training range
    flevellist = [alpha * fseries[0] + (1 - alpha) * (tlevellist[-1] * (ttrendlist[-1] ** delta))]
    ftrendlist = [beta * (flevellist[0] / tlevellist[-1]) + (1 - beta) * (ttrendlist[-1] ** delta)]
    ftaylist = [flevellist[0] * (ftrendlist[0] ** delta)]
    for k in range(1, len(fseries)):
        flevellist.insert(k, alpha * fseries[k] + (1 - alpha) * (flevellist[k - 1] * (ftrendlist[k - 1] ** delta)))
        ftrendlist.insert(k, beta * (flevellist[k] / flevellist[k - 1]) + (1 - beta) * (ftrendlist[k - 1] ** delta))
        ftaylist.insert(k, flevellist[k] * (ftrendlist[k] ** delta))
    return ftaylist

# Add new Taylor's Multiplicative Damped Trend TAY forecasting series with optimized alpha, beta, delta
alpha = alphaopt
beta = betaopt
delta = deltaopt
fdata['TAY'] = TAY(tseries, fseries, alpha, beta, delta)
print(fdata)

print("")
print("Optimized Alpha (Level):", alpha)
print("Optimized Beta (Trend):", beta)
print("Optimized Delta (Trend Damping):", delta)

# Chart Taylor's Multiplicative Damped Trend TAY Forecast with optimized alpha, beta, delta
fdata.plot(y=['Yt', 'TAY'])
plt.title('Taylor Multiplicative Damped Trend ETS(A,Md,N) Forecast')
plt.legend(loc='upper left')
plt.show()

# 3. Taylor's Multiplicative Damped Trend TAY forecasting accuracy

# 3.1. Calculate Scale-Dependant Mean Absolute Error MAE
taymaelist = []
for i in range(0, len(fseries)):
    taymaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['TAY'][i - 1]))
TAYmae = np.mean(taymaelist)

# 3.2. Calculate Scale-Independent Mean Absolute Percentage Error MAPE, Mean Absolute Scaled Error MASE
# using random walk as forecasting benchmark
taymapelist = []
rndmapelist = []
for i in range(0, len(fseries)):
    taymapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['TAY'][i - 1])) / fdata['Yt'][i])
    rndmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['Ytf'][i - 1])) / fdata['Yt'][i])
TAYmape = np.mean(taymapelist) * 100
RNDmape = np.mean(rndmapelist) * 100

TAYmase = TAYmape / RNDmape

# Print forecasting accuracy comparison results
print("")
print("Taylor Multiplicative Damped Trend ETS(A,Md,N) Forecast Accuracy:")
print("")
print("Mean Absolute Error MAE:", TAYmae)
print("Mean Absolute Percentage Error MAPE:", TAYmape)
print("Mean Absolute Scaled Error MASE:", TAYmase)