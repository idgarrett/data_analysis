
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

# Add new forecasting series with last training series value
fdata['Ytf'] = tseries[-1]
fseries = fdata['Ytf']

# 2. Brown's Simple Exponential Smoothing SES general calculation, alpha smoothing parameter optimization
# and forecasting

# 2.1. Brown's Simple Exponential Smoothing SES optimal alpha smoothing parameter calculation for training range

def SESopt(params):
    alpha = params[0]
    seslist = [tseries[0]]
    sse = 0.00
    for j in range(1, len(tseries)):
        seslist.insert(j, alpha * tseries[j] + (1 - alpha) * seslist[j - 1])
        sse += (tseries[j] - seslist[j - 1]) ** 2
    return sse

paramsopt = opt.minimize(SESopt, x0=0.001, bounds=[(0, 1)])
alphaopt = paramsopt.x[0]

# 2.2. Brown's Simple Exponential Smoothing SES calculation for forecasting range with initial value as final
# from training range

def SES(tseries, fseries, alpha):
    tseslist = [tseries[0]]
    for j in range(1, len(tseries)):
        tseslist.insert(j, alpha * tseries[j] + (1 - alpha) * tseslist[j - 1])
    fseslist = [tseslist[-1]]
    for k in range(1, len(fseries)):
        fseslist.insert(k, alpha * fseries[k] + (1 - alpha) * fseslist[k - 1])
    return fseslist

# Add new Brown's Simple Exponential Smoothing SES forecasting series with optimized alpha
alpha = alphaopt
fdata['SES'] = SES(tseries, fseries, alpha)
print(fdata)

print("")
print("Optimized Alpha (Level):", alpha)
print("")

# Chart Brown's Simple Exponential Smoothing SES Forecast with optimized alpha
fdata.plot(y=['Yt', 'SES'])
plt.title('Brown Simple Exponential Smoothing ETS(A,N,N) Forecast')
plt.legend(loc='upper left')
plt.show()

# 3. Brown's Simple Exponential Smoothing SES forecasting accuracy

# 3.1. Calculate Scale-Dependant Mean Absolute Error MAE
sesmaelist = []
for i in range(0, len(fseries)):
    sesmaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['SES'][i - 1]))
SESmae = np.mean(sesmaelist)

# 3.2. Calculate Scale-Independent Mean Absolute Percentage Error MAPE, Mean Absolute Scaled Error MASE
# using random walk as forecasting benchmark
sesmapelist = []
rndmapelist = []
for i in range(0, len(fseries)):
    sesmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['SES'][i - 1])) / fdata['Yt'][i])
    rndmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['Ytf'][i - 1])) / fdata['Yt'][i])
SESmape = np.mean(sesmapelist) * 100
RNDmape = np.mean(rndmapelist) * 100

SESmase = SESmape / RNDmape

# Print forecasting accuracy comparison results
print("")
print("Brown Simple Exponential Smoothing ETS(A,N,N) Forecast Accuracy:")
print("")
print("Mean Absolute Error MAE:", SESmae)
print("Mean Absolute Percentage Error MAPE:", SESmape)
print("Mean Absolute Scaled Error MASE:", SESmase)
