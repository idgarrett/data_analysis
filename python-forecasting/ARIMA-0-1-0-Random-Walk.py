import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA



# Dates must be in format yyyy-mm-dd
data = pd.read_csv('/Users/iangarrett/Apple Daily.csv', index_col='Date',
                   parse_dates=True)

# Imported data chart
data.plot(y=['Yt'])
plt.title('Daily Apple Stock Prices 2014-10-01 to 2015-10-30')
plt.legend(loc='upper left')
plt.axvspan('2015-10-01', '2015-10-30', color='green', alpha=0.25)
plt.show()

#Delimit training and forecasting ranges
tdata = data['2014-10-01':'2015-09-30']
fdata = data['2015-10-01':'2015-10-30']

# Create training and forecasting series
# Create full range and training series
series = data['Yt']
tseries = tdata['Yt']

# Add new forecasting and model data columns with last training series value and create forecasting and model series
fdata['Ytf'] = tseries[-1]
fdata['RWD'] = tseries[-1]
fseries = fdata['Ytf']
RWDseries = fdata['RWD']

#Random Walk with Drift ARIMA(0, 1, 0) model calculation for training range and out of sample
# forecasting for forecasting range

# 2.1. Model calculation  and parameters printing for training range
RWD = ARIMA(series[:len(tseries) - 1], order=(0, 1, 0)).fit()
print("")
print('Random Walk with Drift ARIMA(0,1,0) Parameters:')
print("")
print("Constant:", RWD.params[0])
print("")

#Model forecasting for full and then for only forecasting range

# Forecast at level of integration. ARIMA(p,0,q) level forecast, ARIMA(p,1,q) first difference forecast,
# ARIMA(p,2,q) second difference forecast
RWDfcst = RWD.predict(1, len(series))
for i in range(len(tseries) + 1, len(series) + 1):
    RWDseries[i - len(tseries) - 1] = fseries[i - len(tseries) - 1] + RWDfcst[i]

# Chart model forecast
fdata.plot(y=['Yt', 'RWD'])
plt.title('Random Walk with Drift ARIMA(0,1,0) Forecast')
plt.legend(loc='upper left')
plt.show()

#Random Walk with Drift ARIMA(0, 1, 0) information loss criteria for model selection and forecasting accuracy

#Model information loss criteria
print("")
print("Random Walk with Drift ARIMA(0,1,0) Information Loss Criteria:")
print("")
print("Akaike Information Criterion (AIC):", RWD.aic)
print("Bayesian Information Criterion (BIC):", RWD.bic)
print("Hannan-Quinn Information Criterion (HQIC):", RWD.hqic)
print("")

#Model forecasting accuracy

#Calculate Scale-Dependant Mean Absolute Error MAE
rwdmaelist = []
for i in range(0, len(fseries)):
    rwdmaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['RWD'][i - 1]))
RWDmae = np.mean(rwdmaelist)

# Calculate Scale-Independent Mean Absolute Percentage Error MAPE, Mean Absolute Scaled Error MASE
# using random walk as forecasting benchmark
rwdmapelist = []
rndmapelist = []
for i in range(0, len(fseries)):
    rwdmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['RWD'][i - 1])) / fdata['Yt'][i])
    rndmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['Ytf'][i - 1])) / fdata['Yt'][i])
RWDmape = np.mean(rwdmapelist) * 100
RNDmape = np.mean(rndmapelist) * 100

RWDmase = RWDmape / RNDmape

# Print forecasting accuracy comparison results
print("")
print("Random Walk with Drift ARIMA(0,1,0) Forecast Accuracy:")
print("")
print("Mean Absolute Error MAE:", RWDmae)
print("Mean Absolute Percentage Error MAPE:", RWDmape)
print("Mean Absolute Scaled Error MASE:", RWDmase)

