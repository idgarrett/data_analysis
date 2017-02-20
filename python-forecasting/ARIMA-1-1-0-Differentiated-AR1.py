import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


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
# Create full range and training series
series = data['Yt']
tseries = tdata['Yt']

# Add new forecasting and model data columns with last training series value and create forecasting and model series
fdata['Ytf'] = tseries[-1]
fdata['DAR1'] = tseries[-1]
fseries = fdata['Ytf']
DAR1series = fdata['DAR1']

# 2. Differentiated First Order Autoregressive ARIMA(1, 1, 0) model calculation for training range and out of sample
# forecasting for forecasting range

# 2.1. Model calculation  and parameters printing for training range
DAR1 = ARIMA(series[:len(tseries) - 1], order=(1, 1, 0)).fit()
print("")
print("Differentiated First Order Autoregressive ARIMA(1, 1, 0) Parameters:")
print("")
print("Constant:", DAR1.params[0])
print("AR1:", DAR1.params[1])
print("")

# 2.2. Model forecasting for full range and then for only forecasting range

# Forecast at level of integration. ARIMA(p,0,q) level forecast, ARIMA(p,1,q) first difference forecast,
# ARIMA(p,2,q) second difference forecast
DAR1fcst = DAR1.predict(1, len(series))
for i in range(len(tseries) + 1, len(series) + 1):
    DAR1series[i - len(tseries) - 1] = fseries[i - len(tseries) - 1] + DAR1fcst[i]

# Chart model forecast
fdata.plot(y=['Yt', 'DAR1'])
plt.title('Differentiated First Order Autoregressive ARIMA(1, 1, 0) Forecast')
plt.legend(loc='upper left')
plt.show()

# 3. Differentiated First Order Autoregressive ARIMA(1, 1, 0) information loss criteria for model selection and
# forecasting accuracy

# 3.1. Model information loss criteria
print("")
print("Differentiated First Order Autoregressive ARIMA(1, 1, 0) Information Loss Criteria:")
print("")
print("Akaike Information Criterion (AIC):", DAR1.aic)
print("Bayesian Information Criterion (BIC):", DAR1.bic)
print("Hannan-Quinn Information Criterion (HQIC):", DAR1.hqic)
print("")

# 3.2. Model forecasting accuracy

# 3.2.1. Calculate Scale-Dependant Mean Absolute Error MAE
dar1maelist = []
for i in range(0, len(fseries)):
    dar1maelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['DAR1'][i - 1]))
DAR1mae = np.mean(dar1maelist)

# 3.2.2. Calculate Scale-Independent Mean Absolute Percentage Error MAPE, Mean Absolute Scaled Error MASE
# using random walk as forecasting benchmark
dar1mapelist = []
rndmapelist = []
for i in range(0, len(fseries)):
    dar1mapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['DAR1'][i - 1])) / fdata['Yt'][i])
    rndmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['Ytf'][i - 1])) / fdata['Yt'][i])
DAR1mape = np.mean(dar1mapelist) * 100
RNDmape = np.mean(rndmapelist) * 100

DAR1mase = DAR1mape / RNDmape

# Print forecasting accuracy comparison results
print("")
print("Differentiated First Order Autoregressive ARIMA(1, 1, 0) Forecast Accuracy:")
print("")
print("Mean Absolute Error MAE:", DAR1mae)
print("Mean Absolute Percentage Error MAPE:", DAR1mape)
print("Mean Absolute Scaled Error MASE:", DAR1mase)
