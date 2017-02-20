import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


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
# Create full range and training series
series = data['Yt']
tseries = tdata['Yt']

# Add new forecasting and model data columns with last training series value and create forecasting and model series
fdata['Ytf'] = tseries[-1]
fdata['HOLT'] = tseries[-1]
fseries = fdata['Ytf']
HOLTseries = fdata['HOLT']

# 2. Holt's Linear Trend ARIMA(0, 2, 1) model calculation for training range and
# out of sample forecasting for forecasting range

# 2.1. Model calculation  and parameters printing for training range
HOLT = ARIMA(series[:len(tseries) - 1], order=(0, 2, 1)).fit()
print("")
print("Holt Linear Trend ARIMA(0, 2, 1) Parameters:")
print("")
print("Constant:", HOLT.params[0])
print("MA1:", HOLT.params[1])
print("")

# 2.2. Model forecasting for full range and then for only forecasting range

# Forecast at level of integration. ARIMA(p,0,q) level forecast, ARIMA(p,1,q) first difference forecast,
# ARIMA(p,2,q) second difference forecast
HOLTfcst = HOLT.predict(2, len(series))
difftseries = tseries - tseries.shift()
difftseries[0] = 0
for i in range(len(tseries) + 1, len(series) + 1):
    HOLTseries[i - len(tseries) - 1] = (HOLTseries[i - len(tseries) - 2] + HOLT.params[1] * np.mean(difftseries)
                                       + HOLTfcst[i])

# Chart model forecast
fdata.plot(y=['Yt', 'HOLT'])
plt.title('Holt Linear Trend ARIMA(0, 2, 1) Forecast')
plt.legend(loc='upper left')
plt.show()

# 3. Holt's Linear Trend ARIMA(0, 2, 1) information loss criteria for model selection and
# forecasting accuracy

# 3.1. Model information loss criteria
print("")
print("Holt Linear Trend ARIMA(0, 2, 1) Information Loss Criteria:")
print("")
print("Akaike Information Criterion (AIC):", HOLT.aic)
print("Bayesian Information Criterion (BIC):", HOLT.bic)
print("Hannan-Quinn Information Criterion (HQIC):", HOLT.hqic)
print("")

# 3.2. Model forecasting accuracy

# 3.2.1. Calculate Scale-Dependant Mean Absolute Error MAE
holtmaelist = []
for i in range(0, len(fseries)):
    holtmaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['HOLT'][i - 1]))
HOLTmae = np.mean(holtmaelist)

# 3.2.2. Calculate Scale-Independent Mean Absolute Percentage Error MAPE, Mean Absolute Scaled Error MASE
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
print("Holt Linear Trend ARIMA(0, 2, 1) Forecast Accuracy:")
print("")
print("Mean Absolute Error MAE:", HOLTmae)
print("Mean Absolute Percentage Error MAPE:", HOLTmape)
print("Mean Absolute Scaled Error MASE:", HOLTmase)

