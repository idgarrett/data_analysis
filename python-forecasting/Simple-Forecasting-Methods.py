
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Add new forecasting series with last training series value and second to last for initial value
fdata['Ytf'] = tseries[-1]
fseries = fdata['Ytf']

# 2. Average AVG, Naive or Random Walk RW, Random Walk with Drift RWD and Seasonal Random Walk SRW forecasting

# 2.1. Average Method AVG forecasting

# Average Method AVG calculation for training range
AVG = np.mean(tseries)

# Average Method AVG calculation for forecasting range
for i in range(1, len(fseries)):
    fdata['AVG'] = AVG
print(fdata)

# Chart Average Method AVG Forecast
fdata.plot(y=['Yt', 'AVG'])
plt.title('Average Method AVG Forecast')
plt.legend(loc='upper left')
plt.show()

# 2.2. Naive or Random Walk Method RW forecasting

# Naive or Random Walk Method RW calculation for forecasting range with last data from training range
fdata['RW'] = fseries
print(fdata)

# Chart Naive or Random Walk Method RW Forecast
fdata.plot(y=['Yt', 'RW'])
plt.title('Random Walk Method RW Forecast')
plt.legend(loc='upper left')
plt.show()

# 2.3. Random Walk with Drift Method RWD forecasting

# Arithmetic mean calculation of training range time series differences
difftseries = tseries - tseries.shift()
difftseries[0] = 0.00
avgdifftseries = np.mean(difftseries)

# Random Walk with Drift Method RWD calculation for forecasting range with last data from training range plus
# arithmetic mean of training range time series differences
for i in range(1, len(fseries)):
    fdata['RWD'] = fseries + avgdifftseries
print(fdata)

# Chart Random Walk with Drift Method RWD Forecast
fdata.plot(y=['Yt', 'RWD'])
plt.title('Random Walk with Drift Method RWD Forecast')
plt.legend(loc='upper left')
plt.show()

# 2.4. Seasonal Random Walk Method SRW forecasting

# Seasonal Random Walk Method SRW calculation with last seasonal data from training range
def tseason(tseries, season):
    tseason = []
    for i in range(0, season):
        tseason.insert(i, tseries[i - season])
    return tseason

# Seasonal Random Walk Method SRW calculation for forecasting range with last seasonal data from training range
season = 5
SRW = tseason(tseries, season)
for i in range(season, len(fseries)):
    SRW.insert(i, SRW[i - season])
fdata['SRW'] = SRW
print(fdata)

# Chart Seasonal Random Walk Method SRW Forecast
fdata.plot(y=['Yt', 'SRW'])
plt.title('Seasonal Random Walk Method SRW Forecast')
plt.legend(loc='upper left')
plt.show()


# 3. Average AVG, Naive or Random Walk RW, Random Walk with Drift RWD and Seasonal Random Walk SRW forecasting accuracy

# 3.1. Calculate Scale-Dependant Mean Absolute Error MAE
avgmaelist = []
rwmaelist = []
rwdmaelist = []
srwmaelist = []
for i in range(0, len(fseries)):
    avgmaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['AVG'][i - 1]))
    rwmaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['RW'][i - 1]))
    rwdmaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['RWD'][i - 1]))
    srwmaelist.insert(i, np.absolute(fdata['Yt'][i] - fdata['SRW'][i - 1]))
AVGmae = np.mean(avgmaelist)
RWmae = np.mean(rwmaelist)
RWDmae = np.mean(rwdmaelist)
SRWmae = np.mean(srwmaelist)

# 3.2. Calculate Scale-Independent Mean Absolute Percentage Error MAPE
avgmapelist = []
rwmapelist = []
rwdmapelist = []
srwmapelist = []
for i in range(0, len(fseries)):
    avgmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['AVG'][i - 1])) / fdata['Yt'][i])
    rwmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['RW'][i - 1])) / fdata['Yt'][i])
    rwdmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['RWD'][i - 1])) / fdata['Yt'][i])
    srwmapelist.insert(i, (np.absolute(fdata['Yt'][i] - fdata['SRW'][i - 1])) / fdata['Yt'][i])
AVGmape = np.mean(avgmapelist) * 100
RWmape = np.mean(rwmapelist) * 100
RWDmape = np.mean(rwdmapelist) * 100
SRWmape = np.mean(srwmapelist) * 100

# Print forecasting accuracy comparison results
print("")
print("Mean Absolute Error MAE")
print("AVG:", AVGmae)
print("RW:", RWmae)
print("RWD:", RWDmae)
print("SRW:", SRWmae)

print("")
print("Mean Absolute Percentage Error MAPE")
print("AVG:", AVGmape)
print("RW:", RWmape)
print("RWD:", RWDmape)
print("SRW:", SRWmape)
