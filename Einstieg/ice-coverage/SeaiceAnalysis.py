import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA

ts = pd.read_csv("sea_ice.csv", header=0, index_col=0)

arcticIce = ts["Arctic"]
antarcticIce = ts["Antarctica"]

pyplot.plot(arcticIce)
pyplot.plot(antarcticIce)

pyplot.show()

x = arcticIce - antarcticIce
seaIceDF = pd.concat([arcticIce, antarcticIce, x], axis=1)

seaIceDF.columns = ["Arctic", "Antarctic", "Arctic-Antarctic"]

print(seaIceDF)


# your code goes from here...
def rolling(window):
    rolling_model = seaIceDF.rolling(window=window)
    rolling_mean = rolling_model.mean()
    rolling_mean.plot()


rolling(11)
pyplot.show()

rolling(60)
pyplot.show()

# ACF
pd.plotting.autocorrelation_plot(seaIceDF["Arctic"]).set_xlim([0, 20])
pyplot.show()

# ARIMA
arctic = seaIceDF["Arctic"]

model = ARIMA(arctic, order=(2, 1, 2))
results = model.fit(disp=-1)

# given data
pyplot.plot(arctic)

# estimated differences
pyplot.plot(results.fittedvalues, color='red')

# cumulated sum = prediction for time series
predictions = results.fittedvalues.cumsum()

pyplot.plot(predictions)
pyplot.show()

results.plot_predict(1, 264)
pyplot.show()
