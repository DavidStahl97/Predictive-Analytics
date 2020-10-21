import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA

# read csv into a dataframe
ts = pd.read_csv("airline-passengers.csv", header=0, index_col=0)

# extract column with passengers
nOfPassengers = ts["Passengers"]

model = ARIMA(nOfPassengers, order=(2,1,2))
results = model.fit(disp=-1)

# given data
pyplot.plot(nOfPassengers)

# estimated differences
pyplot.plot(results.fittedvalues, color='red')

# cumulated sum = prediction for time series
predictions = results.fittedvalues.cumsum()

pyplot.plot(predictions)
pyplot.show()

results.plot_predict(1,264)
pyplot.show()