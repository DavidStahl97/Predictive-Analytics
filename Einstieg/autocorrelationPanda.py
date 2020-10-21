import pandas as pd
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

# read csv into a dataframe
ts = pd.read_csv("airline-passengers.csv", header=0, index_col=0)

# extract column with passengers
nOfPassengers = ts["Passengers"]

lag = 100

values = pd.DataFrame(nOfPassengers.values)
dataframe = pd.concat([values.shift(lag), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.corr())


# pd.plotting.lag_plot(nOfPassengers, lag=lag)
pd.plotting.autocorrelation_plot(nOfPassengers)
pyplot.show()
