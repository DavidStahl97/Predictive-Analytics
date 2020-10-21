import pandas as pd
from matplotlib import pyplot

# read csv into a dataframe
ts = pd.read_csv("airline-passengers.csv", header=0, index_col=0)
rolling = ts.rolling(window=10)
rolling_mean = rolling.mean()
rolling_mean.plot()

seasonal = ts["Passengers"] - rolling_mean

pyplot.show()
