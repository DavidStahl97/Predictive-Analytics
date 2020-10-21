import pandas as pd
from matplotlib import pyplot
import numpy as np

# read csv into a dataframe
ts = pd.read_csv("airline-passengers.csv", header=0, index_col=0)

# extract column with passangers and convert to list
passengers = ts["Passengers"].values.tolist()

end = len(passengers)

mean = np.mean(passengers)
variance = np.var(passengers)
n = len(passengers)

acf = []

for tau in range(end - 1):
    sumcov = 0
    for i in range (tau + 1, len(passengers)):
        sumcov = sumcov + (passengers[i]-mean) * (passengers[i - tau] - mean)
    acf.append(sumcov / variance * (n - 1))

pyplot.stem(acf)
pyplot.show();
