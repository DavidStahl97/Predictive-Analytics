import pandas as pd
from matplotlib import pyplot
import math

# read csv into a dataframe
ts = pd.read_csv("airline-passengers.csv", header=0, index_col=0)
# extract column with passengers and convert to list 
nOfPassengers = ts["Passengers"]
passengers = nOfPassengers.values.tolist()
# store values in mvAvg, Window 3, compute start and end
mvAvg = []
seasonalEffect = []
window = 37
start = math.floor(window / 2)
end = len(passengers) - start
# compute moving averages
for i in range(end):
    if i >= start:
        value = 0
        for j in range(i - start, i + start + 1):
            value = value + passengers[j]
        value = value / window
        se = value - passengers[i]
        mvAvg.append(value)
        seasonalEffect.append(se)

pyplot.plot(seasonalEffect)
pyplot.plot(mvAvg)
pyplot.show()
