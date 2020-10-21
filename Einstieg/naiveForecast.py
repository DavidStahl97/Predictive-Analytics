import pandas as pd
import math
from matplotlib import pyplot

ts = pd.read_csv("airline-passengers.csv", header=0, index_col=0)

nOfPassengers = ts["Passengers"]
values = nOfPassengers.values.tolist()

forecast = [math.nan]

for i in range(0, len(values) - 1):
    forecast.append(values[i])

pyplot.plot(values)
pyplot.plot(forecast)

print(values)
print(forecast)

me = 0
mae = 0
mse = 0
n = len(forecast)

for i in range(1, n):
    diff = values[i] - forecast[i]
    me = me + diff
    mae = mae + abs(diff)
    mse = mse + diff * diff

print(me / n, mae / n, mse / n)
pyplot.show()
