import pandas as pd
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

ts = pd.read_csv("sea_ice.csv", header=0, index_col=0)

arcticIce = ts["Arctic"]
antarcticIce = ts["Antarctica"]

seaIceDF = pd.concat([arcticIce, antarcticIce, arcticIce - antarcticIce], axis=1)

seaIceDF.columns = ["Arctic", "Antarctic", "Arctic-Antarctic"]
difference = seaIceDF["Arctic-Antarctic"]

print(seaIceDF)

"""
pyplot.plot(arcticIce)
pyplot.plot(antarcticIce)
pyplot.show()

pyplot.plot(seaIceDF["Arctic-Antarctic"])
pyplot.show()

pd.plotting.autocorrelation_plot(difference)
pyplot.show()
"""

shouldBeStationary = difference.diff().dropna()

result = adfuller(shouldBeStationary.values)

print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

plot_acf(shouldBeStationary, lags=20)
pyplot.show()
