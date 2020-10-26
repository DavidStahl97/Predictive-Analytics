import pandas as pd
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA

ts = pd.read_csv("sea_ice.csv", header=0, index_col=0)

arcticIce = ts["Arctic"]
antarcticIce = ts["Antarctica"]

seaIceDF = pd.concat([arcticIce, antarcticIce, arcticIce - antarcticIce], axis=1)

seaIceDF.columns = ["Arctic", "Antarctic", "Arctic-Antarctic"]
difference = seaIceDF["Arctic-Antarctic"]

print(seaIceDF)

model = ARIMA(difference, order=(2, 1, 2))
results = model.fit(disp=-1)

predictions = results.fittedvalues.cumsum()

results.plot_predict(1, 400)
pyplot.show()
