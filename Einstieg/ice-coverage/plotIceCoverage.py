import pandas as pd
from matplotlib import pyplot

# read csv into a dataframe
ts = pd.read_csv("sea_ice.csv", header=0, index_col=0)

arctic = ts["Arctic"]
antarctica = ts["Antarctica"]

# plot
fig = pyplot.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(arctic, label='Arctic')
ax.plot(antarctica, label='Antarctica')

ax.legend()

pyplot.show()
