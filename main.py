from datetime import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot

DC_POWER = 2
# load dataset
def parser(x):
	return datetime.strptime(x, '%d-%m-%Y %H:%M')

data_series = read_csv('Plant_1_Generation_Data.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
features_name = data_series.columns.tolist()

print(features_name)

series = data_series[features_name[DC_POWER]]

series.index = series.index.to_period('D')

model = ARIMA(series)
model_fit = model.fit()

print(model_fit.summary())

residuals = DataFrame(model_fit.resid)

print(residuals.describe())



autocorrelation_plot(series)
pyplot.show()

autocorrelation_plot(residuals)
pyplot.show()

series.plot(kind = 'kde', title = 'Series de datos')
pyplot.show()

residuals.plot(kind='kde', title = 'Serie residual de datos')
pyplot.show()



# fig, axes = plt.subplots(3, 2, sharex=True)
# series.plot(kind = 'kde', title = 'Series de datos')
# plot_acf(series, ax=axes[0, 1])



