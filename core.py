import numpy as np
import matplotlib.pyplot as plt

from bbref import advanced_stats
from models import LinearModel, Naive
import argparse

parser = argparse.ArgumentParser('NBA Player Projections')

parser.add_argument('metric', type=str, help='Advanced stat to be forecasted.')

args = vars(parser.parse_args())

metric = args['metric']
print("Forecasting {}".format(metric))
xvars1 = [(metric, 1)]
xvars2 = [(metric, 1), (metric, 2)]
xvars3 = [(metric, 1), (metric, 2), (metric, 3)]
test_seasons = list(range(2010, 2022))
fcast_season = 2022

for xvars in [xvars1, xvars2, xvars3]:
    model = LinearModel(metric, xvars)
    fcast = model.forecast(fcast_season)
    rmse_by_season = model.cross_val(test_seasons)
    print("Average RMSE: {}".format(round(sum(rmse_by_season)/len(rmse_by_season),2)))


# # Naive forecast
# rmse_naive = []
# for season in range(2010, 2023):

#     y_val, y_pred = Naive.forecast(season, metric)

#     rmse = mean_squared_error(y_val, y_pred)
#     # print("%2d RMSE: %5.2f" % (season, rmse))
#     rmse_naive.append(rmse)

# print("Average RMSE (Naive {}): {}".format(metric, round(sum(rmse_naive)/len(rmse_naive),2)))

# # forecast time played
# for t in ['MP', 'G']:
#     rmse_time = []
#     for season in range(2010, 2023):
#         y_val, y_pred = TIME_WA3.forecast(season, t)
#         rmse = mean_squared_error(y_val, y_pred)
#         # print("%2d RMSE: %5.2f" % (season, rmse))
#         rmse_time.append(rmse)

#     print("Average RMSE ({}): {}".format(t, round(sum(rmse_time)/len(rmse_time),2)))
# # scores = cross_val_score(reg, X, y, cv=5)
# # print(scores)