import numpy as np
import matplotlib.pyplot as plt

from bbref import advanced_stats
from models import LinearModel, NaiveModel, MovingAvgModel
import argparse

parser = argparse.ArgumentParser('NBA Player Projections')
parser.add_argument('-m', '--metric', type=str, default="BPM", help='Advanced stat to be forecasted.')
parser.add_argument('-s', '--season', type=int, default=2022, help='Season to be forecasted.\
    Note: 2023 indicates season ending in 2023, i.e. 2022-23.')
parser.add_argument('-t', '--test', type=str, help='Enable cross validation.')
args = vars(parser.parse_args())

metric = args['metric']
fcast_season = args['season']

x1 = [(metric, 1)]
x2 = [(metric, 1), (metric, 2)]
x3 = [(metric, 1), (metric, 2), (metric, 3)]
x4 = [(metric, 1), ('MP', 1), ('Age', 1)]
x5 = [(metric, 1), (metric, 2), ('MP', 1), ('MP', 2), ('Age', 1)]

test_seasons = list(range(2010, 2022))

def n_lags(yvar, xvars):
    nlags = 0
    ctrls = []
    for x in xvars:
        if x[0] == yvar:
            nlags += 1
        else:
            ctrls.append(x[0])
    return nlags, ctrls

for xvars in [x1, x2, x3, x4, x5]:
    lags, ctrls = n_lags(metric, xvars)
    if len(ctrls) == 0:
        print("Linear model with {} lag(s)".format(lags))
    else:
        print("Linear model with {} lag(s) and {} controls: {}".format(lags, len(ctrls), ctrls))

    model = LinearModel(metric, xvars,  restrict_sample=True)
    plist, fcast = model.forecast(fcast_season)
    print(len(plist), len(fcast))

    
#     rmse_by_season = model.cross_val(test_seasons)
#     print("RMSE: {}".format(round(sum(rmse_by_season)/len(rmse_by_season),2)))

# print("Naive model:")
# naive = NaiveModel(metric, restrict_sample=True)
# rmse_by_season = naive.cross_val(test_seasons)
# print("Average RMSE: {}".format(round(sum(rmse_by_season)/len(rmse_by_season),2)))
