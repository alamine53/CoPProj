import numpy as np
import matplotlib.pyplot as plt

from bbref import advanced_stats
from models import LinearModel, NaiveModel, MovingAvgModel
import argparse

parser = argparse.ArgumentParser('NBA Player Projections')
parser.add_argument('metric', type=str, help='Advanced stat to be forecasted.')
parser.add_argument('-t', '--test', type=str, help='Enable cross validation.')
args = vars(parser.parse_args())

metric = args['metric']
print("Forecasting {}".format(metric))
xvars1 = [(metric, 1)]
xvars2 = [(metric, 1), (metric, 2)]
xvars3 = [(metric, 1), (metric, 2), (metric, 3)]
xvars4 = [(metric, 1), ('MP', 1), ('Age', 1)]
xvars5 = [(metric, 1), (metric, 2), ('MP', 1), ('MP', 2), ('Age', 1)]

test_seasons = list(range(2010, 2022))
fcast_season = 2022

def n_lags(yvar, xvars):
    nlags = 0
    ctrls = []
    for x in xvars:
        if x[0] == yvar:
            nlags += 1
        else:
            ctrls.append(x[0])
    return nlags, ctrls

for xvars in [xvars1, xvars2, xvars3, xvars4, xvars5]:
    lags, ctrls = n_lags(metric, xvars)
    if len(ctrls) == 0:
        print("Linear model with {} lag(s)".format(lags))
    else:
        print("Linear model with {} lag(s) and {} controls: {}".format(lags, len(ctrls), ctrls))

    model = LinearModel(metric, xvars,  restrict_sample=True)
    fcast = model.forecast(fcast_season)
    rmse_by_season = model.cross_val(test_seasons)
    print("RMSE: {}".format(round(sum(rmse_by_season)/len(rmse_by_season),2)))

print("Naive model:")
naive = NaiveModel(metric, restrict_sample=True)
rmse_by_season = naive.cross_val(test_seasons)
print("Average RMSE: {}".format(round(sum(rmse_by_season)/len(rmse_by_season),2)))
