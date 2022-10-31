import argparse
import numpy as np
import pandas as pd
from sklearn import linear_model

from preprocessor import create_y, create_X, drop_nas

parser = argparse.ArgumentParser('NBA Player Projections')
parser.add_argument('-m', '--metric', type=str, default="BPM", help='Advanced stat to be forecasted.')
parser.add_argument('-s', '--season', type=int, default=2023, help='Season to be forecasted.\
    Note: 2023 indicates season ending in 2023, i.e. 2022-23.')
parser.add_argument('-t', '--test', type=str, help='Enable cross validation.')
args = vars(parser.parse_args())

def createXY(start, end):
    """ We need the "No_nas" option for inference"""
    y, X, plist = [], [], []
    for s in range(start, end):
        sy, plist = create_y(s, m, show_plots=False, MP_min=minimum_playing_time)
        sX = create_X(s, plist, xs)
        y.extend(sy)
        X.extend(sX)
    y, X, dropped = drop_nas(np.array(y), np.array(X))
    return y, X, plist, dropped

s = args['season']
m = args["metric"]
xs = [(m, 1), ('MP', 1), ('Age', 1)]
minimum_playing_time=0

y, X, _, _ = createXY(1990, s)
_, Xp, players, dropidx = createXY(s, s+1)
reg = linear_model.LinearRegression()
fittedreg = reg.fit(X, y)
preds = reg.predict(Xp)

no_pred = []
for i in sorted(dropidx, reverse=True):
    no_pred.append(players[i])
    del players[i]

predicted = pd.Series(preds, index=players)
not_predicted = pd.Series(np.nan, index=no_pred)
f = predicted.append(not_predicted).sort_index()
print("Predicted: {}, Not: {}".format(len(predicted), len(not_predicted)))
print(f)
# show forecast in line bar