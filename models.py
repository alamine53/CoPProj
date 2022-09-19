import numpy as np
from bbref import advanced_stats
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error 

def plt_dist(data, ctitle, ax=None):
    plt.hist(data, bins=30, alpha=0.7, ec="black")
    plt.title(ctitle)
    plt.show()

def drop_nas(y, X, verbose=False):
    if verbose:
        print("Before dropping NAs: \n===================")
        print(y.shape)
        print(X.shape)
    to_drop = []
    for idx, i in enumerate(X):
        if (np.isnan(i).any()):
            to_drop.append(idx)

    y_no_nas = np.delete(y, to_drop)
    X_no_nas = np.delete(X, to_drop, 0)
    if verbose:
        print("After dropping NAs: \n===================")
        print(y.shape)
        print(X.shape)
    return y_no_nas, X_no_nas

def create_y(season, metric, show_plots):
    df = advanced_stats(season)
    active_players = list(df.index.values)
    y = df[metric].astype(float).tolist()
    if show_plots:
        plt_dist(y, "{} distribution in {}".format(metric, season))

    return y, active_players

def create_X(season: int, player_list: list, X_vars: list):
    """
    Xvarlist = [(metric, lag), (metric, lag), (metric, lag)....]
    e.g.
               [('BPM', 1), ('BPM', 2), ('MP', 3), ('Age', 0)]
    """
    X = []
    lags = [i[1] for i in X_vars]
    df = {}
    for l in lags:
        df[l] = advanced_stats(season-l)

    for i in player_list:
        xlist = []
        
        for x in X_vars:
            try:
                x = df[x[1]][x[0]][i]
                xlist.append(x)
            except KeyError:
                xlist.append(np.nan)
        X.append(xlist)
    return X

class LinearModel:

    def __init__(self, yvar: str, xvars: list, sample_start: int=1990):
        self.yvar = yvar
        self.xvars = xvars
        self.sample_start = sample_start

    def create_xy(self, start, end, no_nas=True):
        y, X = [], []
        for s in range(start, end):
            sy, plist = create_y(s, self.yvar, show_plots=False)
            sX = create_X(s, plist, self.xvars)
            y.extend(sy)
            X.extend(sX)
        if no_nas:
            return drop_nas(np.array(y), np.array(X), verbose=False)
        else:
            return np.array(y), np.array(X)

    def estimate(self, start, end):
        y, X = self.create_xy(start, end, no_nas=True)
        reg = linear_model.LinearRegression()
        return reg.fit(X, y)

    def forecast(self, s):
        reg = self.estimate(self.sample_start, s)
        yval, Xval = self.create_xy(s, s+1)
        return reg.predict(Xval)

    def cross_val(self, seasons):
        rmse = []
        for s in seasons:
            reg = self.estimate(self.sample_start, s)
            yval, Xval = self.create_xy(s, s+1)
            yhat = reg.predict(Xval)
            rmse.append(mean_squared_error(yval, yhat))
        return rmse

class TIME_WA3:

    def forecast(season, metric="MP"):

        df = advanced_stats(season)
        active_players = list(df.index.values)
        y = df[metric].astype(float).tolist()
    
        df_l1 = advanced_stats(season-1)
        df_l2 = advanced_stats(season-2)
        yhat = []
        for i in active_players:
                try:
                    most_time = max([df_l1[metric][i], df_l2[metric][i]])
                    least_time = min([df_l1[metric][i], df_l2[metric][i]])
                    yhat.append((2*most_time + least_time)/3)
                except KeyError:
                    yhat.append(np.nan)

        return drop_nas(y, yhat)

class Naive:
    """ Yhat at year t = Y at year t-1 """

    def forecast(season, metric):

        df = advanced_stats(season)
        active_players = list(df.index.values)
        y = df[metric].astype(float).tolist()
    
        df_l1 = advanced_stats(season-1)
        y_hat = []
        for i in active_players:
                try:
                    y_hat.append(df_l1[metric][i])
                except KeyError:
                    y_hat.append(np.nan)

        return drop_nas(y, y_hat)

if __name__ == "__main__":

    import random

    test_year = random.choice(range(2000, 2022))    

    # y, plist = create_y(test_year, 'BPM', show_plots=False)
    
    xvars1 = [('BPM', 1), ('BPM', 2)]
    xvars2 = [('VORP', 1), ('VORP', 2), ('VORP', 3)]
    # x = create_X(test_year, plist, xvars)
    # print(x)

    model = ForecastModel(2012, 2018, 'VORP', xvars2)
    y, X = model.create_xy(no_nas=True)
    print(y, X)