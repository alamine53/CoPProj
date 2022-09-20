import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error 

from bbref import advanced_stats
from utils import plt_dist

def drop_nas(y, X, verbose=False):
    """
    Drop NaN values from two arrays (y, and X) based on the values
    of X.
    """
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

def create_y(season: int, metric: str, show_plots: bool = False, relevant_players_only: bool = False):
    """
    Create labels and relevant player list for/from a given season.
    
    Args:
        season: season for which we want to train / predict. 
        metric: list of players which we care about
        show_plots: Display distribution of y values in a histogram
        relevant_players_only: Will filter down to the players with more than
            1,000 minutes played per season

    Returns:
        y: list of labels
        active_players: list of active players
    """
    df = advanced_stats(season)
    if relevant_players_only:
        df = df[df['MP'] > 1000]
    active_players = list(df.index.values)
    y = df[metric].astype(float).tolist()
    if show_plots:
        plt_dist(y, "{} distribution in {}".format(metric, season))

    return y, active_players

def create_X(season: int, player_list: list, X_vars: list):
    """
    Create a feature vector corresponding to a player list from 
    a given season. 
    
    Args:
        season: season for which we want to train / predict. 
        player_list: list of players which we care about
        X_vars = list of shape: [(metric, lag), (metric, lag), (metric, lag)....]
                where "metric" is the name of the variable as from the BBref dataset 
                and lag is the correspong lag year. 
                e.g. [('BPM', 1), ('BPM', 2), ('MP', 3), ('Age', 0)]

    Returns:
        X: List of features corresponding to the order of players.
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
    """
    Linear regression model of the form Y = aX + b
    where X is a feature vector of lags and controls. 
    """
    def __init__(self, yvar: str, xvars: list, restrict_sample: bool=False, sample_start: int=1990):
        self.yvar = yvar
        self.xvars = xvars
        self.sample_start = sample_start
        self.restrict_sample = restrict_sample

    def create_xy(self, start, end, no_nas=True):
        y, X = [], []
        for s in range(start, end):
            sy, plist = create_y(s, self.yvar, show_plots=False, relevant_players_only=self.restrict_sample)

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

class NaiveModel:
    """
    Forecasts y at year t as equal to the value of 
    y at year t minus 1. This model is meant as a baseline
    for comparison against more advanced models. 
    """
    def __init__(self, yvar, restrict_sample=False):
        self.yvar = yvar
        self.restrict_sample = restrict_sample
    
    def forecast(self, season):
        y, active_players = create_y(season, self.yvar, show_plots=False, relevant_players_only=self.restrict_sample)
        df_l1 = advanced_stats(season-1)
        yhat = []
        for i in active_players:
            try:
                yhat.append(df_l1[self.yvar][i])
            except KeyError:
                yhat.append(np.nan)

        return drop_nas(y, yhat)
    
    def cross_val(self, seasons):
        rmse = []
        for s in seasons:
            yval, ypred = self.forecast(s)
            rmse.append(mean_squared_error(yval, ypred))
        return rmse


class MovingAvgModel:
    """
    Forecasts y at year t as a moving average
    of y at years t-1, t-2... etc.
    """
    def __init__(self, yvar: str, lags: int):
        self.yvar = yvar
        self.lags = lags

    def forecast(self, season):
        y, active_players = create_y(season, self.yvar, show_plots=False)
        df = {}
        for lag in range(self.lags):
            df[lag + 1] = advanced_stats(season-lag-1)

        yhat = []
        for i in active_players:
            xs = []
            for lag in range(self.lags):
                try:
                    xs.append(df[season-lag-1][i])
                except KeyError:
                    xs.append(None)

            xs = [x for x in xs if x]
            avg = sum(xs)/len(xs)
            yhat.append(avg)
        return drop_nas(y, yhat)

    
    def cross_val(self, seasons):
        rmse = []
        for s in seasons:
            yval, ypred = self.forecast(s)
            rmse.append(mean_squared_error(yval, ypred))
        return rmse