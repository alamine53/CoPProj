import numpy as np
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error 

from utils import plt_dist


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
            sy, plist = create_y(s, self.yvar, show_plots=False, sample=self.restrict_sample)

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

    def actual(self, s):

        yval, Xval = self.create_xy(s, s+1)
        return yval

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
