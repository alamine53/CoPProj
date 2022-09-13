import numpy as np
from utils import grab_stats
# xgboost: 3 lags 
# xgboost: 2 lags and age
# naive model 1: value at t equal to value at t-1 (or t - 2 if t-1 doesn't exist)
# naive model 2: weighted moving average of last 3 years

# divide players into tiers (Pros > 3 years, Rookies = 1 year)

def drop_nas(y, X, verbose=True):
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

class A:

    def create_xy(start, end, metric, no_nas=True, show_plots=False):
        """ 3 lags of X """
        Y = []
        X = []
        for season in range(start, end):

            df = grab_stats(season)
            active_players = list(df.index.values)
            y = df[metric].tolist()
            Y.extend(y)
            if show_plots:
                plt_dist(y, "{} distribution in {}".format(metric, season))

            df_l1 = grab_stats(season-1)
            df_l2 = grab_stats(season-2)
            df_l3 = grab_stats(season-3)
            year_x = []
            for i in active_players:
                try:
                    year_x.append([df_l1[metric][i], df_l2[metric][i], df_l3[metric][i]])
                except KeyError:
                    year_x.append([np.nan, np.nan, np.nan])

            X.extend(year_x)
        if no_nas:
            return drop_nas(np.array(Y), np.array(X), verbose=False)
        else:
            return np.array(Y), np.array(X)


class B:

    def create_xy(start, end, metric, no_nas=True, show_plots=False):
        """ 3 lags of X """
        Y = []
        X = []
        for season in range(start, end):

            df = grab_stats(season)
            active_players = list(df.index.values)
            y = df[metric].tolist()
            Y.extend(y)
            if show_plots:
                plt_dist(y, "{} distribution in {}".format(metric, season))

            df_l1 = grab_stats(season-1)
            df_l2 = grab_stats(season-2)
            year_x = []
            for i in active_players:
                try:
                    year_x.append([df_l1[metric][i], df_l2[metric][i]])
                except KeyError:
                    year_x.append([np.nan, np.nan])

            X.extend(year_x)

        if no_nas:
            return drop_nas(np.array(Y), np.array(X), verbose=False)
        else:
            return np.array(Y), np.array(X)
