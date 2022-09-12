import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from utils import grab_stats, get_unique_values

def load_data(start, end):
    for season in range(start, end):
        df = grab_stats(season, from_bbref=True)
        print(df)

def plt_dist(data, ctitle, ax=None):
    plt.hist(data, bins=30, alpha=0.7, ec="black")
    plt.title(ctitle)
    plt.show()

def create_xy(sample_start, sample_end, metric, show_plots=False):
    Y = []
    X = []
    for season in range(sample_start, sample_end):

        df = grab_stats(season)
        df_l1 = grab_stats(season-1)
        df_l2 = grab_stats(season-2)
        df_l3 = grab_stats(season-3)
        active_players = list(df.index.values)
        y = df[metric].tolist()
        Y.extend(y)

        if show_plots:
            plt_dist(y, "{} distribution in {}".format(metric, season))

        year_x = []
        for i in active_players:
            try:
                year_x.append([df_l1[metric][i], df_l2[metric][i], df_l3[metric][i]])
            except KeyError:
                year_x.append([np.nan, np.nan, np.nan])

        X.extend(year_x)
    return np.array(Y), np.array(X)

def drop_nas(y, X):
    """
    Delete rows in X and Y  with na's in X
    """
    to_drop = []
    for idx, i in enumerate(X):
        if (np.isnan(i).any()):
            to_drop.append(idx)

    y_no_nas = np.delete(y, to_drop)
    X_no_nas = np.delete(X, to_drop, 0)
    return y_no_nas, X_no_nas

y1  = 2000
y2 = 2022
metric = 'VORP'

y, X = create_xy(y1, y2, metric)
print("Before dropping NAs: \n===================")
print(y.shape)
print(X.shape)

y, X = drop_nas(y, X)
print("After dropping NAs: \n===================")
print(y.shape)
print(X.shape)

reg = linear_model.LinearRegression()
reg.fit(X, y)
print(reg.coef_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg.fit(X_train, y_train)
print(reg.score(X_test, y_test))

scores = cross_val_score(reg, X, y, cv=5)
print(scores)
