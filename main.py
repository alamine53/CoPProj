import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error 

from utils import grab_stats, drop_nas, create_xy
from models import VORP_L1, VORP_L2, VORP_L3, BPM_L1, BPM_L2, Naive, TIME_WA3
# from models import create_xy

def load_data(start, end):
    for season in range(start, end):
        df = grab_stats(season, from_bbref=True)
        print(df)

def plt_dist(data, ctitle, ax=None):
    plt.hist(data, bins=30, alpha=0.7, ec="black")
    plt.title(ctitle)
    plt.show()

y1  = 1990
metric = 'BPM'

for model in [VORP_L1, VORP_L2, VORP_L3, BPM_L1, BPM_L2]:
    rmse_list = []
    for y2 in range(2010, 2023):

        y, X = model.create_xy(y1, y2, metric, no_nas=True)

        reg = linear_model.LinearRegression()

            # =================== here we would tune hyperparams ===================== #
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # reg.fit(X_train, y_train)
            # ======================================================================== #

        reg.fit(X, y) # over entire sample

        y_val, X_val = model.create_xy(y2, y2+1, metric, no_nas=True)
        y_pred = reg.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred)
        # print("Preds:", reg.predict(X_val))
        # print("%2d RMSE: %5.2f" % (y2, rmse))
        rmse_list.append(rmse)

    print("Average RMSE ({}): {}".format(repr(model()), round(sum(rmse_list)/len(rmse_list),2)))

# Naive forecast
for m in ['BPM', 'VORP']:
    rmse_naive = []
    for season in range(2010, 2023):

        y_val, y_pred = Naive.forecast(season, m)

        rmse = mean_squared_error(y_val, y_pred)
        # print("%2d RMSE: %5.2f" % (season, rmse))
        rmse_naive.append(rmse)

    print("Average RMSE (Naive {}): {}".format(m, round(sum(rmse_naive)/len(rmse_naive),2)))

# forecast time played
for t in ['MP', 'G']:
    rmse_time = []
    for season in range(2010, 2023):
        y_val, y_pred = TIME_WA3.forecast(season, t)
        rmse = mean_squared_error(y_val, y_pred)
        # print("%2d RMSE: %5.2f" % (season, rmse))
        rmse_time.append(rmse)

    print("Average RMSE ({}): {}".format(t, round(sum(rmse_time)/len(rmse_time),2)))
# scores = cross_val_score(reg, X, y, cv=5)
# print(scores)