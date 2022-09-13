import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error 

from utils import grab_stats, drop_nas, create_xy
from models import A, B
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

for model in [A, B]:
    rmse_list = []
    print("Model", model)
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
        print("%2d RMSE: %5.2f" % (y2, rmse))
        rmse_list.append(rmse)

    print("Average RMSE: %5.2f" % (sum(rmse_list)/len(rmse_list)))
# scores = cross_val_score(reg, X, y, cv=5)
# print(scores)