import numpy as np
from bbref import advanced_stats

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
    return y_no_nas, X_no_nas, to_drop

def create_y(season: int, metric: str, show_plots: bool = False, MP_min: int = 0):
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
    if MP_min:
        df = df[df['MP'] > MP_min]
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
