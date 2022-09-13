import re
import os
import numpy as np
import pandas as pd
from requests import get
from bs4 import BeautifulSoup

def grab_stats(season: int, slugs: bool=True, from_bbref: bool=False,one_obs_per_player:bool=True, verbose: bool=False):
    """
    Returns the BBRef "advanced stats" table corresponding to a given a season 
    where the rows are the players and the columns are the various metrics. 

    Args:
        season: the year of season end (e.g. 2012 corresponds to the regular season 2011-12) 
        slugs: Include unique player IDs as provided by BBref
        from_bbref: If set to True, will replace the files on disk by scraping BBref.

    Returns:
        Dataframe with 30 columns (Rk, Player, Pos, Age, Tm, G, MP, PER... etc.) 

    """
    filepath = os.path.join('data', str(season) + '.csv')
    
    if os.path.exists(filepath) and not from_bbref:
        #print("Data for {} is being loaded from disk".format(season))
        df = pd.read_csv(filepath)
    else:
        r = get(
            f'https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html')
        df = None
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, 'html.parser')
            table = soup.find('table')

            df = pd.read_html(str(table))[0]
            df = df[df["Rk"].str.contains("Rk")==False]
            if slugs:

              cont = soup.findAll('tr')
              # add slug
              links = [[td.get('href') for td in cont[i].findAll('a')] for i in range(len(cont))]
              slug = [re.findall("players/\w/(.*?).html", str(links[i])) for i in range(len(links))]
              df_slugs = pd.DataFrame(list(slug[1:]), columns = ['playerID'])

            df = df.join(df_slugs)
            df.to_csv(filepath, index=True)
            print("Player stats for {} have been saved to disk.".format(season))
    
    df.set_index('playerID', inplace=True)
    if one_obs_per_player:
        return df[~df.index.duplicated(keep='first')]
    else:
        return df

def get_unique_values(df, col):
    r = []
    l = df[col].tolist()
    for i in l:
        if l not in r:
            r.append(i)
    return r


def create_xy(sample_start, sample_end, metric, show_plots=False):
    Y = []
    X = []
    for season in range(sample_start, sample_end):

        df = grab_stats(season)
        active_players = list(df.index.values)
        y = df[metric].astype(float).tolist()
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
