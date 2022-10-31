import pandas as pd
import os
from requests import get
from bs4 import BeautifulSoup
import re

def advanced_stats(season: int, slugs: bool=True, from_bbref: bool=False, one_obs_per_player:bool=True):
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
	filepath = os.path.join('data/adv_stats/', str(season) + '.csv')
	
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
			print(df)
			df.to_csv(filepath, index=True)
			print("Player stats for {} have been saved to disk.".format(season))
	
	df.set_index('playerID', inplace=True)
	if one_obs_per_player:
		return df[~df.index.duplicated(keep='first')]
	else:
		return df


def draft_class(season: int, slugs: bool=True, from_bbref: bool=False, persist=True):

	filepath = os.path.join('data/draft/', str(season) + '.csv')

	if os.path.exists(filepath) and not from_bbref:
		#print("Data for {} is being loaded from disk".format(season))
		df = pd.read_csv(filepath)
	else:
		r = get(f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url=%2Fdraft%2FNBA_{season}.html&div=div_stats')
		df = None
		if r.status_code == 200:
			soup = BeautifulSoup(r.content, 'html.parser')
			table = soup.find('table')
			df = pd.read_html(str(table))[0]
			df.drop(['Unnamed: 0_level_0'], inplace=True, axis = 1, level=0)
			df.rename(columns={'Unnamed: 1_level_0': '', 'Pk': 'PICK', 'Unnamed: 2_level_0': '', 'Tm': 'TEAM',
					  'Unnamed: 5_level_0': '', 'Yrs': 'YEARS', 'Totals': 'TOTALS', 'Shooting': 'SHOOTING',
					  'Per Game': 'PER_GAME', 'Advanced': 'ADVANCED', 'Round 1': '', 
					  'Player': 'PLAYER', 'College': 'COLLEGE'}, inplace=True)

			# flatten columns
			df.columns = ['_'.join(x) if x[0] != '' else x[1] for x in df.columns]

			# remove mid-table header rows
			df = df[df['PLAYER'].notna()]
			df = df[~df['PLAYER'].str.contains('Round|Player')]            
			if slugs:
			  cont = soup.findAll('tr')
			  # add slug
			  links = [[td.get('href') for td in cont[i].findAll('a')] for i in range(len(cont))]
			  slug = [re.findall("players/\w/(.*?).html", str(links[i])) for i in range(len(links))]
			  df_slugs = pd.DataFrame(list(slug), columns = ['playerID'])
			  df_slugs = df_slugs[df_slugs['playerID'].notna()].reset_index(drop=True)

			df = df.reset_index(drop=True).join(df_slugs, on=None)
			if persist:
				df.to_csv(filepath, index=True)
				print("Draft class for {} has been saved to disk.".format(season))

	df.set_index('playerID', inplace=True)
	return df

def get_historical(slug, up_to_season=None, playoffs=False, career=False, ask_matches=True):
	initial = slug[0]
	suffix = '/players/{}/{}.html'.format(initial, slug)
	if playoffs:
		selector = 'playoffs_'+selector
	r = get(f'https://widgets.sports-reference.com/wg.fcgi?css=1&site=bbr&url={suffix}&div=div_advanced')
	if r.status_code==200:
		soup = BeautifulSoup(r.content, 'html.parser')
		table = soup.find('table')
		if table is None:
			return pd.DataFrame()
		df = pd.read_html(str(table))[0]
		df.rename(columns={'Season': 'SEASON', 'Age': 'AGE',
				  'Tm': 'TEAM', 'Lg': 'LEAGUE', 'Pos': 'POS'}, inplace=True)
		if 'FG.1' in df.columns:
			df.rename(columns={'FG.1': 'FG%'}, inplace=True)
		if 'eFG' in df.columns:
			df.rename(columns={'eFG': 'eFG%'}, inplace=True)
		if 'FT.1' in df.columns:
			df.rename(columns={'FT.1': 'FT%'}, inplace=True)

		career_index = df[df['SEASON']=='Career'].index[0]
		if career:
			df = df.iloc[career_index+2:, :]
		else:
			df = df.iloc[:career_index, :]

		df = df.reset_index().drop('index', axis=1)
		return df
