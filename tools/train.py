import joblib
from models import create_xy
import argparse
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

import matplotlib.pyplot as plt

def scatter_plot(x, y, x_title=None, y_title=None, ctitle=None, annos=None, ax=None):
	if ax is None:
		ax = plt.subplots()[1]
	
	if annos:
		ax.text(0, 10, annos, fontsize=8)

	ax.scatter(x, y, alpha = 0.4)
	ax.set_xlabel(x_title)
	ax.set_ylabel(y_title)
	ax.set_ylim(-6, 15)
	ax.set_xlim(-6, 15)
	ax.set_title(ctitle)
	ax.grid(which='major', axis='both')
	#plt.gca().set_aspect('equal', adjustable='box')

yvar = 'BPM'
xvars = [(yvar,1), ('MP', 1), ('Age',1)]
model_dir = 'models/'
model1_name = os.path.join(model_dir, f'{yvar}_full.mdl')
model2_name = os.path.join(model_dir, f'{yvar}_1000.mdl')
sample_start = 1990

#m1 = LinearModel()
m2 = GradientBoostingRegressor()
m = m2

ncols = 3
nrows = 4
fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(8,8))
i,j=0,0
for season in list(range(2010, 2022)):
	
	y,X = create_xy(sample_start, season, yvar, xvars, rel_players=True)
	yval, Xval = create_xy(season, season+1, yvar, xvars, rel_players=True)
	reg = m.fit(X, y)
	ypreds = reg.predict(Xval)
	print(season, len(ypreds), len(yval))

	rmse = mean_squared_error(ypreds, yval)
	r2 = r2_score(ypreds, yval)
	metrics = f"RMSE= {str(round(rmse,1))}\nR2={str(round(r2,1))}"
		
	scatter_plot(ypreds, yval, ctitle=str(season), annos=metrics, ax=ax[i][j])
	j += 1
	if j == nrows - 1:
		j = 0
		i += 1

fig.supxlabel("True")
fig.supylabel("Forecast")
plt.show()
