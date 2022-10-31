import matplotlib.pyplot as plt 
import pandas as pd

def line_chart(proj, historical, ax=None):
	if ax is None:
		ax = plt.subplots()[1]

	y1 = historical
	y2 = proj
	
	
