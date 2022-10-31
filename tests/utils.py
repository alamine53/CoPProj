import matplotlib.pyplot as plt

def plt_dist(data, ctitle, ax=None):
    plt.hist(data, bins=30, alpha=0.7, ec="black")
    plt.title(ctitle)
    plt.show()
