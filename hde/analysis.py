
import numpy as np 


def empirical_correlation(x, y):
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    corr = np.mean(xc*yc)/(np.std(x)*np.std(y))
    return corr


def empirical_gram_schmidt(x, y):
    return np.mean(x*y, axis=0)/np.mean(y*y, axis=0)