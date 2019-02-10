
import numpy as np 


def empirical_correlation(x, y):
    xc = x - np.mean(x, axis=0)
    yc = y - np.mean(y, axis=0)
    corr = np.mean(xc*yc, axis=0)/(np.std(x, axis=0)*np.std(y, axis=0))
    return corr

def generate_its(data, lags):
    its = []
    for lag in lags: 
        if type(data) is list:
            x0 = np.concatenate([item[:-lag] for item in data])
            x1 = np.concatenate([item[lag:] for item in data])
        else:
            x0 = data[:-lag]
            x1 = data[lag:]
        ts = empirical_correlation(x0, x1)
        its.append(-lag/np.log(np.abs(ts)))
    
    return np.array(its)


def empirical_gram_schmidt(x, y):
    return np.mean(x*y, axis=0)/np.mean(y*y, axis=0)