import numpy as np

def get_outliers(data, q=0.05):
        med = np.median(data, axis=0)
        dist = np.sqrt(np.sum((data - med)**2, axis=-1))
        n = int(data.shape[0]*q)
        return np.argpartition(-dist, n)[:n]
