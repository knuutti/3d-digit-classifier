import numpy as np

def normalize_sample(sample):
    d = sample
    minX = np.min(d[:,0])
    maxX = np.max(d[:,0])
    minY = np.min(d[:,1])
    maxY = np.max(d[:,1])
    d[:,0] = (d[:,0] - minX) / (maxX - minX)
    d[:,1] = (d[:,1] - minY) / (maxY - minY)   
    return d 