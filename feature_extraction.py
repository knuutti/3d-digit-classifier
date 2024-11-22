import numpy as np

def feature_extraction(data):

    features = []

    N = np.shape(data)[0]

    xy_dist_from_origin = [np.sum(data[i,0:2]**2)**0.5 for i in range(N)]
    features.append(np.max(xy_dist_from_origin)-np.min(xy_dist_from_origin)) # max-min distance from origin

    dist_end_to_start = np.linalg.norm(data[-1] - data[0])
    features.append(dist_end_to_start) # distance from end to start

    features.append(np.std(data[:,0], ddof=0)) # standard deviation of x
    features.append(np.std(data[:,1], ddof=0)) # standard deviation of y
    features.append(np.mean(data[:,0])) # average of x
    features.append(np.mean(data[:,1])) # average of y

    xy_dist_from_mean = [np.sum((data[i,0:2] - np.mean(data[i,0:2]))**2)**0.5 for i in range(N)]
    features.append(np.std(xy_dist_from_mean, ddof=0)) # deviation of distance from mean
    features.append(np.mean(xy_dist_from_mean)) # average distance from mean

    return features