import numpy as np
from stroke_normalization import normalize_sample

def preprocess_data(data):

    # Normalize the data
    data = normalize_sample(data)

    # Remove data points that are too close to each other
    data_points = [data[0,:]]
    for measurement in data[1:]:
        if np.linalg.norm(measurement[0:2] - data_points[-1][0:2]) > 0.05:
            data_points.append(measurement)

    data = normalize_sample(np.array(data_points))

    return data