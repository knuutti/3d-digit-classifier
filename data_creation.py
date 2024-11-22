import numpy as np
import pandas as pd
from image_creation import image_creation
from feature_extraction import feature_extraction
from stroke_preprocess import preprocess_data

data_file = open('data.csv', 'w')
for num in range(0, 10):
    for sample in range(1,101):
        fname = f'data/stroke_{num}_{str(0)*(4-len(str(sample))) + str(sample)}.csv'
        data = pd.read_csv(fname).to_numpy()
        data = preprocess_data(data)
        M = image_creation(data)

        features = np.ravel(M)
        for f in feature_extraction(data):
            features = np.append(features, f)
        for f in features:
            data_file.write(str(f) + ',')
        data_file.write(str(num) + '\n')
data_file.close()