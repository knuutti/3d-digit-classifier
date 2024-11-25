import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_augmentation():
    for num in range(10):
        for sample in range(1,101):
            stroke = pd.read_csv(f'data/stroke_{num}_{sample:04d}.csv').to_numpy()
            N = len(stroke)
            dx = np.max(stroke[:,0]) - np.min(stroke[:,0])
            dy = np.max(stroke[:,1]) - np.min(stroke[:,1])
            scale = max(dx, dy)/20
            for i in range(2):
                theta = np.pi/6*np.random.rand() - np.pi/12
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))
                angled_stroke = stroke.copy()
                angled_stroke[:,0:2] = np.dot(stroke[:,:2], R)
                noise = np.random.normal(0, scale, (N,3))
                noised_stroke = angled_stroke + noise
                file  = open(f'fake_data/stroke_{num}_{i}{sample:03d}.csv', 'w')
                for j in range(N):
                    file.write(f'{noised_stroke[j,0]},{noised_stroke[j,1]},{noised_stroke[j,2]}\n')
                file.close()
    return

data_augmentation()