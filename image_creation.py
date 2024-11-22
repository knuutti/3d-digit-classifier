import numpy as np

def image_creation(stroke, D):
    
    # This function assumes that the stroke is already normalized
    # xmin = 0, xmax = 1, ymin = 0, ymax = 1

    M = np.zeros([D,D])
    for i,point in enumerate(stroke[1:,0:2]):
        x0, y0 = stroke[i,0:2]
        x1, y1 = point
        x0 = int(x0*(D-1))
        y0 = int(y0*(D-1))
        x1 = int(x1*(D-1))
        y1 = int(y1*(D-1))

        dist = int(2*np.linalg.norm([x1-x0, y1-y0]))
        xspan = np.linspace(x0, x1, dist)
        yspan = np.linspace(y0, y1, dist)
        for i in range(dist):
            M[int(xspan[i]),int(yspan[i])] = 1
    return np.flipud(M.T)