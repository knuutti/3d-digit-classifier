import classifier as c
import torch
import numpy as np

# Classifier function, takes Nx3 matrix as an 
# input and returns an integer from range 0-9
def digit_classify(data):
    model = c.DigitClassifier()
    model.load_state_dict(torch.load('best_model.pth', weights_only=False))
    data = c.preprocess_data(data)
    M = c.image_creation(data)
    features = np.ravel(M)
    for f in c.feature_extraction(data):
        features = np.append(features, f)
    features = torch.tensor(np.matrix(features)).float()
    output = model(features)
    return torch.argmax(output).item()
