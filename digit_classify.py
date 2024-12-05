import classifier as c
import torch
import numpy as np
import pandas as pd

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


if __name__ == "__main__":

    # Testing with all data samples
    correct = 0
    print("Starting to classify...")
    for num in range(10):
        for sample in range(1,101):
            fname = f'data/stroke_{num}_{str(0)*(4-len(str(sample))) + str(sample)}.csv'
            testdata = pd.read_csv(fname).to_numpy()
            print(np.shape(testdata))
            predicted = digit_classify(testdata)
            if num == predicted:
                correct += 1
    print(f"Accuracy: {correct/1000:.2f}")

