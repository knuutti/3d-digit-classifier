## 3D Digit Classifier
In this project the goal was to design and train a classifier model for classifying hand written digits from 3D motion sensor data. 
The dataset used for training and validation consists of 1000 strokes, 100 samples for each digit. Each stroke contains $N$ number of 3D data points, that
together form the hand written stroke as a function of time.

This project was done as a practical assignment for course *Pattern Recognition and Machine Learning* lectured in LUT University in Autumn 2024.

## Model
Our model is a multimodal neural network, where we use both individual numerical features of the data that are passed to a dense layer, and a 2D image projection that is passed to convolutional layer. 
Our model achieved over 96% test accuracy on data that we did not have access to. During our own validation we got training accuracy as high as over 99% and test accuracy to 98%.

## Requirements
Required Python packages (install command with pip in parenthesis):
- numpy (`pip install numpy`)
- pandas (`pip install pandas`)
- torch (`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`)
- scipy (`pip install scipy`)

Additional packages required for training and testing the model (NOT required for running digit_classify function):
- sklearn (`pip install -U scikit-learn`)
- matplotlib (`pip install -U matplotlib`)
