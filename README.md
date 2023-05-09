# MNIST Digit Recognition using CNN
This is an implementation of a convolutional neural network (CNN) for the task of digit recognition on the MNIST dataset.

### Dependencies
- Pandas 
- Numpy 
- Keras

To install the required dependencies, run the following command:
~
pip install -r requirements.txt
~

### Dataset
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (28x28 pixels) and their corresponding labels (0-9). The dataset is split into a training set of 60,000 images and a test set of 10,000 images. This dataset can be found which can be found [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/versions/2).


### Model Architecture
The model is a simple CNN with two convolutional layers followed by two max pooling layers, a flatten layer, and two dense layers (one hidden and one output). The architecture is as follows:

~~~~
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_______________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
_______________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
_______________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
_______________________
flatten (Flatten)            (None, 1600)              0         
_______________________
dense (Dense)                (None, 64)                102464    
_______________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 122,930
Trainable params: 122,930
Non-trainable params: 0
~~~~

### Results
After training the model on the training set for 10 epochs with a batch size of 64, the model achieved a test accuracy of 0.9891. Some sample test images and their corresponding predicted values are also visualized in the notebook [mnist_classification.ipynb]().

### Usage
To train the model and evaluate it on the test data, you can run the script [mnist_classification.py](mnist_classification.py), but first ensure that the required dependencies are installed [requirements.txt](requirements.txt). You can run this using the following command:
~~~
pip install -r requirements.txt
~~~

### Acknowledgements
The MNIST dataset is a widely-used dataset for machine learning, and it was originally created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
