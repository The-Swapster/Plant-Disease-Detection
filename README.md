# Plant-Disease-Detection

For this project, we will create a sequential model for detecting plant species and the status of the plant

The project is broken down into multiple steps:

* Creating a feature extractor using InceptionV3
* Using the extracted features to create a model for classifying the images into their respected classes

## **Machine Learning  model using Tensorflow with Keras**

We designed algorithms and models to recognize species and diseases in the crop leaves by using Convolutional Neural Network

### Load the data
For this study we would download a public dataset of 54,305 images of diseased and healthy plant leaves collected under controlled conditions ([PlantVillage Dataset](https://storage.googleapis.com/plantdata/PlantVillage.tar)). The images cover 14 species of crops, including: apple, blueberry, cherry, grape, orange, peach, pepper, potato, raspberry, soy, squash, strawberry and tomato. It contains images of 17 basic diseases, 4 bacterial diseases, 2 diseases caused by mold (oomycete), 2 viral diseases and 1 disease caused by a mite. 12 crop species also have healthy leaf images that are not visibly affected by disease. Then store the downloaded zip file to the "/tmp/" directory.

Input data is resized to 229x229 pixels to be given as input to the InceptionV3 neural network.

### Data Preprocessing

Let's set up data generators that will read pictures in our source folders, convert them to `float32` tensors, and feed them (with their labels) to our network. 

As you may already know, data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network. (It is uncommon to feed raw pixels into a convnet.) In our case, we will preprocess our images by normalizing the pixel values to be in the `[0, 1]` range (originally all values are in the `[0, 255]` range).

### Build the model
All it takes is to put a linear classifier on top of the feature_extractor_layer with the Hub module.

For speed, we start out with a non-trainable feature_extractor_layer, but you can also enable fine-tuning for greater accuracy.

## CONCLUSION
* The plant village dataset is used for training the model
* InceptionV3 is used for feature extraction
* Sequential mode build for classification
* We achive a training accuracy of 89.53% and validation accuracy of 93.67%
