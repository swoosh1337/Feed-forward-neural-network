# Deep Neural Nets
This project involves training a feed-forward neural network with 3 hidden layers each with 64 neurons. The neural network will be trained to accurately classify images of fashion items into 10 different classes. The images used for training will be sourced from the Fashion MNIST dataset.The input to the network will consist of 28 × 28-pixel images, while the output will be a real number. Through this project, the aim is to successfully develop and implement a powerful neural network that can accurately classify fashion item images with high precision.

## Runing the project

To run this project, you will first need to download the required data using the following links:

<a href="https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_images.npy">Fashion MNIST Train Images</a> ("https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_images.npy")
<a href="https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_labels.npy">Fashion MNIST Train Labels</a> ("https://s3.amazonaws.com/jrwprojects/fashion_mnist_train_labels.npy")
<a href="https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_images.npy">Fashion MNIST Test Images</a> ("https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_images.npy")
<a href="https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_labels.npy">Fashion MNIST Test Labels</a> ("https://s3.amazonaws.com/jrwprojects/fashion_mnist_test_labels.npy")

Once you have downloaded the required data, you will need to install all the necessary libraries specified in the requirements.txt file.

To run the project, simply execute the main Python file main.py by running the following command in your terminal or command prompt:
python main.py
You can adjust the hyperparameters in the code as needed to experiment with different settings.

## Output
After running the main.py file, the program will output the accuracy of the trained neural network on the test dataset.
## Conclusion
This project is a simple yet effective example of how to train a feed-forward neural network to classify images of fashion items with high accuracy. By adjusting the hyperparameters and training on different datasets, this neural network can be further optimized to achieve even higher accuracy.
