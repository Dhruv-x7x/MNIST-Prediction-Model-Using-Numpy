# MNIST-Prediction-Model-Using-Numpy

The implementation for the forward prop and backward prop is not my original idea. I followed Samson Zhang's [video](https://www.youtube.com/watch?v=w8yWXqWQYmU). I improved upon his model by using Leaky ReLU and He's initilization. This project was done purely to learn how exactly neural networks learn from scratch without using any libraries such as Keras/Tensorflow. Only Numpy is used in the code for the neural network.

## Dataset

The MNIST dataset, made available from [Kaggle](https://www.kaggle.com/c/digit-recognizer/data) was used for the training. Each image is 28x28 (0-255) of handwritten numbers. There are 42000 training images. I used 37800 (90%) for training and 4200 (10%) for testing. I scaled down the images to (0,1) from (0,255) and reconverted back to the original scale for plotting images. 

## Model

The model is a 3 layer fully connected neural network with Leaky ReLU activation. The input layer has (28x28) 784 neurons and the other two layers have 10 neurons each. The final output has softmax to convert the logits into probabilities. The maximum probability is considered the prediction of the model. 

Samson used ReLU activation but for some reason I got really low (30%) accuracy with that because the gradient for `db2` was vanishing. I had learned about Leaky ReLU in a course so I used that and immediately got better results. Later, I realized that the main cause for my error was not that Leaky ReLU worked better than ReLU for this shallow network but rather the input was not normalized to (0,1) range from (0,255).

I also changed the way the weights and biases were initialized. I used He's initialization which works best for vanishing gradient problems while using ReLU activation. I read about the idea [here](https://www.geeksforgeeks.org/kaiming-initialization-in-deep-learning/). 

## Results

Using standard gradient descent and only 2 hidden layers, 

**Iterations** = 3000 (~10 minutes)

**Training Accuracy (41000 images)** = 90.27%

**Testing Accuracy (1000 images)** = 90.6%

Neuron Activation was consistently above 50% and no gradients vanished or exploded. The model generalizes well for unseen data. At the end I even found a index where the model wrongly predicts the number 4 as 9. 

## Futher Improvements

Increasing the number of layers and number of neurons in each layer. E.g. 784 -> 64 -> 32 -> 10 with Leaky ReLU activation between each layer.
**Training Accuracy** = 92.41%

**Testing Accuracy** = 92.7%

---

I used a traditional momentum based approach: $vW = \beta vW + (1-\beta)dW$. Where dW is the derivative of the weights used in GD, $\beta$ is a tunable hyperparameter (=0.9) and vW is a 'velocity' weight. The formula was described [here](https://www.geeksforgeeks.org/ml-momentum-based-gradient-optimizer-introduction/). Used 2 layer network for this.

**Training Accuracy** = 89.51%

**Testing Accuracy** = 89%

---

Using Adam Optimizer was a significant improvement in training accuracy but testing accuracy did not increase more than 92%. Used 2 layer network for this.

**Training Accuracy** = 95.25%

**Testing Accuracy** = 92.7%

At this point the model was most likely overfitting the training data. 

---

Using Adam Optimizer + 4 hidden layers in the network resulted in great results. 

**Training Accuracy** = 99.99% (Overfitting)

**Testing Accuracy** = 97.11%
