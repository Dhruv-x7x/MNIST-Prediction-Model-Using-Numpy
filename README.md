# MNIST-Prediction-Model-Using-Numpy

## How to run

Clone into the repo
```bash
git clone https://github.com/Dhruv-x7x/MNIST-Prediction-Model-Using-Numpy.git
cd MNIST-Prediction-Model-Using-Numpy
```

Run
```bash
python MLP_final.py
```

## Introduction

The original idea was taken from Samson Zhang's [video](https://www.youtube.com/watch?v=w8yWXqWQYmU&ab_channel=SamsonZhang). I used this for initial research on my own to understand backpropagation from the ground up and building a simple 2 layered MLP to predict the number shown in an image taken from the MNIST dataset. After further research about topics related to model training I improved upon this neural network in the following way:

- Mini-Batch Gradient Descent:
  - Much more memory efficient and avoids overfitting by adding a little noise to the training since it randomly samples training data for each batch. Also the gradient descent is much smoother albeit slower than stochastic gradient descent. The code I wrote allows one to easily change the batch size and train the model. For the results shown here, the batch size used was 128.
  - Batch normalization was implemented for fixing the [internal covariate shift](https://www.geeksforgeeks.org/what-is-batch-normalization-in-deep-learning/). This is not really needed as our network is shallow and the shift is not significant but I tried it because I also experimented with 5-6 layer MLPs. It makes the batch have 0 mean and a variance of 1. Introduces two learnable parameters gamma and beta which adds to the complexity.
- Validation Set:
  - Samson does not use a validation set to monitor overfitting and implement early stopping if valdiation accuracy plateaus. I used a random sample of 10% of the dataset as validation set.
- Testing Set:
  - I used 1000 images in the testing set earlier but I found that to be too little, so I set aside 10% of the dataset (4200 images), randomly sampled. The final split was 80%, 10%, 10%, no overlap between them.
- L2 Regularization:
  - I added regularization to avoid over-reliance on any one weight or bias. We penalize large weights and biases or in other words, force them to be small.
- Bias initialization:
  - Although Zhang used a simple initialization that works decently well, I used He's initialization as it works amazingly for ReLU activation functions. But I had set biases to 0, I changed that to a small positive value and it increased active neuron population.
- Drop-out:
  - Randomly drops neurons in a layer if used. Helps prevent overfitting by making the neurons not rely too much on each other. Initial experiments revealed that my training accuracy was much higher than my testing accuracy, which is evidence of overfitting. Although I also use L2 Regularization and batch norm, I wanted to experiment with drop-out as well.
- Gradient Clipping:
  - Zhang did not face this issue and neither did I, initially. But as I kept experimenting with network depth, types of initialization, activations etc., I faced the issue of gradient explosion. I once got a gradient in the order of 1e97. I added some simple gradient clipping code to prevent this from happening.
- Activation functions:
  - Added more options for activation functions such as sigmoid and tanh. I already tried leaky ReLU in my research and found it to work very well.
- Plotting Loss and Confusion Matrix:
  - For more completeness of results I added plots as well as terminal outputs. Also it is surprisingly easy to plot a confusion matrix, we just add a 1 for every location indexed by (y_pred, y_true) and done.

## Dataset

The MNIST dataset, made available from [Kaggle](https://www.kaggle.com/c/digit-recognizer/data) was used for the training. Each image is 28x28 (0-255) of handwritten numbers. There are 42000 training images. I used 33600 (80%) for training, 4200 (10%) for validation and 4200 (10%) for testing. I scaled down the images to (0,1) from (0,255) and reconverted back to the original scale for plotting images. 

## Initial Research and Mistakes

After I had done the initial experiment following Samson's video, I explored on my own and tried other strategies I had learned during my AI/ML course at IIT Bombay. I tried leaky ReLU instead of ReLU, [momentum](https://www.geeksforgeeks.org/ml-momentum-based-gradient-optimizer-introduction/) based GD, [ADAM](https://www.geeksforgeeks.org/adam-optimizer/) GD. I tried increasing the number of layers as well. The results of all these experiments is given in the table below:

### Table of Results for all experiments

| Experiment | Name | Optimizer | Layers | Training Accuracy | Testing Accuracy | Notes |
|------------|---------------|-----------|--------|-------------------|------------------|-------|
| 1 | Model 0 | Standard GD | 3 (2 hidden + 1 input) | 90.27% | 90.6% | Naive implementation serves as baseline |
| 2 | Model 1 | Standard GD | 5 (4 hidden + 1 input) | 92.41% | 92.7% | Improved depth & non-linearity helped generalization |
| 3 | Model 2 | Momentum GD (β = 0.9) | 3 (2 hidden + 1 input) | 89.51% | 89.0% | Classical momentum update: `vW = β*vW + (1−β)*dW` |
| 4 | Model 3 | ADAM | 3 (2 hidden + 1 input) | 95.25% | 92.7% | Stronger convergence but slight overfitting observed |
| 5 | Model 4 | ADAM | 5 (4 hidden + 1 input) | 99.99% | 97.11% | Best performance; Overfitted on Training Data |

---

### Mistakes

I realized a few frustrating bugs only after spending a long time with the code and training all 5 models. Only when I trained Model 4 (ADAM + 5 layers) did I realize the fatal flaw. The 99% training accuracy clearly hinted at the following things:
- I used the first 4200 images for testing and the rest of training. There may have been some order to the dataset that the model learned and that is why it gave such a high accuracy.
  - SOLUTION: I shuffled the training data every iteration. But it did not change anything. I was performing stochastic GD so the entire training data was passed per epoch, I suspect that shuffing data will not help for SGD. So finally I ended up using Mini-Batch GD with shuffling. 
- Very large network for a tiny dataset. I used 5 layers (784->64->64->32->10) which was overkill. Reasonable network depths gave good generalizations as is evident from the table.
  - SOLUTION: Shallower network
- There was a need for regularization! I am at 100% accuracy which is crazy. This is not how you train neural nets.
  - SOLUTION: L2 Regularization

I fixed all of these mistakes and added all the other improvements mentioned at the top of this README as a list. 

--- 

## Real Results

![Loss Plot](https://github.com/Dhruv-x7x/MNIST-Prediction-Model-Using-Numpy/blob/main/results/loss_plot.png)

![Confusion Matrix](https://github.com/Dhruv-x7x/MNIST-Prediction-Model-Using-Numpy/blob/main/results/confusion_matrix.png)

![Neuron Weights as Images](https://github.com/Dhruv-x7x/MNIST-Prediction-Model-Using-Numpy/blob/main/results/neuron_plot.png)

### Final MLP

The results in the above plots are for the following parameters:

```python
model = MLP(
        layer_dims=[784, 128, 64, 32, 10],
        activations=['leaky_relu', 'leaky_relu', 'leaky_relu', 'softmax'],
        keep_probs=[0.7, 0.7, 0.7, 1.0],
        l2_lambda=0.01,
        random_seed=42
    )
history = model.train(
        X_train, y_train, 
        X_val=X_val, y_val=y_val,
        learning_rate=0.01,
        num_epochs=50,
        batch_size=128,
        print_interval=5,
        patience=10
    )
```

---

### Analysis

| Batch Size | Testing Accuracy | Early Stop Trigger |
|------------|------------------|--------------------|
| 32 | 90.93% | Epoch 15 |
| 64 | 94.21% | Epoch 25 | 
| 128 | 95.55% | Epoch 29 | 
| 256 | 96.19% | Epoch 35 | 
| 512 | **96.36%** | Epoch 31 |

Out network is overkill for MNIST so we see pretty high accuracies. And as expected, increasing batch size also increases the accuracy. A larger batch gives a more meaningful and correct gradient update. Each of these trains took 75 seconds on average. I did not scale the learning rate with batch size here. 

---

| Learning Rate | Testing Accuracy | Early Stop Trigger | Time Taken | 
|------------|------------------|-----------------------|------------|
| 0.001 | **97.26%** | Epoch 29 | 143s |
| 0.01 | 95.14% | Epoch 15 | 45s |
| 0.02 | 92.55% | Epoch 15 | 37s |
| 0.05 | 87.38% | Epoch 18 | 42s |
| 0.1 | 25.19% | Epoch 11 | 27s |

I decided not to use any sort of learning rate annealing strategies. I kept a constant learning rate and this experiment was done just to see how much time it takes for different learning rates and how much overfitting/underfitting it leads to. The batch size used here was 128 and patience level was 10. The results are as expected; a very tiny learning rate takes a long time to converge but gives the highest accuracy. A large step size of 0.1 does not converge and validation check quickly exits the training loop. 

It's interesting to observe that the neural weights for a batch clearly look like individual numbers for learning rate 0.1 while for 0.001 they look more like noise. 

---

| Idx | Layers | Testing Accuracy | Early Stop Trigger | Time Taken | 
|-----|------------|------------------|-----------------------|------------|
| 1 | 784 -> 10 -> 10 | 90.81% | Epoch 22 | 10s |
| 2 | 784 -> 16 -> 10 | 92.83% | Epoch 22 | 15s |
| 3 | 784 -> 32 -> 16 -> 10 | 93.52% | Epoch 34 | 24s |
| 4 | 784 -> 64 -> 32 -> 10 | 94.93% | Epoch 30 | 39s |
| 5 | 784 -> 64 -> 32 -> 16 -> 10 | 93.71% | Epoch 37 | 97s |
| 6 | 784 -> 128 -> 64 -> 32 -> 10 | 94.52% | Epoch 35 | 147s |
| 7 | 784 -> 256 -> 128 -> 64 -> 32 -> 10 | **95.14%** | Epoch 39 | 510s |

Learning rate: 0.01
Batch Size: 128

The first row here is the exact model we used in the first row of our research experiments table above and the results agree!

What we see is that deeper networks and networks with more neurons per layer have higher accuracy as they are able to generalize over more features than shallow networks. Every time we increase the number of layers, there is a significant jump in the training time. This effect is most apparent in the jump from 4 layers to 5 layers, a 58s jump. We don't see accuracy increasing further in excessively deep networks here because of regularization which prevents overfitting and also all the other techniques we utilized to avoid overfitting such as batch normalization and drop out. The last row presents a way too big network for MNIST, with 6 layers. I wouldn't believe MNIST to have this many features to learn for accurate prediction but the accuracy is highest for this network which takes almost 9 minutes to train.

Network 7 and Network 4 have very similar accuracy (only 0.21% difference) but Network 4 is 12 times faster. We can conclude that having a larger network is not necessarily better for accuracy. 

