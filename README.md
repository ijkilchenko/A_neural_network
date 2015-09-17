# Ann
This is an implementation of the [feedforward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network) trained using [backpropagation](https://en.wikipedia.org/wiki/Backpropagation). This Python module is rather flexible because it allows you to create a network with an arbitrary number of hidden layers and an arbitrary number of neurons in each layer simply by passing a vector which represents your desired architecture. 

## Installation
Make sure to have `numpy` installed. 

```
pip install numpy
```

Git clone (or better yet, fork!) this repository somewhere. 

```
git clone https://github.com/ijkilchenko/Ann.git
```

Although some tests and demo scripts will work with Python 2, use Python 3 as this is the preferred version which correctly runs the tests and demos. 

## Getting started
You can run the tests to check that the current `master` is passing. 

```
python3 Ann_test.py
```

Run the demo script to get an intuition about how to declare objects of class `Ann` and what arguments to pass to the constructors. Do 

```
python3 Ann_demo.py
```

to understand why someone would want to make a neural network with more than 0 hidden layers. Hint: if this feedforward neural network is initialized with 0 hidden layers, then you have logistic regression. Logistic regression can't separate non-linearly separable data sets. 

Let's make a small data set to learn the OR function. 

```python
arrs = [] # List to hold the observations
labels = [] # List to hold the classes for each observation
(arrs.append([0, 0]), labels.append('false')) 
(arrs.append([0, 1]), labels.append('true'))
(arrs.append([1, 0]), labels.append('true'))
(arrs.append([1, 1]), labels.append('true'))
num_hidden_layers = 1 # Specify the number of hidden layers
ann = Ann(arrs, labels, n_h=num_hidden_layers) # Create the Ann object
ann.train() # Train using default parameters
ann.validate_train() # Use the learned model to evaluate the training set
```

After running the last command, you should get the following output:
```python
[0, 0] -> (hypothesis: false, expectation: false)
[0, 1] -> (hypothesis: true, expectation: true)
[1, 0] -> (hypothesis: true, expectation: true)
[1, 1] -> (hypothesis: true, expectation: true)
1.0
```

## Implementation details
Neural networks are very costly both in memory and time. This module uses the backpropagation algorithm to efficiently calculate the partial derivatives. Whenever possible, for-loops are avoided and vector calculations using `numpy` are used instead. When you set `n_h = 0` (set the number of hidden layers to 0), your algorithm will be equivalent to a [multinomial logistic regression](https://en.wikipedia.org/wiki/Multinomial_logistic_regression) which is typically a good enough model for most industrial applications and serves as a great baseline statistical model. Additionally, the penalty function is convex which guarantees that you will obtain a unique and global solution (and fast). When you want to learn more complex hypothesis, which is the whole point of this module, you will set the number of hidden layers to 1 or more. In this case, the penalty function is not convex and can have global minimums. Additionally, learning requires more time when you have the hidden neurons. In this module, we use [mini-batch gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) to approximate the partial derivatives quicker and we use the [momentum method](https://en.wikipedia.org/wiki/Gradient_descent#The_momentum_method) for gradient descent to try to overcome local minimums and try to approach the global minimum even on non-convex surfaces. Both of these parameters can be tuned. 

## Contact
Feel free to reach out to the author for feedforward... err feedback (get it?) at ijkilchenko then gmail then com. Feel free to fork this repository. 

