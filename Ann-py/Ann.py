import numpy as np
import numpy.matlib as mp
import math

class Example(object):
    '''(Numeric) array and label class (for convenience)'''    
    def __init__(self, arr, y, label):
        self.arr = arr
        self.y = y
        self.label = label
        
class Model(object):
    '''Stores the weights of the neural network'''    
    def __init__(self, Thetas):
        self.Thetas = Thetas

class Ann(object):
    '''Feed-forward neural network with arbitrary architecture'''    
    def __init__(self, *args, **kwargs):
        self.min = 2 * 10 ** (-308)  # Use this as minimum value (without underflow) bigger than 0
        
        '''Constructor checks if architecture is defined in kwargs, else list of examples and labels is used to define architecture'''
        self.examples = []
        if (len(args) == 0):
            self.n_i = kwargs['n_i']  # Number of input neurons
            self.n_h = kwargs['n_h']  # Number of hidden layers
            self.n_o = kwargs['n_o']  # Number of output neurons
        else:
            (arrs, labels) = args
            if (len(arrs) != len(labels)):
                print('Number of examples do not match the number of labels')
                return
            else:
                '''Makes an array for each label (assumes string labels)'''
                dimension = len(arrs[0])
                self.m = len(arrs)
                self.classes = []
                for x in range(0, len(arrs)):
                    if (labels[x] not in self.classes):
                        self.classes.append(labels[x])
                for x in range(0, len(arrs)):
                    y = [0] * len(self.classes)
                    y[self.classes.index(labels[x])] = 1
                    ex = Example(arrs[x], y, labels[x])
                    self.examples.append(ex)
                    
                if (len(kwargs.keys()) != 0):
                    self.n_h = kwargs['n_h']  # Number of hidden layers
                else:
                    self.n_h = 2  # Default to 2 hidden layers
                self.n_i = dimension
                self.n_o = len(self.classes) 
                self.K = len(self.classes)  # Number of different labels (number of classes)
        
        self.init_architecture() 
        
        '''Our non-linear function (sigmoid) mapping z to (0, 1)'''
        def g(z):
            # Guard against Overflow
            if (-z > 600):
                return self.min
            else:
                return 1 / (1 + math.exp(-z))
        self.g = np.vectorize(g)
        
    def init_architecture(self):
        self.L = 1 + self.n_h + 1  # Total number of layers
                
        '''Calculate number of neurons in each layer (using own heuristic)'''
        self.s = [0] * self.L  # Will hold number of neurons (including hidden) in each layer
        for l in range(0, self.L):
            if (l == 0):
                self.s[l] = int(self.n_i + 1)  # Inputs plus 1 bias
            if (l > 0 and l < self.L - 1):
                '''Grow number of hidden neurons logarithmically after 10'''
                if (self.n_i <= 10):
                    self.s[l] = int(self.n_i + 1) 
                else:
                    self.s[l] = int(math.floor(10 * math.log10(self.n_i)) + 1)
            if (l == self.L - 1):
                self.s[l] = int(self.n_o + 1)  # Adding bias to output layer (for convenience) 
        
        '''Initialize all neuron weights randomly between -1 and 1'''
        self.Thetas = []  # Will hold L-1 matrices
        for l in range(0, self.L - 1):
            '''Neuron activations vector in layer l is shaped (s[l], 1)'''
            '''Number of unbiased neurons in next layer is s[l+1]-1'''
            shape = (self.s[l + 1] - 1, self.s[l])  # Note: we do not compute the activation for the bias neuron in next layer
            Theta = np.ones(shape) - 2 * mp.rand(shape)  # Matrix of 1s minus twice the matrix of random weights between 0 and 1
            self.Thetas.append(Theta)
        '''Examples'''
        '''
        a^(0) # Activations vector in layer 0 with shape (s[0], 1) (Note: a_0 = [x.T 1])
        a_{1}^(1) = g(self.Theta[0][1,:].dot(a^(0))) # Activation of second neuron in second layer
        a^(1) = [g(self.Theta[0].dot(a^(0))) 1]^T # Activations vector in layer 1
        '''
    def h(self, x):
        '''Returns the activation column vector in the output layer (the neural network hypothesis function)'''
        a = self.forward(x)
        return a[len(a) - 1]
    
    def cost(self):
        cost = 0
        for example in self.examples:
            h = self.h(example.arr)
            y = example.y
            for k in range(0, len(self.classes)):
                # Guard against math domain error (log(0) is undefined)
                cost += y[k] * math.log(h[k] + self.min) + (1 - y[k]) * math.log(1 - h[k] + self.min)
        return -cost / self.m
    
    def accuracy(self):
        '''Returns a percentage of all examples to which the label is the same as which the neural network gave the highest confidence to'''
        num_correct = 0
        for ex in self.examples:
            confidences = self.answer_with_confidences(ex.arr)
            y = ex.y
            answer_index = 0
            best_confidence = 0
            for c in range(0, len(confidences)):
                if (confidences[c] > best_confidence):
                    best_confidence = confidences[c]
                    answer_index = c
            if (y[answer_index] == 1):
                num_correct += 1
        return num_correct / self.m
    
    def validate(self):
        for ex in self.examples:
            print(ex.arr, self.answer(ex))
    
    def answer(self, x):
        confidences = self.answer_with_confidences(x.arr)
        answer_index = 0
        best_confidence = 0
        for c in range(0, len(confidences)):
            if (confidences[c] > best_confidence):
                best_confidence = confidences[c]
                answer_index = c
        return self.classes[answer_index]
       
    def answer_with_confidences(self, x):
        '''Interpretation of the output activation vector in terms of probabilities (sum of all elements is 1)'''
        a = self.h(x)
        s = 0
        for i in a:
            s += i
        a = a / s
        return a
    

    def backward_all(self):
        D = []
        for l in range(0, self.L - 1):
            shape = self.Thetas[l].shape
            d = np.zeros(shape)
            D.append(d)
        
        for ex in self.examples:  # L-1 (Jacobian) matrices (matrices of partial derivatives of for each element in Thetas)
            Js = self.backward(ex.arr, ex.y)
            # Accumulate all partial derivatives in D
            for l in range(0, self.L - 1):
                D[l] += Js[l]
        
        for l in range(0, self.L - 1):
            D[l] = D[l] / self.m  # Average the accumulated partial derivatives by number of examples
        
        return D

    def train(self):
        '''Convex optimization (full-batch gradient descent)'''
        it = 10000  # Maximum number of iterations
        tol = 0.000001  # If iteration gave less than tol decrease in cost, then stop
        step = 5  # Gradient descent step size
        
        print('\tStarting accuracy ' + str(self.accuracy()))
        for i in range(0, it):
            cost_before = self.cost()
            if (i % 10 == 0):
                print('Iteration ' + str(i) + '. Cost: ' + str(cost_before))
            
            D = self.backward_all()
            for l in range(0, self.L - 1):
                self.Thetas[l] = self.Thetas[l] - step * D[l]  # Take a step down the gradient (of the cost function)
            
            cost_after = self.cost()
            
            if (abs(cost_before - cost_after) < tol):
                break
                    
        print('Final cost: ' + str(cost_after) + ' (after ' + str(i + 1) + ' iterations)')
        print('\tEnding accuracy ' + str(self.accuracy()))
        
        model = Model(self.Thetas)
        return model
                   
    def forward(self, x):
        '''Assumes x is like [1, 2, 3] (not numpy matrix)'''
        a = []  # Activation vectors for each of self.L layers
        x = x + [1]  # Bias the input (non-destructively)
        a.append(np.matrix([x]).T)  # Add biased input as column vector (s[0], 1) to a
        
        '''Calculate the remaining self.L-1 activation vectors'''
        for l in range(0, self.L - 1):
            z = self.g(self.Thetas[l].dot(a[l]))  # Activation column vector for unbiased neurons in layer l+1
            a.append(np.concatenate((z, np.matrix([[1]])), axis=0))  # Bias z and add vector of shape (s[l+1], 1) to a
        
        v = a[len(a) - 1]  # Last activation column vector (biased)
        a[len(a) - 1] = v[0:v.shape[0] - 1, :]  # Get rid of bias in output layer 
        
        return a
    
    def backward(self, x, y):
        '''The back-propagation algorithm for labeled example (x, y)'''
        '''Assumes y is like [1, 0, 1] (not numpy matrix and each element in y is between 0 and 1)'''
        Y = np.matrix([y]).T  # Make y (non-destructively) into a column vector
        a = self.forward(x)
        
        '''Initialize an error term for each neuron in each layer (errors for input layer will be left as zero for convenience)'''
        d = []
        for l in range(0, self.L):
            shape = a[l].shape
            d.append(np.zeros(shape))
            
        # Errors in the output layer
        d[len(d) - 1] = a[len(d) - 1] - Y
        # For each hidden layer (working backwards)
        hidden_layers = list(range(1, self.L - 1))
        hidden_layers.reverse()
        for l in hidden_layers:
            if (l != self.L - 2):
                # Do not back propagate the bias terms
                D = d[l + 1][0:d[l + 1].shape[0] - 1, :]
            else:
                # In this else-statement, l+1 is the output layer
                D = d[l + 1]
            A = a[l]
            L = self.Thetas[l].T.dot(D)
            R = np.multiply(A, 1 - A)
            d[l] = np.multiply(L, R)
        
        # Will hold L-1 (Jacobian) matrices (Note: for all l, Thetas[l].shape = Js[l].shape)
        Js = []
        for l in range(0, self.L - 1):
            if (l != self.L - 2):
                # Do not back propagate the bias terms
                D = d[l + 1][0:d[l + 1].shape[0] - 1, :]
            else:
                # In this else-statement, l+1 is the output layer
                D = d[l + 1]
            P = D.dot(a[l].T)
            Js.append(P)
        
        return Js  # Returns L-1 Jacobian matrices (matrices of partial derivatives)
