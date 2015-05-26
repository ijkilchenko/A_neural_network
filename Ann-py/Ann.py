import numpy as np
import numpy.matlib as mp
import math
import random
import datetime

class Example(object):
    '''(Numeric) array and label class (for convenience)'''    
    def __init__(self, arr, y, label):
        self.arr = arr
        self.y = y
        self.label = label
        
class Model(object):
    '''Stores the weights and named classes of the neural network'''    
    def __init__(self, Thetas, classes):
        self.Thetas = Thetas
        self.classes = classes
        
        shape_1 = self.Thetas[0].shape
        n_i = shape_1[1] - 1
        L = len(self.Thetas) + 1
        n_h = L - 2
        shape_L = self.Thetas[len(self.Thetas) - 1].shape
        n_o = shape_L[1]
        
        self.name = 'model_n_i_' + str(n_i) + '_n_o_' + str(n_o) + '_n_h_' + str(n_h) + ' ' + str(datetime.datetime.today()) + '.annm'

class Ann(object):
    '''Feed-forward neural network with arbitrary architecture'''    
    def __init__(self, *args, **kwargs):
        self.min = 2 * 10 ** (-308)  # Use this as minimum value (without underflow) bigger than 0
        s = []  # Architecture vector is initially undefined
        '''Constructor checks if a model is passed (assumes if only one argument is passed, then it is the model)'''
        if (len(args) == 1):
            model = args[0]
            '''Make sense of the model'''
            shape_1 = model.Thetas[0].shape
            self.n_i = shape_1[1] - 1
            self.L = len(model.Thetas) + 1
            self.n_h = self.L - 2
            s = []
            for l in range(0, self.L - 2):  # For each hidden layer
                num = model.Thetas[l].shape[1]
                s.append(num + 1)
            shape_L = model.Thetas[len(model.Thetas) - 1].shape
            self.n_o = shape_L[1]  # Note: there is no bias for the output layer
            
            self.classes = model.classes
        else:           
            '''Constructor checks if architecture is defined in kwargs, else list of train_examples and labels is used to define architecture'''
            self.train_examples = []
            self.test_examples = []
            if (len(args) == 0):
                self.n_i = kwargs['n_i']  # Number of input neurons
                self.n_h = kwargs['n_h']  # Number of hidden layers
                self.n_o = kwargs['n_o']  # Number of output neurons
            else:
                t = args
                (arrs, labels) = (t[0], t[1])
                if (len(arrs) != len(labels)):
                    print('Number of train_examples do not match the number of labels!')
                    return
                else:
                    '''Makes an array for each label (assumes string labels)'''
                    dimension = len(arrs[0])
                    self.classes = []
                    for x in range(0, len(arrs)):
                        if (labels[x] not in self.classes):
                            self.classes.append(labels[x])
                    '''If there are at least 10 examples, do a 90-10 random split into train and test'''
                    if (len(arrs) >= 10):
                        l = random.sample(range(0, len(arrs)), math.floor(len(arrs) / 10))
                    for x in range(0, len(arrs)):
                        y = [0] * len(self.classes)
                        y[self.classes.index(labels[x])] = 1
                        ex = Example(arrs[x], y, labels[x])
                        if (len(arrs) >= 10 and x in l):
                            self.test_examples.append(ex)
                        else:
                            self.train_examples.append(ex)
                        
                    if (len(kwargs.keys()) != 0):
                        self.n_h = kwargs['n_h']  # Number of hidden layers
                    else:
                        self.n_h = 2  # Default to 2 hidden layers
                    self.n_i = dimension
                    self.n_o = len(self.classes)  # Number of different labels (number of classes)
        if ('s' in kwargs.keys()):
            s = kwargs['s']
        self.init_architecture(s)
        if (len(args) == 1):
            '''Load Thetas from the model if model was passed (overwrite random initializations)'''
            self.Thetas = model.Thetas
        
        '''Our non-linear function (sigmoid) mapping z to (0, 1)'''
        def g(z):
            # Guard against Overflow
            if (-z > 600):
                return self.min
            else:
                return 1 / (1 + math.exp(-z))
        self.g = np.vectorize(g)
        
    def init_architecture(self, s):
        self.L = 1 + self.n_h + 1  # Total number of layers
        # len(s) == 0 is true when we did not define a hidden layer architecture explicitly
        self.s = []
        if (len(s) == 0):
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
        else:
            self.s.append(int(self.n_i + 1))  # Inputs plus 1 bias
            self.s.extend(s)  # Add the defined hidden layer architecture (1 neuron per layer will be treated as bias)
            self.s.append(int(self.n_o + 1))  # Adding bias to output layer (for convenience)
        
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
    
    def cost(self, **kwargs):
        cost = 0
        if (len(kwargs.keys()) == 0):
            lam = 0
        else:
            lam = kwargs['lam']
        for example in self.train_examples:
            h = self.h(example.arr)
            y = example.y
            for k in range(0, len(self.classes)):
                # Guard against math domain error (log(0) is undefined)
                cost += y[k] * math.log(h[k] + self.min) + (1 - y[k]) * math.log(1 - h[k] + self.min)
        S = 0
        if (lam != 0):
            # Regularization
            for l in range(0, self.L - 1):
                S += np.sum(np.multiply(self.Thetas[l], self.Thetas[l]))                                       
        return -cost / len(self.train_examples) + S * lam / (2 * len(self.train_examples))
    
    def train_accuracy(self):
        '''Returns a percentage of correctly classified train examples by neural network'''
        return self.accuracy(self.train_examples)
    
    def test_accuracy(self):
        if (len(self.test_examples) < 1):
            print('There are 0 test examples!')
        else:
            return self.accuracy(self.test_examples)
    
    def accuracy(self, examples):
        num_correct = 0
        for ex in examples:
            confidences = self.h_with_confidences(ex.arr)
            y = ex.y
            answer_index = 0
            best_confidence = 0
            for c in range(0, len(confidences)):
                if (confidences[c] > best_confidence):
                    best_confidence = confidences[c]
                    answer_index = c
            if (y[answer_index] == 1):
                num_correct += 1
        return num_correct / len(examples)
    
    def validate(self):
        '''Just prints all train examples (vectors) and their classification by the neural network and their expected classification'''
        for ex in self.train_examples:
            print(str(ex.arr) + ' -> ' + '(hypothesis: ' + str(self.h_by_class(ex.arr)) + ', expectation: ' + str(ex.label) + ')')
        return self.train_accuracy()
    
    def h_by_class(self, x):
        '''This function can be used when Ann is initialized with a model only (no examples)'''
        confidences = self.h_with_confidences(x)
        class_index = 0
        best_confidence = 0
        for c in range(0, len(confidences)):
            if (confidences[c] > best_confidence):
                best_confidence = confidences[c]
                class_index = c
        return self.classes[class_index]
       
    def h_with_confidences(self, x):
        '''Interpretation of the output activation vector in terms of probabilities (sum of all elements is 1)'''
        '''This function can be used when Ann is initialized with a model only (no examples)'''
        a = self.h(x)
        s = 0
        for i in a:
            s += i
        a = a / s
        return a

    def backward_all(self, **kwargs):
        if (len(kwargs.keys()) == 0):
            lam = 0
        else:
            lam = kwargs['lam']
        D = []
        for l in range(0, self.L - 1):
            shape = self.Thetas[l].shape
            d = np.zeros(shape)
            D.append(d)
        
        for ex in self.train_examples:  # L-1 (Jacobian) matrices (matrices of partial derivatives of for each element in Thetas)
            Js = self.backward(ex.arr, ex.y)
            # Accumulate all partial derivatives in D
            for l in range(0, self.L - 1):
                D[l] += Js[l]
        
        for l in range(0, self.L - 1):
            # Average the accumulated partial derivatives by number of train_examples
            D[l] = D[l] / len(self.train_examples) + lam * self.Thetas[l] / len(self.train_examples)  # Regularization term
        
        return D
    
    def train(self, **kwargs):
        # Default optimization hyperparameters
        it = 3000  # Maximum number of iterations
        tol = 0.0001  # Stopping tolerance (with respect to decreasing cost function)
        step = 2  # Gradient descent step size
        if ('it' in kwargs.keys()):
            it = kwargs['it']
        if ('tol' in kwargs.keys()):
            tol = kwargs['tol']
        if ('step' in kwargs.keys()):
            step = kwargs['step']        
        if ('lam' in kwargs.keys() or len(self.test_examples) == 0):
            if (len(kwargs.keys()) == 0):
                lam = 0
            else:
                lam = kwargs['lam']
            # If there are no test examples, just set lam = 0 and do not regularize
            print('\n')
            print('Starting train accuracy ' + str(self.train_accuracy()))
            model = self.train_with_lam(lam, it=it, tol=tol, step=step)
            print('Ending train accuracy ' + str(self.train_accuracy()))
            print('\n')
            
            return ([model], [self.train_accuracy()], [self.cost(lam=lam)])
        else:
            ''' Do a search for the best regularization parameter lam in the interval (lam_min, lam_max)'''
            lam_min = 0
            lam_max = 1
            lam_num = 2
            
            test_accuracies = []
            test_costs = []
            models = []
            
            steps = list(np.linspace(lam_min, lam_max, lam_num))
            for lam in steps:
                print('\n')
                print('Setting lambda=' + str(lam))
                print('Starting test accuracy=' + str(self.test_accuracy()))
                print('Starting train accuracy ' + str(self.train_accuracy()))
                model = self.train_with_lam(lam)
                models.append(model)
                t = self.test_accuracy()
                print('Ending test accuracy=' + str(t))
                print('Ending train accuracy ' + str(self.train_accuracy()))
                test_accuracies.append(t)
                test_costs.append(self.cost(lam=lam))
                print('\n')
                
            return (models, test_accuracies, test_costs)
                
    def train_with_lam(self, lam, **kwargs):
        '''Convex optimization (full-batch gradient descent)'''
        it = kwargs['it']
        tol = kwargs['tol']
        step = kwargs['step']    
        print('\tMaximum number of iterations: ' + str(it))
        print('\tTolerance: ' + str(tol))
        print('\tGradient descent step size: ' + str(step))
        
        last_100_costs = []
        count = 0
        cost_before = self.cost(lam=lam)
        for i in range(0, it):
            if (count < 100):
                count += 1
            else:
                last_100_costs = last_100_costs[1:]
            last_100_costs.append(cost_before)
            if (i % 100 == 0):
                print('\tIteration ' + str(i) + '. Cost: ' + str(cost_before))
            
            D = self.backward_all(lam=lam)
            for l in range(0, self.L - 1):
                self.Thetas[l] = self.Thetas[l] - step * D[l]  # Take a step down the gradient (of the cost function)
            
            cost_after = self.cost(lam=lam)
            cost_before = cost_after
            # After the first 100 iterations, check for stopping tolerance condition
            if (count == 100):
                # If the average cost over the last 10 iterations is within tol, then stop
                if (abs(sum(last_100_costs) / len(last_100_costs) - cost_after) < tol):
                    break                    
                    
        print('\tFinal cost: ' + str(cost_after) + ' (after ' + str(i + 1) + ' iterations)')
        
        model = Model(self.Thetas, self.classes)
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
