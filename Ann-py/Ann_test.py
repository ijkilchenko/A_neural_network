import unittest
from Ann import Ann
import numpy as np
import random
import copy

class Test(unittest.TestCase):
    
    def test_all(self):
        ''' The A(lex) Neural Network tests (each sample_test is independent of others and order independent)'''
        # self.sample_test_1()
        # self.sample_test_2()
        # self.sample_test_3()
        # self.sample_test_4()
        # self.sample_test_5()  # Takes over a minute to run
        # self.sample_test_6()  # Takes over 10 minutes to run
        self.sample_test_7()
        # self.sample_test_8()
        
        ''' Good fake data-set tests '''
        # self.sample_test_9()  # Takes about 10 hours to run
        
    def sample_test_9(self):
        '''Creates a fake data-set with points labeled 'yes' around origin and points labeled 'no' outside'''
        arrs = []
        labels = []
        '''Points about the origin (located in a box of length 16 centered at origin)'''
        for i in range(0, 100):
            arr = [random.randint(0, 8) * np.sign(random.random() - 0.5) for x in range(0, 10)]
            label = 'yes'
            arrs.append(arr)
            labels.append(label)
        '''Points outside the box'''
        for i in range(0, 100):
            arr = [random.randint(10, 20) * np.sign(random.random() - 0.5) for x in range(0, 10)]
            label = 'no'
            arrs.append(arr)
            labels.append(label)
        '''Add some noise'''
        for i in range(0, 10):
            arr = [random.randint(0, 8) * np.sign(random.random() - 0.5) for x in range(0, 10)]
            label = 'no'  # Note: this is artificially misclassified
            arrs.append(arr)
            labels.append(label)
        for i in range(0, 10):
            arr = [random.randint(10, 20) * np.sign(random.random() - 0.5) for x in range(0, 10)]
            label = 'yes'  # Note: this is artificially misclassified
            arrs.append(arr)
            labels.append(label)
            
        ann = Ann(arrs, labels, n_h=2)
        ann.train()

    def sample_test_1(self):
        # Test for Ann Architecture#
        
        # First architecture test#
        n_i1 = 4  # Number of input neurons
        n_h1 = 2  # Number of hidden layers
        n_o1 = 1  # Number of output neurons
        
        ann1 = Ann(n_i=4, n_h=2 , n_o=1)  # Create this architecture
        self.assertEqual(n_i1, ann1.n_i)
        self.assertEqual(n_h1, ann1.n_h)
        self.assertEqual(n_o1, ann1.n_o)
        
        self.assertEqual(ann1.s, [5, 5, 5, 2])
        self.assertEqual(len(ann1.Thetas), 3)
        self.assertEqual(ann1.Thetas[0].shape, (4, 5))
        self.assertEqual(ann1.Thetas[1].shape, (4, 5))
        self.assertEqual(ann1.Thetas[2].shape, (1, 5))
        
        # Second architecture test#
        n_i2 = 10  # Number of input neurons
        n_h2 = 1  # Number of hidden layers
        n_o2 = 2  # Number of output neurons
        
        ann2 = Ann(n_i=n_i2, n_h=n_h2, n_o=n_o2)  # Create this architecture
        self.assertEqual(n_i2, ann2.n_i)
        self.assertEqual(n_h2, ann2.n_h)
        self.assertEqual(n_o2, ann2.n_o)
        
        self.assertEqual(ann2.s, [11, 11, 3])
        self.assertEqual(len(ann2.Thetas), 2)
        self.assertEqual(ann2.Thetas[0].shape, (10, 11))
        self.assertEqual(ann2.Thetas[1].shape, (2, 11))
        
        # Third architecture test#
        n_i3 = 100  # Number of input neurons
        n_h3 = 0  # Number of hidden layers
        n_o3 = 10  # Number of output neurons
        
        ann3 = Ann(n_i=n_i3, n_h=n_h3, n_o=n_o3)  # Create this architecture
        self.assertEqual(n_i3, ann3.n_i)
        self.assertEqual(n_h3, ann3.n_h)
        self.assertEqual(n_o3, ann3.n_o)
        
        self.assertEqual(ann3.s, [101, 11])
        self.assertEqual(len(ann3.Thetas), 1)
        self.assertEqual(ann3.Thetas[0].shape, (10, 101))
        
        n_i4 = 1500  # Number of input neurons
        n_h4 = 3  # Number of hidden layers
        n_o4 = 6  # Number of output neurons
        
        # Fourth architecture test#
        ann4 = Ann(n_i=n_i4, n_h=n_h4, n_o=n_o4)  # Create this architecture
        self.assertEqual(n_i4, ann4.n_i)
        self.assertEqual(n_h4, ann4.n_h)
        self.assertEqual(n_o4, ann4.n_o)
        
        self.assertEqual(ann4.s, [1501, 31 + 1, 31 + 1, 31 + 1, 6 + 1])
        self.assertEqual(len(ann4.Thetas), 4)
        self.assertEqual(ann4.Thetas[0].shape, (31, 1501))
        self.assertEqual(ann4.Thetas[1].shape, (31, 32))
        self.assertEqual(ann4.Thetas[2].shape, (31, 32))
        self.assertEqual(ann4.Thetas[3].shape, (6, 32))
        
        # Fourth (arbitrary) architecture test#
        s = [3, 2]
        n_i = 4
        n_h = len(s)
        n_o = 2
        ann1 = Ann(s=s, n_i=n_i, n_h=n_h, n_o=n_o)  # Create this architecture
        self.assertEqual(n_i, ann1.n_i)
        self.assertEqual(n_h, ann1.n_h)
        self.assertEqual(n_o, ann1.n_o)
        
        self.assertEqual(ann1.s, [5, 3, 2, 3])
        self.assertEqual(len(ann1.Thetas), 3)
        self.assertEqual(ann1.Thetas[0].shape, (2, 5))
        self.assertEqual(ann1.Thetas[1].shape, (1, 3))
        self.assertEqual(ann1.Thetas[2].shape, (2, 2))

    def sample_test_2(self):
        # Test for forward-propagation#
        
        # First architecture test#
        # Logistic regression (0 hidden layers) forward propagation test#
        n_i1 = 4  # Number of input neurons
        n_h1 = 0  # Number of hidden layers
        n_o1 = 1  # Number of output neurons
        
        ann1 = Ann(n_i=n_i1, n_h=n_h1, n_o=n_o1)  # Create this architecture
        x1 = [1, 2, 3, 4]  # Array as first example
        x2 = [-1, -1, -1, -1]  # Array as second example
        
        # Set all weights to zero#
        for i in range(0, len(ann1.Thetas)):
            shape = ann1.Thetas[i].shape
            self.assertEqual(shape, (1, 5))
            ann1.Thetas[i] = np.zeros(shape)
        self.assertEqual(ann1.h(x1), 0.5)
        self.assertEqual(ann1.h(x2), 0.5)
        
        # Set all weights to one#
        for i in range(0, len(ann1.Thetas)):
            shape = ann1.Thetas[i].shape
            self.assertEqual(shape, (1, 5))
            ann1.Thetas[i] = np.ones(shape)
        self.assertAlmostEqual(ann1.h(x1), 0.999, delta=0.001)
        self.assertAlmostEqual(ann1.h(x2), 0.0474, delta=0.0001)
        
        # Set all weights randomly between -1 and 1 (and test the range of output)#
        ann1 = Ann(n_i=n_i1, n_h=n_h1, n_o=n_o1)  # Create this architecture
        self.assertAlmostEqual(ann1.h(x1), 0.5, delta=0.5)  # Sigmoid always gives values between 0 and 1
        self.assertAlmostEqual(ann1.h(x2), 0.5, delta=0.5)
        
        # Custom Thetas weights#
        M = np.matrix([[1, -1, 0.5, -0.3, 2]])
        ann1.Thetas[0] = M
        self.assertAlmostEqual(ann1.h(x1), 0.786, delta=0.001)
        self.assertAlmostEqual(ann1.h(x2), 0.858, delta=0.001)
        
        # Second architecture test#
        # 1 hidden layer forward propagation test#
        n_i1 = 4  # Number of input neurons
        n_h1 = 1  # Number of hidden layers
        n_o1 = 1  # Number of output neurons
        
        ann1 = Ann(n_i=n_i1, n_h=n_h1, n_o=n_o1)  # Create this architecture
        x1 = [1, 2, 3, 4]  # Array as first example
        x2 = [-1, -1, -1, -1]  # Array as second example
        
        # Set all weights to zero#
        for i in range(0, len(ann1.Thetas)):
            shape = ann1.Thetas[i].shape
            ann1.Thetas[i] = np.zeros(shape)
        self.assertEqual(ann1.h(x1), 0.5)
        self.assertEqual(ann1.h(x2), 0.5)
        
        # Set all weights to one#
        for i in range(0, len(ann1.Thetas)):
            shape = ann1.Thetas[i].shape
            ann1.Thetas[i] = np.ones(shape)
        self.assertAlmostEqual(ann1.h(x1), 0.993, delta=0.001)
        self.assertAlmostEqual(ann1.h(x2), 0.767, delta=0.001)  
        
        # Set all weights randomly between -1 and 1 (and test the range of output)#
        ann1 = Ann(n_i=n_i1, n_h=n_h1, n_o=n_o1)  # Create this architecture
        self.assertAlmostEqual(ann1.h(x1), 0.5, delta=0.5)  # Sigmoid always gives values between 0 and 1
        self.assertAlmostEqual(ann1.h(x2), 0.5, delta=0.5)
        
        # Custom Thetas weights#
        M1 = np.matrix([[1, -1, 0.5, -0.3, 2],
                       [1, -1, 0.5, -0.3, 2],
                       [1, -1, 0.5, -0.3, 2],
                       [1, -1, 0.5, -0.3, 2]])
        M2 = np.matrix([[1, 1, -1, 0.5, -1]])
        ann1.Thetas[0] = M1
        ann1.Thetas[1] = M2
        # a^(1) Should be [0.786 0.786 0.786 0.786 1]^T#
        self.assertAlmostEqual(ann1.h(x1), 0.545, delta=0.001)
        # a^(1) Should be [0.858 0.858 0.858 0.858 1]^T#
        self.assertAlmostEqual(ann1.h(x2), 0.571, delta=0.001)
        
    def sample_test_3(self):
        
        # Test the dimensions of the Jacobian matrices against Theta matrices for first architecture#
        n_i1 = 4  # Number of input neurons
        n_h1 = 2  # Number of hidden layers
        n_o1 = 2  # Number of output neurons
        
        ann1 = Ann(n_i=n_i1, n_h=n_h1, n_o=n_o1)  # Create this architecture
        x1 = [1, 2, 3, 4]  # Array as first example
        y1 = [1, 0]
        J = ann1.backward(x1, y1)
        for l in range(0, ann1.L - 1):
            self.assertEqual(ann1.Thetas[l].shape, J[l].shape)
            
        # Test the dimensions of the Jacobian matrices against Theta matrices for second architecture#
        n_i1 = 40  # Number of input neurons
        n_h1 = 3  # Number of hidden layers
        n_o1 = 10  # Number of output neurons
        
        ann1 = Ann(n_i=n_i1, n_h=n_h1, n_o=n_o1)  # Create this architecture
        x1 = 10 * [1, 2, 3, 4]  # Array as first example
        y1 = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        J = ann1.backward(x1, y1)
        for l in range(0, ann1.L - 1):
            self.assertEqual(ann1.Thetas[l].shape, J[l].shape)
            
        # Test the dimensions of the Jacobian matrices against Theta matrices for third architecture#
        n_i1 = 40  # Number of input neurons
        n_h1 = 0  # Number of hidden layers
        n_o1 = 10  # Number of output neurons
        
        ann1 = Ann(n_i=n_i1, n_h=n_h1, n_o=n_o1)  # Create this architecture
        x1 = 10 * [1, 2, 3, 4]  # Array as first example
        y1 = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
        J = ann1.backward(x1, y1)
        for l in range(0, ann1.L - 1):
            self.assertEqual(ann1.Thetas[l].shape, J[l].shape)
            
    def sample_test_4(self):
        # Gradient checking (check that a numerical approximation of the gradient is equal to our backpropagation derivation)#
        
        # First data-set with one example
        arrs = []
        labels = []
        arrs.append([1, 2, 4, 5, 5, 5])
        labels.append('cat')
        ann = Ann(arrs, labels, n_h=10)  # Create Ann with these train_examples and labels
        J = ann.backward(ann.train_examples[0].arr, ann.train_examples[0].y)
        T_original = copy.deepcopy(ann.Thetas)
        
        for l in range(0, ann.L - 1):
            shape_J = J[l].shape
            eps = 0.0001  # epsilon for a numerical approximation of the gradient
            for i in range(0, shape_J[0]):
                for j in range(0, shape_J[1]):
                    T_e = np.zeros(shape_J)  # Matrix of zeros
                    T_e[i][j] = eps
                    ann.Thetas[l] = T_original[l] + T_e
                    cost_e = ann.cost()  # Cost at Theta + eps
                    ann.Thetas[l] = T_original[l] - T_e
                    cost_minus_e = ann.cost()  # Cost at Theta - eps
                    P = (cost_e - cost_minus_e) / (2 * eps)  # Numerical approximation
                    J_ij = J[l].item(i, j)  # Backpropagation derivation
                    
                    # print(P, '\t', J_ij, '\t', abs(P - J_ij), (l, i, j))
                    
                    # if (P < 0 and J_ij > 0 or P > 0 and J_ij < 0):
                    #    self.fail()
                    
                    self.assertAlmostEqual(P, J_ij, delta=0.001)
                    ann.Thetas = copy.deepcopy(T_original)
        
        # Second data-set with several train_examples
        arrs = []
        labels = []
        classes = ('cat', 'dog')
        for m in range(0, 100):
            arr = [random.random() for x in range(0, 20)]
            label = classes[random.random() > 0.5]
            arrs.append(arr)
            labels.append(label)
        ann = Ann(arrs, labels, n_h=2)  # Create Ann with these train_examples and labels
        # L-1 matrices of partial derivatives for first example
        J = ann.backward_all()
        T_original = copy.deepcopy(ann.Thetas)
        
        for l in range(0, ann.L - 1):
            shape_J = J[l].shape
            eps = 0.0001  # epsilon for a numerical approximation of the gradient
            a = random.sample(range(0, shape_J[0]), 2)
            b = random.sample(range(0, shape_J[1]), 2)
            for i in a:
                for j in b:
                    T_e = np.zeros(shape_J)  # Matrix of zeros
                    T_e[i][j] = eps
                    ann.Thetas[l] = T_original[l] + T_e
                    cost_e = ann.cost()  # Cost at Theta + eps
                    ann.Thetas[l] = T_original[l] - T_e
                    cost_minus_e = ann.cost()  # Cost at Theta - eps
                    P = (cost_e - cost_minus_e) / (2 * eps)  # Numerical approximation
                    J_ij = J[l].item(i, j)  # Backpropagation derivation
                    
                    # print(P, '\t', J_ij, '\t', abs(P - J_ij), (l, i, j))
                    
                    # if (P < 0 and J_ij > 0 or P > 0 and J_ij < 0):
                    #    self.fail()
                    
                    self.assertAlmostEqual(P, J_ij, delta=0.001)
                    ann.Thetas = copy.deepcopy(T_original)
                    
    def sample_test_5(self):
        # Comprehensive gradient checking #
        
        # Medium size data-set with more than two classes
        arrs = []
        labels = []
        classes = ('cat', 'dog', 'bird', 'turtle', 'dinosaur', 'human')
        for m in range(0, 100):
            arr = [random.random() for x in range(0, 200)]
            z = random.random()
            if (z < 1 / 6):
                label = classes[0]
            elif (z >= 1 / 6 and z < 2 / 6):
                label = classes[1]
            elif (z >= 2 / 6 and z < 3 / 6):
                label = classes[2]
            elif (z >= 3 / 6 and z < 4 / 6):
                label = classes[3]
            elif (z >= 4 / 6 and z < 5 / 6):
                label = classes[4]   
            else:
                label = classes[5]
            arrs.append(arr)
            labels.append(label)
        ann = Ann(arrs, labels, n_h=2)  # Create Ann with these train_examples and labels
        # L-1 matrices of partial derivatives for first example
        J = ann.backward_all()
        T_original = copy.deepcopy(ann.Thetas)
        
        # Just check the neuron connections between first, second, and third layer
        for l in range(0, 2):
            shape_J = J[l].shape
            eps = 0.0001  # epsilon for a numerical approximation of the gradient
            # Randomly select 100 neuron connections to check
            a = random.sample(range(0, shape_J[0]), 10)
            b = random.sample(range(0, shape_J[1]), 10)
            for i in a:
                for j in b:
                    T_e = np.zeros(shape_J)  # Matrix of zeros
                    T_e[i][j] = eps
                    ann.Thetas[l] = T_original[l] + T_e
                    cost_e = ann.cost()  # Cost at Theta + eps
                    ann.Thetas[l] = T_original[l] - T_e
                    cost_minus_e = ann.cost()  # Cost at Theta - eps
                    P = (cost_e - cost_minus_e) / (2 * eps)  # Numerical approximation
                    J_ij = J[l].item(i, j)  # Backpropagation derivation
                    
                    print(P, '\t', J_ij, '\t', abs(P - J_ij), (l, i, j))
                    
                    # if (P < 0 and J_ij > 0 or P > 0 and J_ij < 0):
                    #    self.fail()
                    
                    self.assertAlmostEqual(P, J_ij, delta=0.001)
                    ann.Thetas = copy.deepcopy(T_original)
            
    def sample_test_6(self):
        # Test if training works by checking that training lowers the cost for random small and medium size data-sets#
        
        # Small size random data-set
        arrs = []
        labels = []
        classes = ('cat', 'dog')
        for i in range(0, 2):
            print('\nTesting data-set ' + str(i))
            for m in range(0, 10):
                arr = [random.random() for x in range(0, 5)]
                label = classes[random.random() > 0.5]
                arrs.append(arr)
                labels.append(label)
            ann = Ann(arrs, labels)  # Create Ann with these train_examples and labels
            cost_before = ann.cost()
            ann.train()
            cost_after = ann.cost()
            self.assertTrue(cost_after <= cost_before)
            
        # Medium size random data-set
        arrs = []
        labels = []
        classes = ('cat', 'dog', 'bird')
        for i in range(0, 10):
            print('\nTesting data-set ' + str(i))
            for m in range(0, 100):
                arr = [random.random() for x in range(0, 20)]
                z = random.random()
                if (z < 0.33):
                    label = classes[0]
                elif (z >= 0.33 and z < 0.66):
                    label = classes[1]
                else:
                    label = classes[2]
                arrs.append(arr)
                labels.append(label)
            ann = Ann(arrs, labels)  # Create Ann with these train_examples and labels
            cost_before = ann.cost()
            ann.train()
            cost_after = ann.cost()
            self.assertTrue(cost_after <= cost_before)
        
    def sample_test_7(self):
        # Learn some basic functions#
        
        # Linearly-separable data-sets#
        
        # function 1 (AND function) on 0 hidden layers
        arrs = []
        arrs.append([0, 0])
        arrs.append([0, 1])
        arrs.append([1, 0])
        arrs.append([1, 1])
        labels = []
        labels.append('false')
        labels.append('true')
        labels.append('true')
        labels.append('true') 
        ann = Ann(arrs, labels, n_h=0)
        ann.train()
        ann.validate()
        # Check to see if train_accuracy is over 90%
        self.assertTrue(ann.train_accuracy() > 0.9)
        
        # function 2 on 2 hidden layers
        arrs = []
        arrs.append([1, 1])
        arrs.append([2, 2])
        arrs.append([1, 3])
        arrs.append([2, 10])
        arrs.append([1, -1])
        arrs.append([-2, -2])
        arrs.append([1, -3])
        arrs.append([-2, -10])
        labels = []
        labels.append('false')
        labels.append('false')
        labels.append('false')
        labels.append('false')
        labels.append('true')
        labels.append('true')
        labels.append('true')
        labels.append('true') 
        ann = Ann(arrs, labels, n_h=2)
        ann.train()
        ann.validate()
        # Check to see if train_accuracy is over 90%
        self.assertTrue(ann.train_accuracy() > 0.9)
        
        # Non-linearly-separable data-sets#
        
        # function 1 (XOR function) on 1 hidden layers
        arrs = []
        arrs.append([0, 0])
        arrs.append([0, 1])
        arrs.append([1, 0])
        arrs.append([1, 1])
        labels = []
        labels.append('false')
        labels.append('true')
        labels.append('true')
        labels.append('false') 
        ann = Ann(arrs, labels, n_h=1)
        ann.train()
        ann.validate()
        # Check to see if train_accuracy is over 90%
        self.assertTrue(ann.train_accuracy() > 0.9)
        
        # function 1b (XOR function) on 1 hidden layers (with custom architecture)
        arrs = []
        arrs.append([0, 0])
        arrs.append([0, 1])
        arrs.append([1, 0])
        arrs.append([1, 1])
        labels = []
        labels.append('false')
        labels.append('true')
        labels.append('true')
        labels.append('false')
        s = [10, 11, 10]  # Custom hidden layer architecture
        ann = Ann(arrs, labels, n_h=len(s), s=s)
        ann.train()
        ann.validate()
        # Check to see if train_accuracy is over 90%
        self.assertTrue(ann.train_accuracy() > 0.9)
        
        # function 1 (two nested sets) on 2 hidden layers
        arrs = []
        arrs.append([0, 0])
        arrs.append([0, 1])
        arrs.append([1, 1])
        arrs.append([1, 1])
        arrs.append([10, 0])
        arrs.append([0, 10])
        arrs.append([110, 10])
        arrs.append([-10, 10])
        labels = []
        labels.append('false')
        labels.append('false')
        labels.append('false')
        labels.append('false') 
        labels.append('true')
        labels.append('true')
        labels.append('true')
        labels.append('true') 
        ann = Ann(arrs, labels, n_h=2)
        ann.train()
        ann.validate()
        # Check to see if train_accuracy is over 90%
        self.assertTrue(ann.train_accuracy() > 0.9)
        
    def sample_test_8(self):
        # First test#
        # 1 hidden layer cost test with regularization#       
        x1 = [1, 2, 3, 4]  # Array as first example
        y1 = 'yes'
        arrs = []
        labels = []
        arrs.append(x1)
        labels.append(y1)
        ann1 = Ann(arrs, labels, n_h=1)  # Create this architecture
        
        # Custom Thetas weights#
        M1 = np.matrix([[1, -1, 0.5, -0.3, 2],
                       [1, -1, 0.5, -0.3, 2],
                       [1, -1, 0.5, -0.3, 2],
                       [1, -1, 0.5, -0.3, 2]])
        M2 = np.matrix([[1, 1, -1, 0.5, -1]])
        ann1.Thetas[0] = M1
        ann1.Thetas[1] = M2
        cost_0 = ann1.cost()  # lam equals 0
        cost_1 = ann1.cost(lam=1)  # lam equals 1
        self.assertTrue(cost_1 > cost_0)  # Cost with regularization penalty is always higher than without regularization        

        # Gradient checking (now with regularization)#
        # Medium size data-set with several train_examples
        lam_test = 1  # Regularization parameter
        arrs = []
        labels = []
        classes = ('cat', 'dog')
        for m in range(0, 100):
            arr = [random.random() for x in range(0, 40)]
            label = classes[random.random() > 0.5]
            arrs.append(arr)
            labels.append(label)
        ann = Ann(arrs, labels, n_h=2)  # Create Ann with these train_examples and labels
        # L-1 matrices of partial derivatives for first example
        J = ann.backward_all(lam=lam_test)
        T_original = copy.deepcopy(ann.Thetas)
        
        for l in range(0, ann.L - 1):
            shape_J = J[l].shape
            eps = 0.0001  # epsilon for a numerical approximation of the gradient
            a = random.sample(range(0, shape_J[0]), 2)
            b = random.sample(range(0, shape_J[1]), 2)
            for i in a:
                for j in b:
                    T_e = np.zeros(shape_J)  # Matrix of zeros
                    T_e[i][j] = eps
                    ann.Thetas[l] = T_original[l] + T_e
                    cost_e = ann.cost(lam=lam_test)  # Cost at Theta + eps
                    ann.Thetas[l] = T_original[l] - T_e
                    cost_minus_e = ann.cost(lam=lam_test)  # Cost at Theta - eps
                    P = (cost_e - cost_minus_e) / (2 * eps)  # Numerical approximation
                    J_ij = J[l].item(i, j)  # Backpropagation derivation
                    
                    # print(P, '\t', J_ij, '\t', abs(P - J_ij), (l, i, j))
                    
                    # if (P < 0 and J_ij > 0 or P > 0 and J_ij < 0):
                    #    self.fail()
                    
                    self.assertAlmostEqual(P, J_ij, delta=0.001)
                    ann.Thetas = copy.deepcopy(T_original)
    
if __name__ == "__main__":
    unittest.main()
    
