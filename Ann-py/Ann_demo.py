from Ann import Ann

def demo_helper():
    print('\t** Learn the AND function using 0 hidden layers (logistic regression) **')
    arrs = []
    labels = []
    (arrs.append([0, 0]), labels.append('false')) 
    (arrs.append([0, 1]), labels.append('true'))
    (arrs.append([1, 0]), labels.append('true'))
    (arrs.append([1, 1]), labels.append('true'))
    num_hidden_layers = 0
    ann = Ann(arrs, labels, n_h=num_hidden_layers)
    ann.train()
    if (ann.validate() == 1):
        print('\t** The AND function was learned correctly using 0 hidden layers **\n')
    else:
        print('\t** ERROR (when learning the AND function using 0 hidden layers **\n')
    
    print('\t** Learn the AND function using 1 hidden layer **')
    arrs = []
    labels = []
    (arrs.append([0, 0]), labels.append('false')) 
    (arrs.append([0, 1]), labels.append('true'))
    (arrs.append([1, 0]), labels.append('true'))
    (arrs.append([1, 1]), labels.append('true'))
    num_hidden_layers = 1
    ann = Ann(arrs, labels, n_h=num_hidden_layers)
    ann.train()
    if (ann.validate() == 1):
        print('\t** The AND function was learned correctly using 1 hidden layers **\n')
    else:
        print('\t** ERROR (when learning the AND function using 1 hidden layers **\n')
        
    print('\t** Learn the XOR function using 0 hidden layers (logistic regression) **')
    arrs = []
    labels = []
    (arrs.append([0, 0]), labels.append('false')) 
    (arrs.append([0, 1]), labels.append('true'))
    (arrs.append([1, 0]), labels.append('true'))
    (arrs.append([1, 1]), labels.append('false'))
    num_hidden_layers = 0
    ann = Ann(arrs, labels, n_h=num_hidden_layers)
    ann.train()
    if (ann.validate() != 1):
        print('\t** The XOR function was not learned correctly (as expected) because logistic regression (0 hidden layers) \n' + 
              'cannot create a boundary through a non-linearly separable data-set (which the XOR function is)**\n')
    else:
        print('\t** ERROR (when learning the XOR function using 0 hidden layers **\n')
    
    print('\t** Learn the XOR function using 1 hidden layer **')
    arrs = []
    labels = []
    (arrs.append([0, 0]), labels.append('false')) 
    (arrs.append([0, 1]), labels.append('true'))
    (arrs.append([1, 0]), labels.append('true'))
    (arrs.append([1, 1]), labels.append('false'))
    num_hidden_layers = 1
    ann = Ann(arrs, labels, n_h=num_hidden_layers)
    ann.train()
    if (ann.validate() == 1):
        print('\t** The XOR function was learned correctly using 1 hidden layers **\n')
    else:
        print('\t** ERROR (when learning the XOR function using 1 hidden layers **\n')
    
if __name__ == '__main__':
    demo_helper()
    