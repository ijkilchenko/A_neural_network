'''Incomplete!'''

import unittest
from PIL import Image
import os
import pickle
from Ann import Ann

class Test(unittest.TestCase):


    def setUp(self):
        pass

    def tearDown(self):
        pass


    def test_1(self):
        classes = ('smiley', 'frowny')
        arrs = []
        labels = []
        file_names = []
    
        for c in classes:
            files = [name for name in os.listdir('../library/' + c)]
            for el in files:
                img = Image.open('../library/' + c + '/' + el).convert('L')
                img = img.resize((50, 50), Image.ANTIALIAS)
                arrs.append(img.getdata())
                labels.append(c)
                file_names.append(el)
        
        
        name = '../Ann-models/model_n_i_2500_n_o_34_n_h_2 2015-05-27 22:15:24.990089.annm'
        model = pickle.load(open(name, 'rb'))[0][0]
        ann = Ann(model)
        print(ann.h(arrs[0]))

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
