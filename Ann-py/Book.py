'''
Created on Jun 1, 2015

@author: Alex Ilchenko
'''

import re

class Book(object):
    
    def __init__(self, author, title, lines):
        self.author = author
        self.title = title
        self.lines = lines
        
        self.sentences = self.clean_and_segment(self.lines)
    
    def clean_and_segment(self, lines):
        # Concatenate sentences that might have been split over multiple lines
        text = ''
        for line in lines:
            clean_line = re.sub('\n', ' ', line).lower() # Also convert everything to lower case 
            text += clean_line
            
        
                