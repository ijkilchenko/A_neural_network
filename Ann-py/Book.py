'''
Created on Jun 1, 2015

@author: Alex Ilchenko
'''

import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import random
import math

class Book(object):
    
    def __init__(self, author, title, lines):
        self.author = author
        self.title = title
        self.lines = lines
        
        self.sentences = self.clean_and_segment(self.lines)
    
    def clean_and_segment(self, lines):
        # Note: we choose not to strip out stop words because these might help us understand an author's writing style
        
        # Concatenate text that might have been split over multiple lines
        text = ''
        for line in lines:
            clean_line = re.sub('\n', ' ', line)
            text = text + clean_line
        text = re.sub(r'[ ]+', r' ', text).strip()  # Collapse consecutive white spaces
        # Use NLTK to segment the text into text        
        text = sent_tokenize(text)
        
        # Clean the text for use in a machine learning application
        sentences = []
        for sentence in text:
            sentence = sentence.lower()  # Convert all to lower case
            sentence = re.sub(r'[\W_]', ' ', sentence)  # Strip out non alphanumerics
            sentence = re.sub(r'[ ]+', r' ', sentence).strip()
            sentences.append(sentence)
            
        partial_sentences = []
        # Keep only 20% of all sentences
        num = int(math.floor(len(sentences) * 0.2))
        l = random.sample(range(0, len(sentences)), num)
        for i in l:
            partial_sentences.append(sentences[i])
            
        return partial_sentences

