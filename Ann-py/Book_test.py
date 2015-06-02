'''
Created on Jun 1, 2015

@author: Alex Ilchenko
'''

import unittest
import re
from Book import Book


class Test(unittest.TestCase):


    def test_clean_and_segment(self):
        name = 'JackLondon_TheCalloftheWild.txt'
        m = re.search('(.*?)_(.*?)\.txt', name)
        # Get the author from the text file name
        author = re.sub(r'([A-Z])', r' \1', m.group(1)).strip()
        # Get the title from the text file name
        title = re.sub(r'([A-Z])', r' \1', m.group(2)).strip()
        f = open('../library/books/' + name, 'r')
        lines = f.readlines()
        f.close()
        
        book = Book(author, title, lines)
        self.assertEqual(author, 'Jack London')
        self.assertEqual(title, 'The Callofthe Wild')
        
        # Check to see if (a few random) sentences were properly segmented from the text 
        sen1 = 'buck lived at a big house in the sun kissed santa clara valley'
        sen2 = 'they had made short work of the snowshoe rabbit these dogs that were ill tamed wolves and they were now drawn up in an expectant circle'
        sen3 = 'that was the man buck divined the next tormentor and he hurled himself savagely against the bars'
        sen4 = 'in the tween decks of the narwhal buck and curly joined two other dogs'
        sen5 = 'one devil dat spitz remarked perrault'
        
        self.assertTrue(book.sentences.index(sen1) >= 0)
        self.assertTrue(book.sentences.index(sen2) >= 0)
        self.assertTrue(book.sentences.index(sen3) >= 0)
        self.assertTrue(book.sentences.index(sen4) >= 0)
        self.assertTrue(book.sentences.index(sen5) >= 0)

if __name__ == "__main__":
    unittest.main()
    
