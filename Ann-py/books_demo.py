'''
Created on Jun 1, 2015

@author: Alex Ilchenko
'''

import os
import re

def main():
    # An array of all text files
    file_names = [name for name in os.listdir('../books/')]
    for file_name in file_names:
        m = re.search('(.*?)_(.*?)\.txt', file_name)
        # Get the author from the text file name
        author = re.sub(r'([A-Z])', r' \1', m.group(1)).strip()
        # Get the title from the text file name
        title = re.sub(r'([A-Z])', r' \1', m.group(2)).strip()
        f = open(file_name, 'r')
        lines = f.readlines()
               

if __name__ == '__main__':
    main()
    