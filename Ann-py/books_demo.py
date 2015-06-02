'''
Created on Jun 1, 2015

@author: Alex Ilchenko
'''

import os
import re
from Book import Book
import codecs
import pickle
import random
import math
from Ann import Ann  # My personal implementation of a feed-forward neural network
from collections import OrderedDict

def main():
    # An array of all text files
    dir = '../library/books/'
    
    # Using pickle so I don't keep re-reading these books
    print('\n\nReading books..')
    books = []
    if (os.path.exists(dir + '../my_books')):
        books = pickle.load(open(dir + '../my_books', 'rb'))
    else:
        # Just use the first 10 books
        file_names = [name for name in os.listdir(dir)][0:10]
        for file_name in file_names:
            m = re.search('(.*?)_(.*?)\.txt', file_name)
            # Get the author from the text file name
            author = re.sub(r'([A-Z])', r' \1', m.group(1)).strip()
            # Get the title from the text file name
            title = m.group(2).strip()
            f = codecs.open('../library/books/' + file_name, 'r', encoding='utf-8', errors='ignore')
            # print(author + ' ' + title)
            lines = f.readlines()
            book = Book(author, title, lines)
            books.append(book)
        
        pickle.dump(books, open(dir + '../my_books', 'wb'))
    for book in books:
        print(book.title + ' by ' + book.author + '\t\t has ' + str(len(book.sentences)) + ' sentences.')
        
    n = 2  # The size of our n-grams (we choose to use bi-grams)
    
    print('\n\nMaking a vocabulary of n-grams...')
    # Using pickle so I don't keep re-making a vocabulary
    n_gram_vocab = []
    if (os.path.exists(dir + '../my_n_grams')):
        n_gram_vocab = pickle.load(open(dir + '../my_n_grams', 'rb'))
    else:
        n_gram_vocab = {}  # Treated as a set (faster 'in' operation than list)
        for book in books:
            # print(book.author + ' ' + book.title)
            # print(len(n_gram_vocab))
            n_gram_vocab = add_to_n_gram_vocab(n_gram_vocab, book.sentences, n=n)
        
        # n_gram_vocab = OrderedDict(n_gram_vocab)  # Convert to an ordered list
        n_gram_vocab = list(n_gram_vocab.keys())  # Convert to an ordered list
        pickle.dump(n_gram_vocab, open(dir + '../my_n_grams', 'wb'))
        
    print('There are ' + str(len(n_gram_vocab)) + ' n-grams of size ' + str(n))
    
    print('\n\nBuilding a labeled data-set...')
    # We will do our training and testing on samples where a sample is a 5 sentence continuous text
    # Chunks are further broken down into a train and test sets by Ann
    # We look for the book with the smallest number of sentences and then get 50% of all of its 5-sentence chunks
    # For every other book, we randomly sample the same number of chunks (all labels have the same number of data points)
    
    arrs = []  # Holds vectorial representation of our 5-sentence chunks
    labels = []  # Holds the corresponding labels (author + title) of our chunks
    
    chunk_length = 5
    percentage = 0.5 
    
    # Get minimum number of sentences across all our books
    min_num_sentences = -1
    for book in books:
        if (len(book.sentences) < min_num_sentences or min_num_sentences == -1):
            min_num_sentences = len(book.sentences)
        
    for book in books:
        # We can't start a chunk at the last 4 sentences
        num_chunks = min_num_sentences - chunk_length + 1
        this_num_sentences = len(book.sentences) - chunk_length + 1
        num_samples = int(math.floor(num_chunks * percentage))
        # Randomly pick 50% of all 5-sentence chunks
        samples = random.sample(range(0, this_num_sentences), num_samples)
        
        label = book.title + ' by ' + book.author
        print(label)
        # Convert our sampled 5-sentence chunks into vectors
        for sample in samples:
            # print(sample)
            # Take some 5-sentence chunk
            chunk = book.sentences[sample:sample + chunk_length + 1]
            chunk = ''.join(str(elem + ' ') for elem in chunk)
            v = sen_2_vec(chunk, n_gram_vocab, n=n)
            arrs.append(v)
            labels.append(label)
            
    print('\n\nTraining logistic regression classifier using Ann...')
    ann = Ann(arrs, labels, n_h=0)  # n_h=0 means we are using 0 hidden layers
    ann.train(lam=100)
    
    print('\n\nFinding the top 5 most distinguishing bi-grams...')
    for k in range(0, len(books)):  # Number of classes
        v = ann.Thetas[0][k, :].tolist()[0]
        s = sorted((e, i) for i, e in enumerate(v))
        s.reverse()
        print(books[k].title + ' by ' + books[k].author)
        for i in range(0, 5):
            print(n_gram_vocab[s[i][1]])
         
def sen_2_vec(sentence, vocab, **kwargs):
    if ('n' in kwargs.keys()):
        n = kwargs['n']
    else:
        n = 2  # Default to bi-grams
    v = [0] * len(vocab)
    n_grams = get_n_grams(sentence)
    for n_gram in n_grams:
        if (n_gram in vocab):
            index = vocab.index(n_gram)
            v[index] = 1
    return v

def add_to_n_gram_vocab(vocab, sentences, **kwargs):
    if ('n' in kwargs.keys()):
        n = kwargs['n']
    else:
        n = 2  # Default to bi-grams
        
    for sentence in sentences:
        n_grams = get_n_grams(sentence, n=n)
        
        # Keep only new n_grams (we don't care about frequency right now)
        for n_gram in n_grams:
            vocab[n_gram] = 1
            
    return vocab
        
def get_n_grams(sentence, **kwargs):
    # Assume sentence is lower case and contains only alphanumerics
    if ('n' in kwargs.keys()):
        n = kwargs['n']
    else:
        n = 2  # Default to bi-grams
    
    words = sentence.split(' ')
    
    # Note: there are at most len(words)-n unique n-grams
    n_grams = []
    for i in range(0, len(words) - n + 1):
        n_gram = ''.join(str(elem + ' ') for elem in words[i:i + n])
        n_grams.append(n_gram.strip())
    return n_grams    

if __name__ == '__main__':
    main()
    
