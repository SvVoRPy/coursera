# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:31:39 2018

@author: SvenV
"""

import matplotlib.pyplot as plt          # plotting
import numpy as np                       # dense matrices
from scipy.sparse import csr_matrix      # sparse matrices

import os
path = 'C:/Users/SvenV/Desktop/Coursera Kurse/Machine Learning Spezialization IV/Week2'
os.chdir(path)
os.listdir(path)

import pandas as pd
import numpy as np

wiki = pd.read_csv('people_wiki.csv')

def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    
    return csr_matrix( (data, indices, indptr), shape)

word_count = load_sparse_csr('people_wiki_word_count.npz')

from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)

print wiki[wiki['name'] == 'Barack Obama']

distances, indices = model.kneighbors(word_count[35817], n_neighbors=10)

neighbors = pd.DataFrame({'distance':distances.flatten(), 'id':indices.flatten()})

wiki.merge(neighbors, how='inner', left_index = True, right_on='id')[['id','name','distance']]

map_index_to_word = pd.read_json('people_wiki_map_index_to_word.json', typ = 'series')

map_index_to_word = pd.Series(map_index_to_word.index.values, index=map_index_to_word.values)

def unpack_dict(matrix, map_index_to_word):
    table = sorted(map_index_to_word, key=map_index_to_word.get)
   
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    
    num_doc = matrix.shape[0]

    return [{k:v for k,v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i+1]] ],
                                 data[indptr[i]:indptr[i+1]].tolist())} \
               for i in xrange(num_doc) ]

wiki['word_count'] = unpack_dict(word_count, map_index_to_word)

def top_words(name):
    """
    Get a table of the most frequent words in the given person's wikipedia page.
    """
    row = wiki[wiki['name'] == name]
    word_count_table = pd.DataFrame.from_dict(row['word_count'].values[0], orient='index').reset_index(inplace=False)
    word_count_table.rename(columns = {'index':'word', 0:'count'}, inplace=True)  
    return word_count_table.sort('count', ascending=False)

obama_words = top_words('Barack Obama')
print obama_words

barrio_words = top_words('Francisco Barrio')
print barrio_words

combined_words = obama_words.merge(barrio_words, on='word', suffixes=['_obama','_barrio'])
# 5 most frequent obama words overlapping
common_words = set(combined_words.sort_values(by='count_obama',ascending=False)[0:5]['word'])

# Check for all others
def has_top_words(word_count_vector):
    # extract the keys of word_count_vector and convert it to a set
    unique_words = set(word_count_vector.keys())
    # return True if common_words is a subset of unique_words
    # return False otherwise
    return unique_words.issubset(common_words)

wiki['has_top_words'] = wiki['word_count'].apply(has_top_words)

# use has_top_words column to answer the quiz question
wiki['has_top_words']

temp = wiki[wiki['name'].isin(['Barack Obama','Joe Biden','George W. Bush'])]['word_count']

from sklearn.metrics.pairwise import euclidean_distances

import math
math.sqrt(sum((temp.iloc[0].get(d,0) - temp.iloc[1].get(d,0))**2 for d in set(temp.iloc[0]) | set(temp.iloc[1])))
math.sqrt(sum((temp.iloc[0].get(d,0) - temp.iloc[2].get(d,0))**2 for d in set(temp.iloc[0]) | set(temp.iloc[2])))
math.sqrt(sum((temp.iloc[1].get(d,0) - temp.iloc[2].get(d,0))**2 for d in set(temp.iloc[1]) | set(temp.iloc[2])))

# Same word in Obama und Bush

bush_words = top_words('George W. Bush')

combined_words = obama_words.merge(bush_words, on='word', suffixes=['_obama','_bush'])
# 5 most frequent obama words overlapping
common_words = set(combined_words.sort_values(by='count_obama',ascending=False)[0:10]['word'])






