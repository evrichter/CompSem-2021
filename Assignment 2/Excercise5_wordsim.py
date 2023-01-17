# -*- coding: utf-8 -*-
"""
Assignment 2, excercise 5 Word similarity/relatedness: based on the correlation between cosines for word pairs in
semantic space and word pair similarity/relatedness scores given by humans.

evaluate one or more distributional model of
English of choice using the WordSim353 dataset, in the version that distinguishes between
similarity and relatedness (files wordsim_relatedness_goldstandard.txt and
wordsim_similarity_goldstandard.txt from
http://alfonseca.org/eng/research/wordsim353.html). In your error analysis, make sure to
address the following Qs: Which notion do word embeddings model better, similarity, or
relatedness? For which kinds of semantic phenomena are human vs. model similarities the
most dissimilar? (Both for similarity and relatedness subsets.)
"""

import nltk
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import gensim
from nltk.corpus import wordnet
from typing import Dict, List, Tuple
from scipy.stats import ttest_ind
import os
from ranking import *
from IPython.display import display, HTML
from IPython.utils import io
dsm_file = 'cc.en.300.vec.gz'
if not os.path.exists(dsm_file):
  print('Downloading Distributional Semantics model ...')
  with io.capture_output() as captured:
    !wget 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz' ;


###### WordSim ##########

#path to data
path_wordsim= 'ws353simrel/'
gold_rel, gold_sim = {}, {} #dict match human_gold scores with word pairs
# read in data:
for file in ["wordsim_relatedness_goldstandard.txt", "wordsim_similarity_goldstandard.txt"]:
    with open(path_wordsim + file) as f:
        d = {}
        for line in f:
            l=line.strip().split('\t')
            d[(l[0],l[1])]=float(l[2])
        if "rel" in str(file):
            gold_rel = d
        else:
            gold_sim = d

###### Embeddings ##########

def compute_scores(gold_dict: Dict, model, wordsim="WordSim353"):
    ''' calculate cosines for word pairs in gold_dict of models's semantic space 
    and compare to word pair similarity/relatedness scores given by humans'''
    manual_dict, auto_dict = {}, {}
    not_found, total_size = [], 0
    for pair, score in gold_dict.items():
        word1, word2, val = pair[0], pair[1], score
        if word1 in model and word2 in model:
            manual_dict[(word1, word2)] = float(val)
            auto_dict[(word1, word2)] = model.similarity(word1, word2)
        else:
            not_found.append((word1, word2))
        total_size += 1
    print(f'Evaluate calculated scores on: {wordsim}')
    print(f'Total number of pairs: {total_size}')
    print(f'Number of pairs not found: {len(not_found)}')
    if len(not_found) > 0:
        print('\n'.join(map(str, not_found)))
    print(f'Spearmans_Rho: {spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))}\n')
    return plotting(manual_dict, auto_dict, wordsim)

def embedding_info(model):
    print(f'Dimensions: {model.vector_size}')
    print(f'Number of vectors: {len(model)}')

def plotting(gold, auto, name):
    '''get sorted scores of humans,
    compare to calculated ones and plot'''
    pair_g, val_g = zip(*[(key,value) for key, value in sorted(gold.items(), key=lambda x: x[1], reverse=True)])
    val_auto = []
    for key in pair_g:
        val_auto.append(auto[key]*10)
    
    #print("scores that have a difference greater than 5")
    difference=[]
    for word_pair, values in zip(pair_g, zip(val_g,val_auto)):
        #if abs(values[0]-values[1])>5:
            #print(word_pair)
        difference.append(abs(values[0]-values[1]))
    # dictionary of lists 
    dict = {'word_pair': pair_g, 'human': val_g, 'score': val_auto, 'difference': difference} 
    df = pd.DataFrame(dict)
    return df
    '''plt.plot(val_g, label='human')
    plt.legend()
    plt.plot(val_auto, 'r+')
    plt.title(name, fontsize=18)
    plt.ylabel('score', fontsize=14)
    plt.xlabel('rank', fontsize=14)
    plt.show()'''
    return df
    
# finding word2vec embeddings in our harddrive
path_to_word2vec_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')
# /home/nora/nltk_data/models/word2vec_sample/pruned.word2vec.txt'

# loading embeddings. With gensim, we can load our embeddings directly into a dictionary
print('Loading Distributional Semantic Model (DSM) Word2Vec...')
word2vec_gensim = gensim.models.KeyedVectors.load_word2vec_format(path_to_word2vec_sample)
embedding_info(word2vec_gensim)
print('Computing cosines for word pairs, Word2Vec')
df1= compute_scores(gold_rel, word2vec_gensim, "WordSim353_relatedness")
#df1.to_csv('CSV_W2V_Rel:Results.csv', sep='\t', columns=["word_pair", "human", "score", "difference"], header=["word pair", "human", "score", "difference"], index=True)
compute_scores(gold_sim, word2vec_gensim, "WordSim353_similarity")
#df1_sim= compute_scores(gold_sim, word2vec_gensim, "WordSim353_similarity")
#df1_sim.to_csv('CSV_W2V_Sim:Results.csv', sep='\t', columns=["word_pair", "human", "score", "difference"], header=["word pair", "human", "score", "difference"], index=True)

print('Loading DSM Fasttext ...')
distributional_semantic_model = gensim.models.KeyedVectors.load_word2vec_format(dsm_file)
embedding_info(distributional_semantic_model)
print('Computing cosines for word pairs, Fasttext')
df2_rel=compute_scores(gold_rel, distributional_semantic_model, "WordSim353_relatedness")
df2_rel.to_csv('CSV_Fasttext_Rel:Results.csv', sep='\t', columns=["word_pair", "human", "score", "difference"], header=["word pair", "human", "score", "difference"], index=True)
df2_sim=compute_scores(gold_sim, distributional_semantic_model, "WordSim353_similarity")
df2_sim.to_csv('CSV_Fasttext_Sim:Results.csv', sep='\t', columns=["word_pair", "human", "score", "difference"], header=["word pair", "human", "score", "difference"], index=True)

#print('Loading DSM Glove ...')
#data_test.to_csv('CSV_Test_Results.csv', sep='\t', columns=[""], header=["gold labels", "scores", "words"], index=True)

