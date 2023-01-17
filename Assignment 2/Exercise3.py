# -*- coding: utf-8 -*-
"""
Assignment 2, excercise 3
using Word2Vec and WordNet (use at least 500 data points):
• Which word pairs appear in more similar contexts, synonyms, or antonyms?
• Which word pairs appear in more similar contexts, hypernyms, or hyponyms?

"""

import nltk
import numpy as np
import pandas as pd
import gensim
import random
from typing import Dict, List, Tuple
from nltk.corpus import wordnet
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt

###### W2V embedding ##########

# finding word2vec embeddings in our harddrive
path_to_word2vec_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')
print('Loading Distributional Semantic Model (DSM) Word2Vec...')
word2vec_gensim = gensim.models.KeyedVectors.load_word2vec_format(path_to_word2vec_sample)

###### synonyms and antonyms ##########

def compare_word_pairs_SYN_ANTONYMS(sim1: List, sim2:List, limit: int, model:gensim.models.keyedvectors.KeyedVectors):
    for ss in wordnet.all_synsets():
        # get same amount of word pairs:
        assert len(sim1) == len(sim1)
        # check that synset has multiple lemmas (synonyms) and at least one antonyms,
        if not len(ss.lemma_names()) > 1 or not len(ss.lemmas()[0].antonyms()) > 0:
            continue
        synonyms = ss.lemma_names()
        antonyms = [ss.lemmas()[0].antonyms()[0].name()] #only one
        # check whether all synonyms and antonyms are in model-vocab
        model_vocab= list(model.key_to_index.keys())
        if not all(item in model_vocab for item in synonyms):
            continue
        if not all(item in model_vocab for item in antonyms):
            continue
        
        word1=synonyms[0]
        temp = [] # calculate cosine-similarity for all synonyms and average
        for word2 in synonyms[1:]:
            temp.append(model.similarity(word1, word2))
        sim1.append(np.mean(temp))
    
        temp = []# calculate cosine-similarity for all antonyms and average
        for word2 in antonyms:
            temp.append(model.similarity(word1, word2))
            print(word1, word2)
        sim2.append(np.mean(temp))
        
        if len(sim1) == limit:
            break
    print(f'Mean synonym-similarity: {np.mean(sim1)}')
    print(f'Mean antonyms-similarity: {np.mean(sim2)}')
    print(f'Significant difference? t-test: p: {ttest_ind(sim1, sim2)[1]<=0.05}')
    print(f' t-test: p: {ttest_ind(sim1, sim2)[1]}')
    #plot
    fig, ax = plt.subplots()
    ax.boxplot([sim1, sim2], widths=0.5)
    plt.xticks([1, 2], ['Synonyms', 'Antonyms'])
    plt.xlabel("Distribution")
    plt.ylabel("Similarity")
    plt.show()
    

syn_similarities, ant_similarities = [], []
compare_word_pairs_SYN_ANTONYMS(syn_similarities, ant_similarities, 500, word2vec_gensim)

    
###### hypernyms and hyponyms ##########

def compare_word_pairs_HYPER_HYPO(sim1: List, sim2:List, limit: int, model:gensim.models.keyedvectors.KeyedVectors):
    for ss in wordnet.all_synsets():
        # get same amount of word pairs:
        assert len(sim1) == len(sim1)
        # check that synset has multiple lemmas (synonyms) and at least one hypernyms(hyponym),
        if not len(ss.lemma_names()) >= 1 or not len(ss.hypernyms()) > 0 or not len(ss.hyponyms()) > 0:
            continue
        synonyms = ss.lemma_names()
        hyponyms = ss.hyponyms()[0].lemma_names()
        hypernyms = ss.hypernyms()[0].lemma_names()
        
        #print(hyponyms, hypernyms)
        # check whether all in model-vocab
        model_vocab= list(model.key_to_index.keys())
        if not all(item in model_vocab for item in synonyms):
            continue
        if not all(item in model_vocab for item in hyponyms):
            continue
        if not all(item in model_vocab for item in hypernyms):
            continue
        
       # word1=synonyms[0] 
        #print(word1)
        temp = [] # calculate cosine-similarity for all synonyms or specific word/hyponyms and average
        for word1 in synonyms:
            #word1=synonyms[0]
            for word2 in hyponyms:
                temp.append(model.similarity(word1, word2))
        sim1.append(np.mean(temp))
    
        temp = []# calculate cosine-similarity for all antonyms and average
        for word1 in synonyms:
            #word1=synonyms[0]
            for word2 in hypernyms:
                temp.append(model.similarity(word1, word2))
            #print(word1, word2)
        sim2.append(np.mean(temp))
        
        if len(sim1) == limit:
            break
    print(f'Mean hyponyms-similarity: {np.mean(sim1)}')
    print(f'Mean hypernyms-similarity: {np.mean(sim2)}')
    print(f'Significant difference? t-test: p: {ttest_ind(sim1, sim2)[1]<=0.05}')
    print(f' t-test: p: {ttest_ind(sim1, sim2)[1]}')
    #plot
    fig, ax = plt.subplots()
    ax.boxplot([sim1, sim2], widths=0.5)
    plt.xticks([1, 2], ['Hyponyms', 'Hypernyms'])
    plt.xlabel("Distribution")
    plt.ylabel("Similarity")
    plt.show()
   
hyper_similarities, hypo_similarities = [], []
compare_word_pairs_HYPER_HYPO(hyper_similarities, hypo_similarities, 500, word2vec_gensim)
