# Python program to generate word vectors using Word2Vec
# 
# References:
# https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.html
# https://radimrehurek.com/gensim/models/keyedvectors.html 
# https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/


# importing all necessary modules
from nltk.tokenize import sent_tokenize, word_tokenize

import warnings
import nltk
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize


# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()


def create_ngrams(file_name):
    #open and read the file of trajectories:
    sample = open(file_name)
    sentences = sample.read()
    
    sentence_words = []

    # iterate through each sentence in the file
    for i in sent_tokenize(sentences):
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            if not j == '.': 
                temp.append(j.lower())

        sentence_words.append(temp) 


    # store all the sentences as TaggedDocument objects
    trajectory_objs = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence_words)]

    # train the model on our sentences
    model = Doc2Vec(vector_size=100, min_count=3, epochs=20)
    model.build_vocab(trajectory_objs)
    model.train(trajectory_objs, total_examples=model.corpus_count, epochs=model.epochs)
    
    # return the model and list of document objects
    return model, trajectory_objs
    

def get_vector(traj_index, model, trajectory_objs):
    # select one of the vector representations of a sentence
    vector = model.infer_vector(trajectory_objs[traj_index].words)
    return vector

def find_most_similar_vector(vector, model):
    # find the vectors which are most similar to one another
    similar_doc = model.docvecs.most_similar(positive=[vector], negative=[])
    return similar_doc

# warnings.filterwarnings(action='ignore')
