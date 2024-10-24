# Python program to generate word vectors using Word2Vec
# https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.html
# https://radimrehurek.com/gensim/models/keyedvectors.html 
# https://www.analyticsvidhya.com/blog/2020/08/top-4-sentence-embedding-techniques-using-python/

# importing all necessary modules
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from trajectory import _get_sa_sequences
import warnings
import nltk
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.utils import simple_preprocess


# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

#Tokenizing with simple preprocess gensim's simple preprocess
# def sent_to_words(sentences):
#     for sentence in sentences:
#         yield(simple_preprocess(str(sentence), deacc=True)) # returns lowercase tokens, ignoring tokens that are too short or too long

def create_ngrams(file_name):
    #open and read the file of trajectories:
    sample = open(file_name)
    sentences = sample.read()
    
    # tokenized_sent = []
    # for s in word_tokenize(sentences):
    #     # print(s)
    #     tokenized_sent.append(word_tokenize(s.lower()))
        
    # sentence_words = list(enumerate(word_tokenize(sentences)))
    sentence_words = []

    # iterate through each sentence in the file
    for i in sent_tokenize(sentences):
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            if not j == '.': 
                temp.append(j.lower())

        sentence_words.append(temp)
    # print(sentence_words)
    # documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(word_tokenize(sentence_words))]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence_words)]
    # print("DOCUMENTS")
    # print(documents)
    # print(documents[0])
    # model = Doc2Vec(documents, vector_size=100, window=8, min_count=5, workers=2, dm = 1, epochs=20)
    model = Doc2Vec(vector_size=100, min_count=3, epochs=20)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    # print(f"Word '11-5right' appeared {model.wv.get_vecattr('11-5right', 'count')} times in the training corpus.")
    
    inferred_vector = model.infer_vector(documents[2].words)
    print("VECTOR")
    print(inferred_vector)
    # sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
    # print(inferred_vector)
    # similar_doc = model.doc2vec.most_similar('0')
    print("DOCS 0")
    print(documents[0][0])
    similar_doc = model.docvecs.most_similar(positive=[inferred_vector], negative=[])
    print("SIMILAR")
    print(similar_doc[0])
    # tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]
    # model = Doc2Vec(tagged_data, vector_size = 20, window = 2, min_count = 1, epochs = 100)
    # print(tagged_data)
    
    # print(model.wv.vocab)
    # query = '11-5right'
    # query_vec = model.encode([query])[0]
    # for sent in sentences:
    #     sim = cosine(query_vec, model.encode([sent])[0])
    #     print("Sentence = ", sent, "; similarity = ", sim)
    
    return None
# warnings.filterwarnings(action='ignore')
# -------------------------
# -------------------------
# word2vec algorithm, which learns an embedding in service of an objective to either 
# (1) predict the occurrence of a word given surrounding
# words (called Continuous Bag-of-Words or CBOW), or (2) predict the
# occurrence of surrounding words given the a target word (called Skip-gram)

# 1. Use 2-layer neural nets to map input words to output words. 
# 
# 2. context words are coded as a sparse vectors (i.e., with only one nonzero entry in each) are input to
# the network and embedded using a single hidden layer.
# 
# 3. These context vectors are then averaged into a single context vector and multiplied
# with a final weight matrix before being passed to the final softmax
# layer, which in the general inefficient case outputs for each word in the
# vocabulary an estimated probability of it occurring within that context
# 
# 4. The intended output words used for training are also coded as sparse
# vectors. Since, the final weight matrix computes inner products (i.e.,
# similarities) between each context and a vector of weights for each
# word in the vocabulary, these weights are taken as the “word embed-
# dings” (as opposed to context embeddings from the first layer)
# 
# 5. In Skip-gram networks, as expected, only a single word is given as input and
# context words are predicted as outputs. 
# 
# 6. Since corpus-scale vocabularies
# can be very large, a number of training optimizations are often em-
# ployed, such as hierarchical softmax and negative sampling (Mikolov
# et al., 2013).
def old_create_ngrams(file_name):

    #open and read the file of trajectories:
    sample = open(file_name)
    f = sample.read()
    

    data = []

    # iterate through each sentence in the file
    for i in sent_tokenize(f):
        temp = []

        # tokenize the sentence into words
        for j in word_tokenize(i):
            temp.append(j.lower())

        data.append(temp)

    # Create CBOW model
    cbow_model = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5)
    # Create Skip Gram model
    skip_gram_model = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)
    
    # print("Cosine similarity between '7-6right' " + "and '10-5right' - Skip Gram : ", model2.wv.similarity('7-6right', '10-5right'))
    # print("most similar to 7-6right: ", model2.wv.most_similar(positive=['7-6right']))

    # combine state-action vectors for each trajectory 
    sent_sum = []
    for sent in data:
        temp_sent = []
        for word in sent:
            temp_sent.append(skip_gram_model.wv.get_vector(word))
        sent_sum.append(temp_sent)
    
    # compute difference between two trajectories
    print("DIFF")
    # print(np.setdiff1d(sent_sum[0], sent_sum[1]))
    # print(type(sent_sum))
    # difference_vector = model2.wv.distance('7-6right', '10-5right')
    
    # print(model2.wv.distance('7-6right', '10-5right'))
    # Finds two vectors which are similar to each other by some vector difference
    print(sent_sum[0])
    print(skip_gram_model.wv.similar_by_vector(sent_sum[0], sent_sum[1]))
    # print(model2.wv.similar_by_vector(difference_vector))
    
    # print(model2.wv.most_similar(positive=difference_vector))
    
    return None
    # -------------------------
    # -------------------------