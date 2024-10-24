# Python program to generate word vectors using Word2Vec
# https://tedboy.github.io/nlps/generated/generated/gensim.models.Word2Vec.html
# https://radimrehurek.com/gensim/models/keyedvectors.html 


# importing all necessary modules
from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from trajectory import _get_sa_sequences
import warnings
import nltk
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()


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
def create_ngrams(trajectories, reward, ngram_type):
    # get the lists of states and actions
    s_seq, a_seq = _get_sa_sequences(trajectories)
    
    sa_str = ""
    # save state-action pairs to the file
    for traj_num in range(0, len(s_seq)): # go through each trajectory
        for i in range(0, len(s_seq[traj_num])):
            sa_str += " " + s_seq[traj_num][i] + "" + a_seq[traj_num][i]
        sa_str += "."
            
    print("sastr")
    print(sa_str)
    f = open("state-action.txt", "w")
    f.write(sa_str)
    f.close()

    #open and read the file after the appending:
    sample = open("state-action.txt")
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
    model1 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5)
    # Print results
    print("Cosine similarity between '7-6right' " + "and '10-5right' - CBOW : ", model1.wv.similarity('7-6right', '10-5right'))

    # Create Skip Gram model
    model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)
    print("Cosine similarity between '7-6right' " + "and '10-5right' - Skip Gram : ", model2.wv.similarity('7-6right', '10-5right'))
    print("most similar to 7-6right: ", model2.wv.most_similar(positive=["7-6right"]))
    
    # vec1 = model2.wv.get_vector('7-6right')
    # vec2 = model2.wv.get_vector('10-5right')
    sent_sum = []
    for sent in data:
        # sent_sum.append([model2.wv.get_vector(w) for w in sent])
        # print("W: ", sent[1])
        temp_sent = []
        for word in sent:
            print("W: ", word)
            temp_sent.append(model2.wv.get_vector('10-5right'))
        sent_sum.append(temp_sent)
        
    # print("SIMILAR")
    # print(sent_sum)
    difference_vector = model2.wv.distance('7-6right', '10-5right')
    # print(difference_vector)
    # print(model2.wv.distance('7-6right', '10-5right'))
    # Finds two vectors which are similar to each other by some vector difference
    # print(model2.wv.similar_by_vector([1,2]))
    # print(model2.wv.similar_by_vector(difference_vector))
    
    # print(model2.wv.most_similar(positive=difference_vector))
    
    return None
    # -------------------------
    # -------------------------