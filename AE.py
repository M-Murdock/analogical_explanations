"""
This code mainly follows the Geeks4Geeks Pytorch Autoencoder example.
https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/

Any modifiations are made by the AABL Lab.
"""

import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.optimizers import Adam
import pickle
from collections import Counter
import numpy as np
import keras

class AE(nn.Module):

    def __init__(self, training_file, model_save_file):
        self.model_save_file = model_save_file
        
        self.training_data = None
        self.model_save_file = model_save_file
        
        with open(training_file, "rb") as fp:   
            self.training_data = pickle.load(fp)
        
        # convert document to lists of sentences
        self.documents = self._sent_to_list()
        self.processed_data = [self._preprocess(doc) for doc in self.documents]
        
        # Parameters
        self.vector_size = 3       # Dimension of word vectors and document vectors
        self.window_size = 2         # Context window size
        self.epochs = 100        # Number of training epochs
        self.batch_size = 1          # Batch size (for simplicity, we'll train one document at a time)
        
        # self.processed_data = [] 
        # self.vocab_size = 0
        # with open(self.model_save_file, "rb") as fp:   
        #     self.training_data = pickle.load(training_file)
            
    def process_data(self):
        # Build vocabulary
        all_words = [word for doc in self.processed_data for word in doc]
        vocab = Counter(all_words)
        word2index = {word: i for i, (word, _) in enumerate(vocab.items())}

        # index2word = {i: word for word, i in word2index.items()}
        self.vocab_size = len(vocab)
        

        # Generate training data (context words and document labels)
        self.training_data = []

        for doc_id, doc in enumerate(self.processed_data):
            for i, word in enumerate(doc):
                # Define the context window for the word
                context = []
                # Gather words within the window around the target word
                for j in range(max(0, i - self.window_size), min(len(doc), i + self.window_size + 1)):
                    if j != i:
                        context.append(doc[j])
                
                # Create pairs of document ID (label) and context word
                for target_word in context:
                    self.training_data.append((doc_id, word, target_word))

        # Convert words to indices
        self.X_train = np.array([doc_id for doc_id, _, _ in self.training_data])    # Document ID as input
        self.Y_train = np.array([word2index[target_word] for _, _, target_word in self.training_data])  # Context word as target
        
    def _preprocess(self, doc):
        words = doc.split(" ")
        return words
        
    def _sent_to_list(self):
        return self.training_data.split(';')
            
    def train(self):
        # Define the model architecture
        self.model = Sequential()

        # Input layer: Document vectors
        self.model.add(Embedding(input_dim=len(self.processed_data), output_dim=self.vector_size, name="doc_embedding"))

        # Output layer: Prediction of context word using softmax
        self.model.add(Dense(self.vocab_size, activation='softmax'))
    

        # Compile the model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        # Train the model
        self.model.fit(self.X_train, self.Y_train, epochs=self.epochs, batch_size=self.batch_size)

        # After training, we can extract the document and word embeddings
        doc_embedding_layer = self.model.get_layer('doc_embedding')

        # Extract document vectors
        self.doc_vectors = doc_embedding_layer.get_weights()[0]  

        # Save the model so we can retrive it later
        self.model.save(self.model_save_file)
        
    def load(self):
        self.model = keras.models.load_model(self.model_save_file)
        self.doc_vectors = self.model.get_layer('doc_embedding').get_weights()[0]