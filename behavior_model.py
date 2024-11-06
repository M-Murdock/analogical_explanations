# Reference: https://spotintelligence.com/2023/09/06/doc2vec/
import pickle
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense
import numpy as np
from tensorflow.keras.optimizers import Adam
import keras


class BehaviorModel:
    def __init__(self, training_file):
        self.training_data = None
        
        with open(training_file, "rb") as fp:   
            self.training_data = pickle.load(fp)
        
        # convert document to lists of sentences
        self.documents = self._sent_to_list()
        self.processed_data = [self._preprocess(doc) for doc in self.documents]

        # self.train()
        
    def load(self):
        self.model = keras.models.load_model("traj_model.keras")
        self.doc_vectors = self.model.get_layer('doc_embedding').get_weights()[0]
        
    def _preprocess(self, doc):
        words = doc.split(" ")
        return words
        
    def _sent_to_list(self):
        return self.training_data.split(';')
    
    def train(self):
        # PV-DBOW (Distributed Bag of Words version of Paragraph Vector): In this approach, the 
        # model predicts words independently based solely on the document vector. It treats each 
        # document as a “bag of words” and tries to predict words randomly sampled from that bag.
        # Build vocabulary
        all_words = [word for doc in self.processed_data for word in doc]
        vocab = Counter(all_words)
        word2index = {word: i for i, (word, _) in enumerate(vocab.items())}

        index2word = {i: word for word, i in word2index.items()}
        vocab_size = len(vocab)

        # Parameters
        vector_size = 10        # Dimension of word vectors and document vectors
        window_size = 2         # Context window size
        epochs = 10 #100        # Number of training epochs
        batch_size = 1          # Batch size (for simplicity, we'll train one document at a time)

        # Generate training data (context words and document labels)
        training_data = []

        for doc_id, doc in enumerate(self.processed_data):
            for i, word in enumerate(doc):
                # Define the context window for the word
                context = []
                # Gather words within the window around the target word
                for j in range(max(0, i - window_size), min(len(doc), i + window_size + 1)):
                    if j != i:
                        context.append(doc[j])
                
                # Create pairs of document ID (label) and context word
                for target_word in context:
                    training_data.append((doc_id, word, target_word))

                # Convert words to indices
        X_train = np.array([doc_id for doc_id, _, _ in training_data])    # Document ID as input
        y_train = np.array([word2index[target_word] for _, _, target_word in training_data])  # Context word as target

        # Define the model architecture
        self.model = Sequential()


        # Input layer: Document vectors
        # model.add(Embedding(input_dim=len(self.processed_data), output_dim=vector_size, name="doc_embedding"))
        self.model.add(Embedding(input_dim=len(self.processed_data), output_dim=vector_size, name="doc_embedding"))

        # # Hidden layer: Word vectors
        # model.add(Embedding(input_dim=vocab_size, output_dim=vector_size, name="word_embedding"))

        # # Output layer: Prediction of context word using softmax
        self.model.add(Dense(vocab_size, activation='softmax'))

        # Compile the model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # After training, we can extract the document and word embeddings
        doc_embedding_layer = self.model.get_layer('doc_embedding')
        # word_embedding_layer = model.get_layer('word_embedding')

        # Extract document vectors
        self.doc_vectors = doc_embedding_layer.get_weights()[0]  # Shape: (num_documents, vector_size)

        # Extract word vectors
        # word_vectors = word_embedding_layer.get_weights()[0]  # Shape: (vocab_size, vector_size)

        # Test: Get the vector for a document
        doc_id = 0  # Example: Document ID 0
        print(f"Document vector for document {doc_id}:")
        print(self.doc_vectors[doc_id])
        
        self.model.save("traj_model.keras")
        
    