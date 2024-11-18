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

class AE(nn.Module):

    def __init__(self, X_train, y_train, epochs, batch_size, vector_size, vocab_size, model_save_file, processed_data, latent_dim=10, learning_rate=1e-3):
        # Define the model architecture
        self.model = Sequential()


        # Input layer: Document vectors
        self.model.add(Embedding(input_dim=len(processed_data), output_dim=vector_size, name="doc_embedding"))

        # Output layer: Prediction of context word using softmax
        self.model.add(Dense(vocab_size, activation='softmax'))
    

        # Compile the model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        # Train the model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

        # After training, we can extract the document and word embeddings
        doc_embedding_layer = self.model.get_layer('doc_embedding')

        # Extract document vectors
        self.doc_vectors = doc_embedding_layer.get_weights()[0]  

        # Save the model so we can retrive it later
        self.model.save(model_save_file)
        