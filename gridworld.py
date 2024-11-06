#!/usr/bin/env python3

# Python imports.
from __future__ import print_function
from embedding_space import EmbeddingSpace
from interactivePlotABCD import InteractivePlotABCD


def main():
    # -------------------------
    # -----------------------
    # STEP 1: Create Trajectories (via optimal policies)
    NUM_TRAJECTORIES = 100
    # STEP 2: Create embeddings of the trajectories (using whatever technique)
    # Options for n-gram type include: "state-action", "action-reward", "states", "actions", "state-action-reward", "state-reward"
    N_GRAM_TYPE = "state-reward"
    

    embedding_space = EmbeddingSpace(NUM_TRAJECTORIES, N_GRAM_TYPE)
    # embedding_space.new_model()
    embedding_space.load_model()
    # -------------------------
    
    InteractivePlotABCD(embedding_space)
    # -------------------------
    # -----------------------
    




if __name__ == "__main__":
    main()