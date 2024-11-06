#!/usr/bin/env python3

# Python imports.
from __future__ import print_function
# from ngrams import create_ngrams, get_vector, find_most_similar_vector
# from trajectory import save_traj_to_file, create_optimal_trajectories, visualize_trajectory
# from interactivePlot import InteractivePlot
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
    embedding_space.new_model()
    # embedding_space.load_model()
    # embedding_space.test_parallelogram()
    # -------------------------
    

    
    # paralellogram = InteractivePlotABCD(embedding_space)
    # -------------------------
    # -----------------------
    




if __name__ == "__main__":
    main()