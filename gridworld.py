#!/usr/bin/env python3

from __future__ import print_function
from gridworld_embedding_space import GridWorldEmbeddingSpace
from interactivePlotABCD import InteractivePlotABCD


def main():
    # -------------------------
    # -----------------------
    # STEP 1: Create Trajectories (via optimal policies)
    # STEP 2: Create embeddings of the trajectories (using whatever technique)
    # Options for n-gram type include: "state-action", "action-reward", "states", "actions", "state-action-reward", "state-reward"
    
    embedding_space = GridWorldEmbeddingSpace(TASK="four_room", N_GRAM_TYPE="state-action", load_agents=True)
    # embedding_space = GridWorldEmbeddingSpace(TASK="gridworld", N_GRAM_TYPE="state-action", load_agents=True)
    # embedding_space.new_model()
    embedding_space.load_model()
    # -------------------------
    
    InteractivePlotABCD(embedding_space)
    # -------------------------
    # -----------------------
    




if __name__ == "__main__":
    main()