#!/usr/bin/env python3

from __future__ import print_function
from embedding_space import EmbeddingSpace
from interactivePlotABCD import InteractivePlotABCD


def main():
    # -------------------------
    # -----------------------
    # STEP 1: Create Trajectories (via optimal policies)
    # STEP 2: Create embeddings of the trajectories (using whatever technique)
    # Options for n-gram type include: "state-action", "action-reward", "states", "actions", "state-action-reward", "state-reward"
    
    state_action_embedding_space = EmbeddingSpace(NUM_TRAJECTORIES=100, N_GRAM_TYPE="state-action")
    state_action_embedding_space.new_model()
    # state_action_embedding_space.load_model()
    
    # action_reward_embedding_space = EmbeddingSpace(NUM_TRAJECTORIES=100, N_GRAM_TYPE="action-reward")
    # action_reward_embedding_space.new_model()
    # action_reward_embedding_space.load_model()
    
    # states_embedding_space = EmbeddingSpace(NUM_TRAJECTORIES=100, N_GRAM_TYPE="states")
    # states_embedding_space.new_model()
    # states_embedding_space.load_model()
    
    # actions_embedding_space = EmbeddingSpace(NUM_TRAJECTORIES=100, N_GRAM_TYPE="actions")
    # actions_embedding_space.new_model()
    # actions_embedding_space.load_model()
    
    # state_action_reward_embedding_space = EmbeddingSpace(NUM_TRAJECTORIES=100, N_GRAM_TYPE="state-action-reward")
    # state_action_reward_embedding_space.new_model()
    # state_action_reward_embedding_space.load_model()
    
    # state_reward_embedding_space = EmbeddingSpace(NUM_TRAJECTORIES=100, N_GRAM_TYPE="state-reward")
    # state_reward_embedding_space.new_model()
    # state_reward_embedding_space.load_model()
    
    
    # embedding_space.load_model()
    # -------------------------
    
    InteractivePlotABCD(state_action_embedding_space)
    # -------------------------
    # -----------------------
    




if __name__ == "__main__":
    main()