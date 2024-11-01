#!/usr/bin/env python3

# Python imports.
from __future__ import print_function
from ngrams import create_ngrams, get_vector, find_most_similar_vector
# from trajectory import save_traj_to_file, create_optimal_trajectories, visualize_trajectory
from interactivePlot import InteractivePlot
from embedding_space import EmbeddingSpace


def main():
    # -------------------------
    # -----------------------
    # STEP 1: Create Trajectories (via optimal policies)
    NUM_TRAJECTORIES = 10
    # STEP 2: Create embeddings of the trajectories (using whatever technique)
    # Options for n-gram type include: "state-action", "action-reward", "states", "actions", "state-action-reward", "state-reward"
    N_GRAM_TYPE = "state-reward"
    
    # optimal_trajectories, rewards, ql_agents, mdps = create_optimal_trajectories(map_name="maps/easygrid.txt", num_agents=NUM_TRAJECTORIES, episodes=1000, steps=200, slip_prob=0.1)
    embedding_space = EmbeddingSpace(NUM_TRAJECTORIES, N_GRAM_TYPE)
    # -------------------------
    # -----------------------
    
    # -------------------------
    # -----------------------
    
    
    # save the trajectories to a file so that we can turn them into embeddings later
    # save_traj_to_file(optimal_trajectories, rewards, file_name=("saved_trajectories/" + N_GRAM_TYPE + ".txt"), ngram_type=N_GRAM_TYPE) # NOTE: this can be commented out
    # create embeddings
    # model, trajectory_objs = create_ngrams(("saved_trajectories/" + N_GRAM_TYPE + ".txt"))
    # -------------------------
    # -----------------------
    
    
    # TODO: We need to establish common ground first
    
    # -------------------------
    # -----------------------
    # STEP 3: Use parallelogram method
    
    # A:B :: C:D
    # Choose an "A" trajectory and its analogous "B" trajectory
    # A_TRAJECTORY_INDEX = 1
    # B_TRAJECTORY_INDEX = 2

    # a_trajectory = get_vector(A_TRAJECTORY_INDEX, model, trajectory_objs)
    # b_trajectory = get_vector(B_TRAJECTORY_INDEX, model, trajectory_objs)
    
    # # Find the difference between the two trajectories (the difference vector)
    # diff_vector = a_trajectory - b_trajectory
    
    # # Choose a "C" trajectory
    # C_TRAJECTORY_INDEX = 3
    # c_trajectory = get_vector(C_TRAJECTORY_INDEX, model, trajectory_objs)
    
    # approximate_vector = c_trajectory + diff_vector
    # # get a list of trajectories in order of similarity, from highest to lowest
    # unfiltered_similarity_ranking = find_most_similar_vector(approximate_vector, model)
    # # we need to filter out the A,B, and C trajectories 
    # D_TRAJECTORY_INDEX = 0
    # for r in unfiltered_similarity_ranking:
    #     if r[0] == A_TRAJECTORY_INDEX or r[0] == B_TRAJECTORY_INDEX or r[0] == C_TRAJECTORY_INDEX:
    #         pass 
    #     else:
    #         D_TRAJECTORY_INDEX = r[0]
    #         break
    # d_trajectory = get_vector(D_TRAJECTORY_INDEX, model, trajectory_objs)
    

    
    # paralellogram = InteractivePlot(all_vector_embeddings=[get_vector(i, model, trajectory_objs) for i in range(0, len(trajectory_objs))], embedding_indices=[A_TRAJECTORY_INDEX, B_TRAJECTORY_INDEX, C_TRAJECTORY_INDEX, D_TRAJECTORY_INDEX], optimal=optimal_trajectories)
    paralellogram = InteractivePlot(embedding_space)
    # -------------------------
    # -----------------------
    




if __name__ == "__main__":
    main()