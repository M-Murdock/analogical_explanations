from AE import AE
import torch
import torch.utils.data as data_utils
import numpy as np
import re

# Embedding types:
# S-A 
# S-A-Reward 

# -------------------------
# -------------------------
# Create an embedding of a specified type
def _create_vector(trajectory, r, embedding_type):
    s, a = _get_sa_sequences(trajectory)
    
    # NOTE: This is something to play around with. Maybe use an AE?
    # I'll probably have to flatten this to make it work
    if embedding_type == "state-action": 
        return [s, a]
    elif embedding_type == "state-action-reward":
        return [s, a, r]
    # use np to flatten the array
    
# -------------------------
# -------------------------


# -------------------------
# -------------------------
# NOTE: this function is specific to this gridworld environment
def _get_sa_sequences(trajectories):
    actions = []
    states = []
    
    for traj in trajectories:
        # get the list of states for the current trajectory
        s_sequence = [t[0] for t in traj]

        
        temp_state = []
        # get the two values for the current state
        for s in s_sequence:
            temp_state.append("" + str(s[0]) + "-" + str(s[1]))
            
        # Convert up/down/left/right to numbers
        a_sequence = [t[1] for t in traj]
           
           
        actions.append(a_sequence)
        states.append(temp_state)   
        # print("actions")
        # print(actions)
        # print("states")
        # print(states)

    
    return states, actions
# -------------------------
# -------------------------


# -------------------------
# -------------------------
def paralellogram_analogy(embedding_A, diff_vector):
    new_point = embedding_A - diff_vector # NOTE: We might want to add here instead of subtracting. I'm not sure.
    return find_closest_embedding(new_point)
# -------------------------
# -------------------------    

# -------------------------
# -------------------------
# NOTE: This is buggy
def create_embeddings(trajectories, reward, embedding_type):
    vectors = [_create_vector(trajectories[i], reward[i], embedding_type) for i in range(0, len(trajectories))]

#     # initialize the Autoencoder model with input dimension equal to the length of a trajectory
    # model = AE(input_dim=len(vectors))

    # # prepare features and labels for training
    # features = torch.FloatTensor(vectors)
    # labels = torch.FloatTensor(vectors)
    
#     # create a TensorDataset and DataLoader for training
#     train = data_utils.TensorDataset(features, labels)
#     train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)

#     # train the Autoencoder model
#     model.train(train_loader, 100)

#     # create trajectory objects with the original trajectory, reward, and trained encoder module
#     trajectories = []
#     for traj_num in range(0, len(traj_info_list)):
#         trajectories.append(Traj(traj_info_list[traj_num][0], traj_info_list[traj_num][1], model.encoder[traj_num]))

#     return trajectories
    return vectors
# -------------------------
# -------------------------




# -------------------------
# -------------------------
def find_closest_embedding(trajectories, embeddings, traj_of_interest, technique):
    # This is where we find distances between our trajectory of interest and the other trajectories in the space
    if technique == "subtraction":
        dist = []
        for t in trajectories:
            dist.append(t - traj_of_interest)
        # find smallest vector (larger than 0)
        closest_indices = sorted(range(len(dist)), key=lambda sub: dist[sub])[:K]
        # closest_trajectories = [sorted_trajectories[i] for i in closest_indices]
# -------------------------
# -------------------------        

        
