from AE import AE
import torch
import torch.utils.data as data_utils
import numpy as np
# import regex_utils
import re

def _create_embedding(trajectory, reward, embedding_type):
    get_sa_sequence(trajectory)
    if embedding_type == "reward-trajectory": 
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        
def get_sa_sequence(trajectory):
    # use regular expression to capture the pair of numbers 
    sequence = [t[0] for t in trajectory]
    print(sequence[0])
    # print(re.match(str(sequence[0]), "([0-9]+, [0-9]+)"))
    # print(str(sequence[0]))
    match = re.search(r'^.*(\d+,.*)$', str(sequence[0]))
    print(match.groups())
    # print(match)
    # print(re.search(r'.*', str(sequence[0])))
    
def create_embeddings(trajectories, reward, embedding_type):
    return [_create_embedding(t, reward, embedding_type) for t in trajectories]
    
        
# class Traj:
#     """
#     A class representing a trajectory with associated reward and embedding.

#     Attributes:
#     traj (list): A list of state-action pairs representing the trajectory.
#     reward (float): The reward associated with the trajectory.
#     common_ground (bool): A flag indicating whether the trajectory is part of common ground.
#     embedding (torch.nn.Module): The trained encoder module of the AE model for the trajectory.
#     """

#     def __init__(self, traj, reward, embedding):
#         """
#         Initialize a new Traj object.

#         Parameters:
#         traj (list): A list of state-action pairs representing the trajectory.
#         reward (float): The reward associated with the trajectory.
#         embedding (torch.nn.Module): The trained encoder module of the AE model for the trajectory.
#         """
#         self.traj = traj
#         self.reward = reward
#         self.common_ground = False
#         self.embedding = embedding

    
# def create_embeddings(traj_info_list):
#     """
#     Create embeddings for a list of trajectories using an Autoencoder (AE) model.

#     This function takes a list of trajectory information tuples, processes the trajectories, adds environmental information,
#     and trains an AE model to generate embeddings for each trajectory.

#     Parameters:
#     traj_info_list (list): A list of trajectory information tuples. Each tuple contains:
#         - traj (list): A list of state-action pairs representing a trajectory.
#         - reward (float): The reward associated with the trajectory.
#         - environment (object): The environment in which the trajectory was generated.

#     Returns:
#     list: A list of trajectory objects. Each trajectory object contains:
#         - traj (list): The original trajectory.
#         - reward (float): The reward associated with the trajectory.
#         - embedding (torch.nn.Module): The trained encoder module of the AE model for the trajectory.
#     """
    
#     # get a list of state/actions in trajectory and add them to environmental information
#     input_list = []
#     for t in traj_info_list:
#         input_list.append(_traj_to_int(t[0]) + _environment_to_int(t[2]))

#     # make all trajectories of the same length
#     input_list = _make_length_uniform(input_list)

#     # prepare output list with a constant value for training
#     output_list = input_list#[8 for i in input_list]

#     # initialize the Autoencoder model with input dimension equal to the length of a trajectory
#     model = AE(input_dim=len(input_list))

#     # prepare features and labels for training
#     features = torch.FloatTensor(input_list)
#     labels = torch.FloatTensor(output_list)

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


# def find_closest_reward(trajectories, reference_traj, K=5, exclude=[]):
#     """
#     Find the K trajectories with the closest rewards to a given reference trajectory.

#     This function takes a list of trajectories, a reference trajectory, and an optional parameter K.
#     It sorts the trajectories by their rewards, removes the reference trajectory from the list,
#     calculates the absolute difference in rewards between the reference trajectory and each remaining trajectory,
#     and selects the K trajectories with the smallest differences.

#     Parameters:
#     trajectories (list): A list of trajectory objects. Each trajectory object contains a trajectory and its reward.
#     reference_traj (Traj): The reference trajectory for which to find the closest trajectories.
#     K (int, optional): The number of closest trajectories to find. Default is 5.

#     Returns:
#     list: A list of K trajectory objects that have the closest rewards to the reference trajectory.
#     """
#     if len(exclude) > 0:
#         # remove excluded trajectories from the list
#         for t in exclude:
#             if t in trajectories:
#                 trajectories.remove(t)

#     # create a list of trajectories that is sorted by reward
#     sorted_trajectories = sorted(trajectories, key=lambda x: x.reward)
#     if reference_traj in sorted_trajectories:
#         sorted_trajectories.remove(reference_traj)

#     # find the absolute difference in reward between reference_traj and each trajectory
#     absolute_difference = [abs(x.reward - reference_traj.reward) for x in sorted_trajectories]

#     # find K number of closest trajectories 
#     closest_indices = sorted(range(len(absolute_difference)), key=lambda sub: absolute_difference[sub])[:K]
#     closest_trajectories = [sorted_trajectories[i] for i in closest_indices]

#     return closest_trajectories


# def initialize_common_ground(trajectories, N):
#     """
#     Initialize a subset of trajectories as common ground.

#     This function selects N random trajectories from the given list and marks them as common ground.
#     The common ground is represented as a list of trajectory objects.

#     Parameters:
#     trajectories (list): A list of trajectory objects. Each trajectory object contains a trajectory and its reward.
#     N (int): The number of trajectories to be selected as common ground.

#     Returns:
#     list: A list of trajectory objects that have been selected as common ground.
#     """
#     # Choose N trajectories to add to the common ground
#     common_ground = []
#     for _ in range(N):
#         random_index = np.random.randint(0, len(trajectories))
#         common_ground.append(trajectories[random_index])
#         trajectories[random_index].common_ground = True

#     return common_ground

# def select_analogical_example(trajectories):
#     """
#     Select a random trajectory that is not part of common ground.

#     This function randomly selects a trajectory from the given list of trajectory objects.
#     It ensures that the selected trajectory is not part of common ground by checking the 'common_ground' attribute of each trajectory.

#     Parameters:
#     trajectories (list): A list of trajectory objects. Each trajectory object contains a trajectory and its reward.

#     Returns:
#     Traj: A randomly selected trajectory object that is not part of common ground.
#     """
#     random_index = np.random.randint(0, len(trajectories))
#     while trajectories[random_index].common_ground:
#         random_index = np.random.randint(0, len(trajectories))

#     return trajectories[random_index]
    
# # Utility Functions
# def _traj_to_int(traj):
#     numerical_traj = []
#     pair_sum = 0
    
#     # (state, action), (state, action)
#     for state_action in traj:
#         pair = str(state_action[0]) + str(state_action[1])
#         for letter in pair:
#             pair_sum += ord(letter)

#         numerical_traj.append(pair_sum)
#     return numerical_traj

# def _environment_to_int(environment):
#     int_environment = []
#     str_environment = str(environment)
#     for letter in str_environment:
#         int_environment.append(ord(letter))
        
#     return int_environment

# # NOTE: this definitely isn't the best way to do this
# def _make_length_uniform(input_list):
#     longest = 0
#     # find longest trajectory
#     for i in input_list:
#         if len(i) > longest:
#             longest = len(i)
            
#     # make the other trajectories the same length
#     for i in input_list:
#         if len(i) < longest:
#             i.extend([0]*(longest - len(i)))
    
#     return input_list 
