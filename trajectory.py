from AE import AE
import torch
import torch.utils.data as data_utils
import numpy as np

class Traj:

    def __init__(self, traj, reward, common_ground=False):
        self.traj = traj
        self.reward = reward
        self.common_ground = common_ground
        self.embedding = self.calc_embedding()
        
    def calc_embedding(self):
        
        return self.traj
    
def create_embeddings(traj_info_list):
    # traj_info_list = [traj_info1, traj_info2, ...]
    # traj_info = (traj, reward, environment)

    # input = state-action sequence  
    # output = environment??
 

    # get a list of state/actions in trajectory
    # [[traj1],[traj2], ...]
    input_list = []
    for t in traj_info_list:
        input_list.append(traj_to_int(t[0]))
        
    # make all trajectories of the same length
    input_list = _make_length_uniform(input_list)

    # print("INPUT: ", input_list)

    output_list = [8 for i in input_list]
    # print("OUTPUT: ", output_list)

    model = AE(input_dim=len(input_list))
    
    # print("FEATURES: ", input_list)
    features = torch.FloatTensor(input_list)
    labels = torch.FloatTensor(output_list)

    train = data_utils.TensorDataset(features, labels)
    train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)

    
    model.train(train_loader, 100)
    
    trajectories = []
    for traj_num in range(0, len(traj_info_list)):
        # trajectories.append(Traj(traj_info_list[traj_num][0], traj_info_list[traj_num][1], model.encoder[traj_num].weight))
        trajectories.append(Traj(traj_info_list[traj_num][0], traj_info_list[traj_num][1], model.encoder[traj_num]))
        

    return trajectories

def find_closest_reward(trajectories, reference_traj, K=5):
    # create a list of trajectories that is sorted by reward
    sorted_trajectories = sorted(trajectories, key=lambda x: x.reward)
    sorted_trajectories.remove(reference_traj)

    # find the absolute difference in reward between reference_traj and each trajectory
    absolute_difference = [abs(x.reward - reference_traj.reward) for x in sorted_trajectories]
    print(absolute_difference)
    
    # find K number of closest trajectories 
    closest_indices = sorted(range(len(absolute_difference)), key=lambda sub: absolute_difference[sub])[:K]
    closest_trajectories = [sorted_trajectories[i] for i in closest_indices]
    
    return closest_trajectories

    
def traj_to_int(traj):
    numerical_traj = []
    pair_sum = 0
    
    # (state, action), (state, action)
    for state_action in traj:
        pair = str(state_action[0]) + str(state_action[1])
        for letter in pair:
            pair_sum += ord(letter)

        numerical_traj.append(pair_sum)
    return numerical_traj

# NOTE: this definitely isn't the best way to do this
def _make_length_uniform(input_list):
    longest = 0
    # find longest trajectory
    for i in input_list:
        if len(i) > longest:
            longest = len(i)
            
    # make the other trajectories the same length
    for i in input_list:
        if len(i) < longest:
            i.extend([0]*(longest - len(i)))
    
    return input_list 
