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
        input_list.append(traj_to_int(t[1]))

    print("INPUT: ", input_list)

    output_list = [8 for i in input_list]
    print("OUTPUT: ", output_list)

    model = AE(input_dim=len(input_list))
 
    features = torch.FloatTensor(input_list)
    labels = torch.FloatTensor(output_list)
 

    train = data_utils.TensorDataset(features, labels)
    train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)

    
    model.train(train_loader, 100)
    
    trajectories = []
    for traj_num in range(0, len(traj_info_list)):
        # trajectories.append(Traj(traj_info_list[traj_num][1], traj_info_list[traj_num][0], model.encoder[0].weight[traj_num]))
        trajectories.append(Traj(traj_info_list[traj_num][0], traj_info_list[traj_num][1], model.encoder[traj_num].weight))
        

    return trajectories
    
def traj_to_int(traj):
    numerical_traj = []
    pair_sum = 0
    
    # (state, action), (state, action)]
    for state_action in traj:
        pair = str(state_action[0]) + str(state_action[1])
        for letter in pair:
            pair_sum += ord(letter)

        numerical_traj.append(pair_sum)
    return numerical_traj
