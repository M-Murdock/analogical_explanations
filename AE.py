"""
This code mainly follows the Geeks4Geeks Pytorch Autoencoder example.
https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/

Any modifiations are made by the AABL Lab.
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

class AE(nn.Module):

    def __init__(self, input_dim, learning_rate=1e-3):
        super().__init__()
        #TODO: It might be good to be able to calculate reduction till it hits desired latent space
        # Perhaps in a loop, etc. but for the sake of initial implementation I have not done this.
        
        self.input_dim = input_dim # Input dimensions of data

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, (input_dim//2)),
            nn.ReLU(),
            nn.Linear((input_dim//2), (input_dim//4)),
            nn.ReLU(),
            nn.Linear((input_dim//4), (input_dim//8)),
            nn.ReLU(),
            nn.Linear((input_dim//8), (input_dim//16)),
            nn.ReLU(),
            nn.Linear((input_dim//16), (input_dim//32)),
            nn.ReLU(),
            nn.Linear((input_dim//32), (input_dim//64))
        )

        self.decoder = nn.Sequential(
            nn.Linear((input_dim//64), (input_dim//32)),
            nn.ReLU(),
            nn.Linear((input_dim//32), (input_dim//16)),
            nn.ReLU(),
            nn.Linear((input_dim//16), (input_dim//8)),
            nn.ReLU(),
            nn.Linear((input_dim//8), (input_dim//4)),
            nn.ReLU(),
            nn.Linear((input_dim//4),  (input_dim//2)),
            nn.ReLU(),
            nn.Linear((input_dim//2), input_dim),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train(self, data_loader, epochs, print_epoch_loss=True):
        for epoch in range(epochs):
            loss = 0
            for batch_data, _ in data_loader:
                # Reshape mini-batch data to [N, 784] matrix
                batch_data = batch_data.view(-1, self.input_dim)
                
                self.optimizer.zero_grad() # Zero gradient for training
                
                outputs = self.forward(batch_data) # Feed forwards and get outputs
                
                train_loss = self.criterion(outputs, batch_data) # Compute loss
                train_loss.backward() # Compute backprop gradients from loss
                
                self.optimizer.step() # Update with computed gradients
                
                # Add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            train_loader_dim = 0
            for idx, data in enumerate(data_loader):
                train_loader_dim = len(data[0])
                break

            # Compute the epoch training loss
            loss = loss / train_loader_dim
            
            # Display the epoch training loss
            if print_epoch_loss:
                print("Epoch : {}/{}, Loss = {:.6f}".format(epoch + 1, epochs, loss))
    

if __name__ == "__main__":
    model = AE(input_dim=784) # Dimensions of MNIST images flattened (28 x 28 = 784)

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    model.train(train_loader, 100)
    
    
    
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
