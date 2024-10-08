import gymnasium as gym
import numpy as np
import random
from AE import AE
import torch
import torchvision
import tensorflow as tf
import torch.utils.data as data_utils

def main():
    # Constants
    NUM_TRAJECTORIES = 200
    COMMON_GROUND_SIZE = 20
    TRAJ_LEN = 15

    model = AE(input_dim=50)


    rand_list = torch.rand((10,10,), dtype=torch.float)
    labels_list = [random.randint(3,9) for x in range(0,10)]

    features = torch.tensor(rand_list)
    labels = torch.tensor(labels_list)

    train = data_utils.TensorDataset(features, labels)
    train_loader = data_utils.DataLoader(train, batch_size=5, shuffle=True)

    # inputs  = torch.tensor(inputs)
    # targets = torch.IntTensor(targets)
        
    # dataset = TensorDataset(inputs, targets)
    # data_loader = DataLoader(dataset, batch_size, shuffle=True)
# ------------
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # train_loader = torch.utils.data.DataLoader(
    #     test_trajectories, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
    # )

#     print("Train loader:")
#     print(train_loader)
# # float tensor
# # tensor.float2d float tensor - you can flatten
    
    model.train(train_loader, 100)

if __name__ == "__main__":
    main()