from AE import AE
import torch
import torch.utils.data as data_utils

def main():
    # Constants
    NUM_TRAJECTORIES = 200
    COMMON_GROUND_SIZE = 20
    TRAJ_LEN = 15

    model = AE(input_dim=NUM_TRAJECTORIES)
    print(model.encoder[0].weight)

    rand_list = torch.rand((TRAJ_LEN,NUM_TRAJECTORIES,), dtype=torch.float)
    labels_list = torch.rand((TRAJ_LEN,), dtype=torch.float)

    features = torch.tensor(rand_list)
    labels = torch.tensor(labels_list)

    train = data_utils.TensorDataset(features, labels)
    train_loader = data_utils.DataLoader(train, batch_size=50, shuffle=True)

# # float tensor
# # tensor.float2d float tensor - you can flatten
    
    model.train(train_loader, 100)

if __name__ == "__main__":
    main()