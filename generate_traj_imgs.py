#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pickle
import os

def main():
    
    file_name = "n-grams/states-n-grams-gridworld.txt"
    data = ""
    
    with open(file_name, 'rb') as f:
        # Load the data from the file
        data = pickle.load(f)

    traj_x = []
    traj_y = []
    
    for trajectory in data:
        traj_x.append([int(t.split('-')[0]) for t in trajectory[1:]])
        traj_y.append([int(t.split('-')[1]) for t in trajectory[1:]])
    print(traj_y)
    
    for i in range(0, len(data)): # loop through every trajectory 
        plt.plot(traj_x[i], traj_y[i])
        plt.xlim(0, 14)
        plt.ylim(0, 14)
        # plt.scatter(traj_x[i][0], traj_y[i][0], color='red')
        # plt.scatter(traj_x[i][-1], traj_y[i][-1], color='green')
        # plt.axis('off')
        plt.title('Trajectory '+str(i))
        img_filename = os.path.join("trajectory_imgs", 'img'+str(i)+'.png')
        plt.savefig(img_filename) 
        plt.clf()
        
if __name__ == "__main__":
    main()