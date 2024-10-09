#!/usr/bin/env python3

# Python imports.
from __future__ import print_function

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world import GridWorldMDPClass
from simple_rl.agents import QLearningAgent
from mdp import run_mdp
from simple_rl.run_experiments import run_single_agent_on_mdp
from trajectory import Traj, create_embeddings

def main():
    # make_grid_world_from_file(file_name, randomize=False, num_goals=1, name=None, goal_num=None, slip_prob=0.0):
    mdp = GridWorldMDPClass.make_grid_world_from_file("easygrid.txt")
    ql_agent = QLearningAgent(actions=mdp.get_actions()) 
    
    # print(mdp.get_goal_locs())
    
    # train policy
    run_single_agent_on_mdp(ql_agent, mdp, episodes=1000, steps=200, verbose=False)
    
    # run agent according to policy
    traj, reward = run_mdp(ql_agent, mdp)
    # print(traj)

    traj_embedding_info = [(reward, traj, mdp)]
    # print(trajectory.embedding)
    create_embeddings(traj_embedding_info)
    # visualize policy
    mdp.visualize_agent(ql_agent)
    
    
    
if __name__ == "__main__":
    main()