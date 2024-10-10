#!/usr/bin/env python3

# Python imports.
from __future__ import print_function

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world import GridWorldMDPClass
from simple_rl.agents import QLearningAgent
from mdp import run_mdp
from simple_rl.run_experiments import run_single_agent_on_mdp
from trajectory import Traj, create_embeddings, find_closest_reward

def main():
    # make_grid_world_from_file("easygrid.txt", randomize=True, num_goals=1, name=None, goal_num=None, slip_prob=0.0)
    mdp = GridWorldMDPClass.make_grid_world_from_file("easygrid.txt")
    ql_agent = QLearningAgent(actions=mdp.get_actions()) 
    # train policy
    run_single_agent_on_mdp(ql_agent, mdp, episodes=1000, steps=200, verbose=False)
    # run agent according to policy
    traj, reward = run_mdp(ql_agent, mdp)
    
    # -----------------------
    # create second trajectory
    mdp2 = GridWorldMDPClass.make_grid_world_from_file("easygrid.txt", randomize=True, num_goals=1, name=None, goal_num=None, slip_prob=0.1)
    ql_agent2 = QLearningAgent(actions=mdp.get_actions()) 
    # train policy
    run_single_agent_on_mdp(ql_agent2, mdp2, episodes=1000, steps=200, verbose=False)
    # run agent according to policy
    traj2, reward2 = run_mdp(ql_agent2, mdp2)
    # -------------------------
    # -----------------------
    # create third trajectory
    mdp3 = GridWorldMDPClass.make_grid_world_from_file("easygrid.txt", randomize=True, num_goals=1, name=None, goal_num=None, slip_prob=0.5)
    ql_agent3 = QLearningAgent(actions=mdp.get_actions()) 
    # train policy
    run_single_agent_on_mdp(ql_agent3, mdp3, episodes=1000, steps=200, verbose=False)
    # run agent according to policy
    traj3, reward3 = run_mdp(ql_agent3, mdp3)
    # -------------------------
    
    # traj_embedding_info = [(traj, reward, mdp)]
    traj_embedding_info = [(traj, reward, mdp), (traj2, reward2, mdp2), (traj3, reward3, mdp3)]
    trajectories = create_embeddings(traj_embedding_info)
    find_closest_reward(trajectories, reference_traj=trajectories[0], K=1)
    
    # visualize policy
    mdp.visualize_agent(ql_agent)
    
    
    
if __name__ == "__main__":
    main()