#!/usr/bin/env python3

# Python imports.
from __future__ import print_function

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world import GridWorldMDPClass
from simple_rl.agents import QLearningAgent
from mdp import solve_mdp, run_mdp
from simple_rl.run_experiments import run_single_agent_on_mdp, run_agents_lifelong

def main():
    mdp = GridWorldMDPClass.make_grid_world_from_file("easygrid.txt")
    ql_agent = QLearningAgent(actions=mdp.get_actions()) 


    run_single_agent_on_mdp(ql_agent, mdp, episodes=1000, steps=200, verbose=False)
    # run_agents_lifelong([ql_agent], mdp)
    
    # The function `solve_mdp(ql_agent, mdp, episodes=5000, steps=1000)` is
    # likely training the Q-learning agent (`ql_agent`) on the given Markov
    # Decision Process (MDP) (`mdp`) for a specified number of episodes and
    # steps.
    # solve_mdp(ql_agent, mdp, episodes=5000, steps=200)
    run_mdp(ql_agent, mdp)
    mdp.visualize_agent(ql_agent)
    
    
    
if __name__ == "__main__":
    main()