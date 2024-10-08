#!/usr/bin/env python3

# Python imports.
from __future__ import print_function

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world import GridWorldMDPClass
from simple_rl.agents import QLearningAgent
from mdp import solve_mdp, run_mdp

def main():
    mdp = GridWorldMDPClass.make_grid_world_from_file("octogrid.txt")
    ql_agent = QLearningAgent(actions=mdp.get_actions()) 

    mdp, ql_agent = solve_mdp(ql_agent, mdp, episodes=50, steps=1000)
    run_mdp(ql_agent, mdp, steps=1000)
    
if __name__ == "__main__":
    main()