#!/usr/bin/env python3

# Python imports.
from __future__ import print_function

# Other imports.
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world import GridWorldMDPClass
from simple_rl.agents import QLearningAgent
# from mdp import run_mdp, generate_policies_for_agents
from simple_rl.run_experiments import run_single_agent_on_mdp
from trajectory import create_embeddings
from ngrams import create_ngrams
# , find_closest_reward, initialize_common_ground, select_analogical_example

def main():
    # -------------------------
    # -----------------------
    # STEP 1: Create Trajectories (via optimal policies)
    optimal_trajectories, rewards = create_optimal_trajectories(map_name="easygrid.txt", num_agents=3, episodes=1000, steps=200, slip_prob=0.1)
    # print(optimal_trajectories)
    # print(rewards)
    print("DONE")
    # -------------------------
    # -----------------------
    
    # -------------------------
    # -----------------------
    # STEP 2: Create embeddings of the trajectories (using whatever technique)
    # -------------------------
    # -----------------------
    embedded_trajectories = create_ngrams(optimal_trajectories, rewards, "state-action")
    # embedded_trajectories = create_embeddings(optimal_trajectories, rewards, "state-action")
    print(embedded_trajectories)
    # print(embedded_trajectories)

    # -------------------------
    # -----------------------
    # TODO: We need to establish common ground first
    # STEP 3: Use parallelogram method
    
    # -------------------------
    # -----------------------


    
    # # traj_embedding_info = [(traj, reward, mdp)]
    # traj_embedding_info = [(traj, reward, mdp), (traj2, reward2, mdp2), (traj3, reward3, mdp3)]
    # trajectories = create_embeddings(traj_embedding_info)
    
    # common_ground = initialize_common_ground(trajectories, 1)
    # print("Common ground: ", common_ground)
    
    # # remove values in common ground from trajectories
    # unknown_ground = [i for i in trajectories if i not in common_ground]
    
    # closest_reward = find_closest_reward(trajectories, reference_traj=trajectories[0], K=1, exclude=unknown_ground)
    # print("Closest reward: ", closest_reward)
    
    # analogical_example = select_analogical_example(trajectories)
    # print("Analogical example: ", analogical_example)
    # # visualize policy
    # mdp.visualize_agent(ql_agent)
    

def create_optimal_trajectories(map_name="easygrid.txt", num_agents=3, episodes=1000, steps=200, slip_prob=0.1, traj_len=5):
    mdps = [GridWorldMDPClass.make_grid_world_from_file(map_name, randomize=True, num_goals=1, name=None, goal_num=None, slip_prob=slip_prob) for _ in range(0, num_agents)]
    actions = mdps[0].get_actions() 
    q_learning_agents = [QLearningAgent(actions=actions) for _ in range(0, num_agents)]
    # train policy
    for i in range(0, len(q_learning_agents)):
        run_single_agent_on_mdp(q_learning_agents[i], mdps[i], episodes=episodes, steps=steps, verbose=False)

    # get the trajectories and their associated rewards
    optimal_trajectories = []
    rewards = []
    for i in range(0, num_agents):
        traj, r = _run_mdp(q_learning_agents[i], mdps[i])
        # if not (len(traj ) == traj_len):
        # create_optimal_trajectories(map_name=map_name, num_agents=num_agents, episodes=episodes, steps=steps, slip_prob=slip_prob, traj_len=traj_len)
        optimal_trajectories.append(traj)
        rewards.append(r)
    
    # return a list of optimal trajectories
    return optimal_trajectories, rewards


def _run_mdp(agent, mdp):
    trajectory = [] 
    rewards = []
    state = mdp.get_init_state()

    while not state.is_terminal():
        # Compute the agent's policy.
        action = agent.act(state, None, learning=False)

        # Terminal check.
        if state.is_terminal():
            break

        # Execute in MDP.
        _, next_state = mdp.execute_agent_action(action)

        trajectory.append((state, action))
        rewards.append(agent.get_value(state) )

        # Update pointer.
        state = next_state

    # Reset the MDP, tell the agent the episode is over.
    mdp.reset()
    agent.end_of_episode()
    return trajectory, rewards


if __name__ == "__main__":
    main()