from simple_rl.tasks.grid_world import GridWorldMDPClass
from simple_rl.agents import QLearningAgent
from simple_rl.run_experiments import run_single_agent_on_mdp
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from matplotlib.widgets import Slider, Button, RadioButtons


# -------------------------
# -------------------------
# # NOTE: this function is specific to this gridworld environment
# def _get_sa_sequences(trajectories):
#     actions = []
#     states = []
    
#     for traj in trajectories:
#         # get the list of states for the current trajectory
#         s_sequence = [t[0] for t in traj]

        
#         temp_state = []
#         # get the string for the current state
#         for s in s_sequence:
#             temp_state.append("" + str(s[0]) + "-" + str(s[1]))
            
#         a_sequence = [t[1] for t in traj]   
#         actions.append(a_sequence)
#         states.append(temp_state)   

    
#     return states, actions
# -------------------------
# -------------------------

# def _get_sa_sequence(trajectory):
#     actions = []
#     states = []
    
#     # get the list of states for the current trajectory
#     states = [(t[0][0],t[0][1]) for t in trajectory]    
#     actions = [t[1] for t in trajectory]    
    
#     return states, actions
# -------------------------
# # -------------------------
# # get the lists of states and actions
# def save_traj_to_file(trajectories, rewards, file_name="state-action.txt", ngram_type="state-action"):
#     contents = ""
#     s_seq, a_seq = _get_sa_sequences(trajectories)
    
#     for traj_num in range(0, len(s_seq)): # go through each trajectory
#         for i in range(0, len(a_seq[traj_num])):
#             # ----------------------------------------------------------------
#             if ngram_type == "state-action":
#                 # save state-action pairs to the file
#                 contents += " " + s_seq[traj_num][i] + "" + a_seq[traj_num][i]
#             # ----------------------------------------------------------------        
#             if ngram_type == "action-reward":
#                 # save action-reward pairs to the file
#                 contents += " " + a_seq[traj_num][i] + "" + str(rewards[traj_num][i])   
#             # ----------------------------------------------------------------
#             if ngram_type == "state-reward":
#                 # save state to the file
#                 contents += " " + s_seq[traj_num][i] + "" + str(rewards[traj_num][i])   
#             # ----------------------------------------------------------------
#             if ngram_type == "states":
#                 # save state sequence to the file
#                 contents += " " + s_seq[traj_num][i] + ""
#             # ----------------------------------------------------------------
#             if ngram_type == "actions":
#                 # save action sequence to the file
#                 contents += " " + a_seq[traj_num][i] + ""
#             # ----------------------------------------------------------------
#             if ngram_type == "state-action-reward":
#                 # save state-action-reward to the file
#                 contents += " " + s_seq[traj_num][i] + "" + a_seq[traj_num][i] + str(rewards[traj_num][i]) 
#             # ----------------------------------------------------------------  
#         contents += "."
    
            
#     f = open(file_name, "w")
#     f.write(contents)
#     f.close()

# def create_optimal_trajectories(map_name="easygrid.txt", num_agents=3, episodes=1000, steps=200, slip_prob=0.1, traj_len=5):
#     mdps = [GridWorldMDPClass.make_grid_world_from_file(map_name, randomize=True, num_goals=1, name=None, goal_num=None, slip_prob=slip_prob) for _ in range(0, num_agents)]
#     # See: https://github.com/david-abel/simple_rl/blob/master/examples/viz_example.py
#     actions = mdps[0].get_actions() 
#     q_learning_agents = [QLearningAgent(actions=actions) for _ in range(0, num_agents)]
#     # train policy
#     for i in range(0, len(q_learning_agents)):
#         run_single_agent_on_mdp(q_learning_agents[i], mdps[i], episodes=episodes, steps=steps, verbose=False)

#     # get the trajectories and their associated rewards
#     optimal_trajectories = []
#     rewards = []
#     for i in range(0, num_agents):
#         traj, r = _run_mdp(q_learning_agents[i], mdps[i])
#         # if not (len(traj ) == traj_len):
#         # create_optimal_trajectories(map_name=map_name, num_agents=num_agents, episodes=episodes, steps=steps, slip_prob=slip_prob, traj_len=traj_len)
#         optimal_trajectories.append(traj)
#         rewards.append(r)
    
#     # return a list of optimal trajectories
#     return optimal_trajectories, rewards, q_learning_agents, mdps


# def _run_mdp(agent, mdp):
#     trajectory = [] 
#     rewards = []
#     state = mdp.get_init_state()

#     while not state.is_terminal():
#         # Compute the agent's policy.
#         action = agent.act(state, None, learning=False)

#         # Terminal check.
#         if state.is_terminal():
#             break

#         # Execute in MDP.
#         _, next_state = mdp.execute_agent_action(action)

#         trajectory.append((state, action))
#         rewards.append(agent.get_value(state) )

#         # Update pointer.
#         state = next_state

#     # Reset the MDP, tell the agent the episode is over.
#     mdp.reset()
#     agent.end_of_episode()
#     return trajectory, rewards

# def visualize_trajectory(trajectories, labels=["A", "B", "C", "D"]):
#     fig, ax = plt.subplots()
    
#     all_of_x = []
#     all_of_y = []
    
#     for trajectory in trajectories: 
#         states, _ = _get_sa_sequence(trajectory)
    
#         x = [s[0] for s in states]
#         y = [s[1] for s in states]
#         all_of_x.append(x)
#         all_of_y.append(y)
    
#     for i in range(0, len(trajectories)):
#         ax.plot(all_of_x[i], all_of_y[i], label=labels[i])
    

#     ax.set(xlabel='X', ylabel='Y',
#         title='Trajectory graph')
#     ax.grid()
#     plt.show()

