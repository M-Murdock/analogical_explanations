import numpy as np

def run_mdp(agent, mdp):
    trajectory = [] 
    state = mdp.get_init_state()
    cumulative_reward = 0
    
    while not state.is_terminal():
        # Compute the agent's policy.
        action = agent.act(state, None, learning=False)
        
        # Terminal check.
        if state.is_terminal():
            break 
        
        # Execute in MDP.
        _, next_state = mdp.execute_agent_action(action)
        
        trajectory.append((state, action))

        # Update the agent's cumulative reward
        cumulative_reward += agent.get_value(state) 
        
        # Update pointer.
        state = next_state

    # print(trajectory)
    
    # Reset the MDP, tell the agent the episode is over.
    mdp.reset()
    agent.end_of_episode()
    return trajectory, cumulative_reward