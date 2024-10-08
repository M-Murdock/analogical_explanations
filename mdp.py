import numpy as np
def solve_mdp(agent, mdp, episodes, steps, reset_at_terminal=False, resample_at_terminal=False):

    for episode in range(1, episodes + 1):
        
        # Compute initial state/reward.
        state = mdp.get_init_state()
        reward = 0
        
        for step in range(1, steps + 1):
            # Compute the agent's policy.
            action = agent.act(state, reward)
            
            # Terminal check.
            if state.is_terminal():
                break 
            
            # Execute in MDP.
            reward, next_state = mdp.execute_agent_action(action)
            
            # Update agent's state/action value.
            agent.update(state, action, reward, next_state)
            
            
            if next_state.is_terminal():
                break
            
            if reset_at_terminal:
                # Reset the MDP.
                next_state = mdp.get_init_state()
                mdp.reset()
            elif resample_at_terminal and step < steps:
                mdp.reset()

            # Update pointer.
            state = next_state
            
        # Reset the MDP, tell the agent the episode is over.
        mdp.reset()
        agent.end_of_episode()

        return mdp, agent
        

def run_mdp(agent, mdp, steps, reset_at_terminal=False, resample_at_terminal=False):
    trajectory = [] 
    state = mdp.get_init_state()
    reward = 0
    
    for step in range(1, steps + 1):
        # mdp.visualize_learning(agent)
        # mdp.visualize_agent(agent)
        # Compute the agent's policy.
        action = agent.act(state, reward, learning=False)
        
        # Terminal check.
        if state.is_terminal():
            break 
        
        # Execute in MDP.
        reward, next_state = mdp.execute_agent_action(action)
        
        trajectory.append((state, action))
        
        if next_state.is_terminal():
            break
        
        if reset_at_terminal:
            # Reset the MDP.
            next_state = mdp.get_init_state()
            mdp.reset()
        elif resample_at_terminal and step < steps:
            mdp.reset()

        # Update pointer.
        state = next_state

    print(trajectory)
    
    # Reset the MDP, tell the agent the episode is over.
    mdp.reset()
    agent.end_of_episode()
    return trajectory