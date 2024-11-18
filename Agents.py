import pickle 

class Agents:
    def __init__(self, traj_filename="optimal_agents/optimal_trajs", rewards_filename="optimal_agents/optimal_rewards"):
        self.traj_filename = traj_filename
        self.rewards_filename = rewards_filename
        
        
    
    def train_agents(self, mdps, q_learning_agents, episodes=10, steps=500):
        self.optimal_trajectories = []
        self.optimal_rewards = []
        for i in range(0, len(mdps)):
            traj, r = self._train_agent(q_learning_agents[i], mdps[i], episodes, steps)
            self.optimal_trajectories.append(traj)
            self.optimal_rewards.append(r)
        
        # save trajectories/rewards
        with open(self.traj_filename, "wb") as fp:
            pickle.dump(self.optimal_trajectories, fp)
        with open(self.rewards_filename, "wb") as fp:
            pickle.dump(self.optimal_rewards, fp)
            
        return self.optimal_trajectories, self.optimal_rewards
            
            
    def _train_agent(self, agent, mdp, episodes, steps):
        # ----------------------------------------------------------------
        # Train the agent
        for _ in range(1, episodes + 1):
            state = mdp.get_init_state()
            reward = 0
            
            for _ in range(1, steps + 1):
                action = agent.act(state, reward)
                
                if state.is_terminal():
                    continue
                
                # Execute in MDP.
                reward, next_state = mdp.execute_agent_action(action)
                
                if next_state.is_terminal():
                    # Reset the MDP.
                    next_state = mdp.get_init_state()
                    mdp.reset()
                    
                # Update pointer.
                state = next_state
                
            mdp.reset()
            agent.end_of_episode()

        # ----------------------------------------------------------------
        # Get the optimal trajectory and rewards
        self.trajectory = [] 
        self.rewards = []
        state = mdp.get_init_state()

        while not state.is_terminal():
            # Compute the agent's policy.
            action = agent.act(state, None, learning=False)

            # Terminal check.
            if state.is_terminal():
                break

            # Execute in MDP
            _, next_state = mdp.execute_agent_action(action)

            self.trajectory.append((state, action))
            self.rewards.append(agent.get_value(state))

            # Update pointer.
            state = next_state

        # Reset the MDP, tell the agent the episode is over.
        mdp.reset()
        agent.end_of_episode()
        
        return self.trajectory, self.rewards
    
    def load_agents(self):
        with open(self.traj_filename, "rb") as fp:   
            self.optimal_trajectories = pickle.load(fp)
        with open(self.rewards_filename, "rb") as fp:
            self.optimal_rewards = pickle.load(fp)
            
        return self.optimal_trajectories, self.optimal_rewards 