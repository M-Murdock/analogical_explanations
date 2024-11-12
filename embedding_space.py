from simple_rl.tasks.grid_world import GridWorldMDPClass
# from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.agents import QLearningAgent
from simple_rl.run_experiments import run_single_agent_on_mdp
import pickle
from behavior_model import BehaviorModel
from scipy import spatial
import os

class EmbeddingSpace:
    def __init__(self, N_GRAM_TYPE="state-action", MAP_DIRECTORY="maps2/"):
        self.NUM_TRAJECTORIES = 0
        self.N_GRAM_TYPE = N_GRAM_TYPE
        self.n_gram_file = "n-grams/" + N_GRAM_TYPE + "-n-grams.txt"
        self.MAP_DIRECTORY = MAP_DIRECTORY
        self.model_save_file = "keras/" + N_GRAM_TYPE + "-model.keras"

        self._generate_optimal_trajectories()
        self._traj_to_sentences()
        self._states_to_coord()
        
        with open(self.n_gram_file, "rb") as fp:   
            self.training_data = pickle.load(fp)
            
        
    def new_model(self):
        self.behavior_model = BehaviorModel(self.n_gram_file, self.model_save_file)
        self.behavior_model.train()
        self.vectors = self.behavior_model.doc_vectors
        
    def load_model(self):
        self.behavior_model = BehaviorModel(self.n_gram_file, self.model_save_file)
        self.behavior_model.load()
        self.vectors = self.behavior_model.doc_vectors
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------- 
    
    # Generate trajectories --> convert to sentences --> train model
    
    # Generate trajectories    
    def _generate_optimal_trajectories(self, episodes=100, steps=200, slip_prob=0.1):
        # by default, movements are deterministic and reward is 1 for reaching goal
        
        # create MDPs for each possible trajectory (i.e. from every possible start position)
        self.mdps = []

        # for path in os.listdir(self.MAP_DIRECTORY): # look through everything in the directory of maps
            
        #     if os.path.isfile(os.path.join(self.MAP_DIRECTORY, path)): # check if current path is a file
        #         filename = os.path.join(self.MAP_DIRECTORY, path)
        #         print(filename)
        #         self.mdps.append(GridWorldMDPClass.make_grid_world_from_file(filename, num_goals=1, name=None, slip_prob=slip_prob))

        # self.NUM_TRAJECTORIES = len(self.mdps)
        # print("NUM:", self.NUM_TRAJECTORIES)
        self.mdps = [GridWorldMDPClass.make_grid_world_from_file(os.path.join(self.MAP_DIRECTORY, path), num_goals=1, name=None, slip_prob=slip_prob) for path in os.listdir(self.MAP_DIRECTORY)]
        # self.mdps = [GridWorldMDPClass.make_grid_world_from_file("maps/easygrid.txt", num_goals=1, name=None, slip_prob=slip_prob) for _ in range(0,30)]
        self.NUM_TRAJECTORIES = len(self.mdps)
        actions = self.mdps[0].get_actions() 
        # create Q-learning agents for each trajectory
        self.q_learning_agents = [QLearningAgent(actions=actions) for _ in range(0, self.NUM_TRAJECTORIES)]
        

        # train policy
        for i in range(0, len(self.q_learning_agents)):
            run_single_agent_on_mdp(self.q_learning_agents[i], self.mdps[i], episodes, steps, verbose=False)
        
        # get the trajectories and their associated rewards
        self.optimal_trajectories = []
        self.rewards = []
        for i in range(0, self.NUM_TRAJECTORIES):
            traj, r = self._run_mdp(self.q_learning_agents[i], self.mdps[i])
            self.optimal_trajectories.append(traj)
            self.rewards.append(r)

    # Convert to sentences
    def _traj_to_sentences(self):
        self.state_sequences = [] 
        self.actions_sequences = []
        
        for traj in self.optimal_trajectories:
            # get the list of states for the current trajectory
            s_sequence = [t[0] for t in traj]
            
            temp_state = []
            # get the string for the current state
            for s in s_sequence:
                temp_state.append("" + str(s[0]) + "-" + str(s[1]))
                
            a_sequence = [t[1] for t in traj]   
            self.actions_sequences.append(a_sequence)
            self.state_sequences.append(temp_state)   
        contents = ""
        
        for traj_num in range(0, len(self.state_sequences)): # go through each trajectory
            for i in range(0, len(self.actions_sequences[traj_num])):
                # ----------------------------------------------------------------
                if self.N_GRAM_TYPE == "state-action":
                    # save state-action pairs to the file
                    contents += " " + self.state_sequences[traj_num][i] + "" + self.actions_sequences[traj_num][i]
                # ----------------------------------------------------------------        
                if self.N_GRAM_TYPE == "action-reward":
                    # save action-reward pairs to the file
                    contents += " " + self.actions_sequences[traj_num][i] + "" + str(self.rewards[traj_num][i])   
                # ----------------------------------------------------------------
                if self.N_GRAM_TYPE == "state-reward":
                    # save state to the file
                    contents += " " + self.state_sequences[traj_num][i] + "" + str(self.rewards[traj_num][i])   
                # ----------------------------------------------------------------
                if self.N_GRAM_TYPE == "states":
                    # save state sequence to the file
                    contents += " " + self.state_sequences[traj_num][i] + ""
                # ----------------------------------------------------------------
                if self.N_GRAM_TYPE == "actions":
                    # save action sequence to the file
                    contents += " " + self.actions_sequences[traj_num][i] + ""
                # ----------------------------------------------------------------
                if self.N_GRAM_TYPE == "state-action-reward":
                    # save state-action-reward to the file
                    contents += " " + self.state_sequences[traj_num][i] + "" + self.state_sequences[traj_num][i] + str(self.rewards[traj_num][i]) 
                # ----------------------------------------------------------------  
            contents += ";"

        # save the new representation to a file
        with open(self.n_gram_file, "wb") as fp:
            pickle.dump(contents, fp)

    
    
    def _run_mdp(self, agent, mdp):
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
            

    # converts state representation into xy coordinates
    def _states_to_coord(self):
        self.state_coords = []
        for traj in self.state_sequences:
            traj_temp = []
            for s in traj:
                split_temp = s.split("-")
                traj_temp.append((int(split_temp[0]), int(split_temp[1])))
            self.state_coords.append(traj_temp)

    def infer_D(self, ABC_indices=[]):
        A = self.vectors[ABC_indices[0]]
        B = self.vectors[ABC_indices[1]]
        C = self.vectors[ABC_indices[2]]
        
        diff_vector = A-B 
        D_estimate = C + diff_vector

        
        # find the vector closest to D_estimate
        tree = spatial.KDTree(self.vectors)
        dist, indices = tree.query(D_estimate, k=4) 
        
        index = 0
        # make sure that the vector closest to D_estimate is not the original vectors A, B, or C
        for i in range(0,3):
            if not (self.vectors[indices[i]].all() == A.all() or self.vectors[indices[i]].all() == B.all() or self.vectors[indices[i]].all() == C.all()):
                break
            index = indices[i]
            
        return index
        