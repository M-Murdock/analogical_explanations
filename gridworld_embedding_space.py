from simple_rl.tasks.grid_world import GridWorldMDPClass
from simple_rl.tasks import FourRoomMDP
from simple_rl.agents import QLearningAgent
import pickle
from AE import AE
from scipy import spatial
import os
from Agents import Agents

class GridWorldEmbeddingSpace:
    def __init__(self, load_agents=True, TASK="gridworld", N_GRAM_TYPE="state-action", MAP_DIRECTORY="maps/"):
        
        self.N_GRAM_TYPE = N_GRAM_TYPE
        self.TASK = TASK
        self.MAP_DIRECTORY = MAP_DIRECTORY
        
        self.n_gram_file = "n-grams/" + N_GRAM_TYPE + "-n-grams-" + TASK + ".txt"
        self.model_save_file = "keras/" + N_GRAM_TYPE + "-model-" + TASK + ".keras"
        self.traj_filename="optimal_agents/optimal_trajs-" + self.TASK + ""
        self.rewards_filename="optimal_agents/optimal_rewards-" + self.TASK + ""

        if load_agents:
            self.load_agents()
        else:
            self.train_agents() 
        
        self._traj_to_sentences()
        self._states_to_coord()

        
    def new_model(self):
        self.ae = AE(self.n_gram_file, self.model_save_file)
        self.ae.process_data()
        self.ae.train()
        self.vectors = self.ae.doc_vectors
        
    def load_model(self):
        self.ae = AE(self.n_gram_file, self.model_save_file)
        self.ae.load()
        self.vectors = self.ae.doc_vectors
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------   
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
        # contents = ""
        sentences = []
        
        for traj_num in range(0, len(self.state_sequences)): # go through each trajectory
            
            contents = ""
            chunk_size = 5
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
                if self.N_GRAM_TYPE == "state-action-chunks":
                    # save state-action pairs to the file
                    contents += self.state_sequences[traj_num][i] + "" + self.actions_sequences[traj_num][i]
                    if i % chunk_size == 0:
                        contents += " "
                # ----------------------------------------------------------------     
            # contents += ";"
            sentences.append(contents.split(" "))

        # save the new representation to a file
        with open(self.n_gram_file, "wb") as fp:
            pickle.dump(sentences, fp)
            


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

        
        # find the point closest to D_estimate
        tree = spatial.KDTree(self.vectors)
        dist, indices = tree.query(D_estimate, k=4) 
        
        index = 0
        # make sure that the vector closest to D_estimate is not the original vectors A, B, or C
        for i in range(0,3):
            index = indices[i]
            if not (self.vectors[index].all() == A.all() or self.vectors[index].all() == B.all() or self.vectors[index].all() == C.all()):
                break
            
        return index
    
    
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Save/Load trained agents
    # ----------------------------------------------------------------   
    def train_agents(self, episodes=100, steps=200, slip_prob=0.1):
        # create MDPs for each possible trajectory (i.e. from every possible start position)
        if self.TASK == "gridworld":
            self.mdps = [GridWorldMDPClass.make_grid_world_from_file(os.path.join(self.MAP_DIRECTORY, path), num_goals=1, name=None, slip_prob=slip_prob) for path in os.listdir(self.MAP_DIRECTORY)]
        elif self.TASK == "four_room":
            # self.mdps = [FourRoomMDP(9, 9, init_loc=(1,i), goal_locs=[(0,0)]) for i in range(0,9)]
            self.mdps = []
            for i in range(0,8):
                for j in range(0,8):
                    self.mdps.append(FourRoomMDP(9, 9, init_loc=(i,j), goal_locs=[(0,0)]))
            # self.mdps = [FourRoomMDP(9, 9, init_loc=(i,j), goal_locs=[(0,0)]) for i in range(0,8) j in range(0,8)]
            
        # create Q-learning agents for each trajectory
        self.q_learning_agents = [QLearningAgent(actions=self.mdps[i].get_actions() ) for i in range(0, len(self.mdps))]
        
        # train the agents
        self.agents = Agents(traj_filename=self.traj_filename, rewards_filename=self.rewards_filename)
        self.optimal_trajectories, self.optimal_rewards = self.agents.train_agents(self.mdps, self.q_learning_agents, episodes, steps)

        self.NUM_TRAJECTORIES = len(self.optimal_trajectories)
    
    def load_agents(self):
        self.agents = Agents(traj_filename=self.traj_filename, rewards_filename=self.rewards_filename)
        self.optimal_trajectories, self.optimal_rewards = self.agents.load_agents()
        self.NUM_TRAJECTORIES = len(self.optimal_trajectories)

        