from simple_rl.tasks.grid_world import GridWorldMDPClass
from simple_rl.agents import QLearningAgent
from simple_rl.run_experiments import run_single_agent_on_mdp
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from behavior_model import BehaviorModel
import tensorflow as tf
from tensorflow import keras

class EmbeddingSpace:
    def __init__(self, NUM_TRAJECTORIES=50, N_GRAM_TYPE="state-action", FILE_NAME="state-action.txt", MAP_NAME="maps/easygrid.txt"):
        self.NUM_TRAJECTORIES = NUM_TRAJECTORIES
        self.N_GRAM_TYPE = N_GRAM_TYPE
        self.FILE_NAME = FILE_NAME
        self.MAP_NAME = MAP_NAME
        
        # self.new_model()
        # self.load_model()
        self._generate_optimal_trajectories()
        self._traj_to_sentences()
        self._states_to_coord()
        
        with open(self.FILE_NAME, "rb") as fp:   
            self.training_data = pickle.load(fp)
            
        self.behavior_model = BehaviorModel(FILE_NAME)
        
    
        
        
    def new_model(self):
        self.behavior_model.train()
        self.behavior_model.model.save("traj_model.keras")
        print("New model created")
    def load_model(self):
        self.behavior_model.model.load("traj_model.keras")
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------- 
    
    # Generate trajectories --> convert to sentences --> train model
    
    # Generate trajectories    
    def _generate_optimal_trajectories(self, episodes=1000, steps=200, slip_prob=0.1):
        self.mdps = [GridWorldMDPClass.make_grid_world_from_file(self.MAP_NAME, randomize=True, num_goals=1, name=None, goal_num=None, slip_prob=slip_prob) for _ in range(0, self.NUM_TRAJECTORIES)]
        # See: https://github.com/david-abel/simple_rl/blob/master/examples/viz_example.py
        actions = self.mdps[0].get_actions() 
        self.q_learning_agents = [QLearningAgent(actions=actions) for _ in range(0, self.NUM_TRAJECTORIES)]
        # train policy
        for i in range(0, len(self.q_learning_agents)):
            run_single_agent_on_mdp(self.q_learning_agents[i], self.mdps[i], episodes=episodes, steps=steps, verbose=False)

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
        with open(self.FILE_NAME, "wb") as fp:
            pickle.dump(contents, fp)
        
    # # ----------------------------------------------------------------
    # # ----------------------------------------------------------------
    # # Create a new model and save it for later retrieval
    # # ----------------------------------------------------------------
    # def new_model(self, file_name="model.bin"):
    #     # create optimal trajectories + rewards, then save them
    #     self._create_optimal_trajectories(episodes=1000, steps=200, slip_prob=0.1)
    #     self._save_optimal_trajectories()
        
    #     self._get_sa_sequences_from_traj()
        
    #     # convert trajectories to tokenized form
    #     sentences = self._traj_to_sentences()
    #     sentence_words = self._tokenize(sentences)
        
    #     # store all the sentences as TaggedDocument objects
    #     self.trajectory_objs = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence_words)]

    #     self._states_to_coord()
    #     # train the model on our sentences
    #     self.model = Doc2Vec(vector_size=100, min_count=3, epochs=20)
    #     self.model.build_vocab(self.trajectory_objs)
    #     self.model.train(self.trajectory_objs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        
    #     # save the model
    #     self.model.save(file_name)
        
        
    # # ----------------------------------------------------------------
    # # ----------------------------------------------------------------
    # # Load an existing model
    # # ----------------------------------------------------------------
    # def load_model(self, file_name="model.bin"):
    #     # load the model
    #     self.model = Doc2Vec.load(file_name)
    #     # load the optimal trajectories
    #     self._load_optimal_trajectories()
    #     self._get_sa_sequences_from_traj()
    #     self._states_to_coord()
        


    
    
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
            
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Get vector representations of the trajectories
    # ----------------------------------------------------------------
    # def get_vector(self, traj_index):
    #     # select one of the vector representations of a sentence
    #     vector = self.model.docvecs[traj_index]
    #     return vector

        
    # def get_all_vectors(self):
    #     all_vectors = []
    #     for i in range(0, len(self.model.docvecs)):
    #         all_vectors.append(self.model.docvecs[i])
    #     return all_vectors
    
    # converts state representation into xy coordinates
    def _states_to_coord(self):
        self.state_coords = []
        for traj in self.state_sequences:
            traj_temp = []
            for s in traj:
                split_temp = s.split("-")
                traj_temp.append((int(split_temp[0]), int(split_temp[1])))
            self.state_coords.append(traj_temp)
            
        