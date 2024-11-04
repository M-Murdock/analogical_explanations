from simple_rl.tasks.grid_world import GridWorldMDPClass
from simple_rl.agents import QLearningAgent
from simple_rl.run_experiments import run_single_agent_on_mdp
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class EmbeddingSpace:
    def __init__(self, NUM_TRAJECTORIES=50, N_GRAM_TYPE="state-action", FILE_NAME="state-action.txt", MAP_NAME="maps/easygrid.txt"):
        self.NUM_TRAJECTORIES = NUM_TRAJECTORIES
        self.N_GRAM_TYPE = N_GRAM_TYPE
        self.FILE_NAME = FILE_NAME
        self.MAP_NAME = MAP_NAME
        
        # self.new_model()
        self.load_model()
        
    def load(self):
        self.load_model()
        
    def rebuild(self):
        self.new_model()
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------------- 

    def _save_optimal_trajectories(self, save_file_name="state-action.txt"):
        with open(save_file_name, "wb") as fp:
            pickle.dump(self.optimal_trajectories, fp)


    def _load_optimal_trajectories(self, load_file_name="state-action.txt"): 
        with open(load_file_name, "rb") as fp:   
            self.optimal_trajectories = pickle.load(fp)
                    
                

    def _tokenize(self, sentences):
        sentence_words = []

        # iterate through each sentence in the file
        for i in sent_tokenize(sentences):
            temp = []

            # tokenize the sentence into words
            for j in word_tokenize(i):
                if not j == '.': 
                    temp.append(j.lower())

            sentence_words.append(temp)
            
        return sentence_words
        
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Create a new model and save it for later retrieval
    # ----------------------------------------------------------------
    def new_model(self, file_name="model.bin"):
        # create optimal trajectories + rewards, then save them
        self._create_optimal_trajectories(episodes=1000, steps=200, slip_prob=0.1)
        self._save_optimal_trajectories()
        
        self._get_sa_sequences_from_traj()
        
        # convert trajectories to tokenized form
        sentences = self._traj_to_sentences()
        sentence_words = self._tokenize(sentences)
        
        # store all the sentences as TaggedDocument objects
        self.trajectory_objs = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence_words)]

        self._states_to_coord()
        # train the model on our sentences
        self.model = Doc2Vec(vector_size=100, min_count=3, epochs=20)
        self.model.build_vocab(self.trajectory_objs)
        self.model.train(self.trajectory_objs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        
        # save the model
        self.model.save(file_name)
        
        
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Load an existing model
    # ----------------------------------------------------------------
    def load_model(self, file_name="model.bin"):
        # load the model
        self.model = Doc2Vec.load(file_name)
        # load the optimal trajectories
        self._load_optimal_trajectories()
        self._get_sa_sequences_from_traj()
        self._states_to_coord()
        
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Create a set of optimal trajectories
    # ----------------------------------------------------------------
    def _create_optimal_trajectories(self, episodes=1000, steps=200, slip_prob=0.1):
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

    
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Convert trajectories to sentence format
    # ----------------------------------------------------------------
    def _traj_to_sentences(self, save_file_name="state-action.txt"):
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
            contents += "."
        
        return contents
        
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Generate state-action sequences based on optimal trajectories
    # ----------------------------------------------------------------
    def _get_sa_sequences_from_traj(self):
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

        return self.state_sequences, self.actions_sequences
    
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
    def get_vector(self, traj_index):
        # select one of the vector representations of a sentence
        vector = self.model.docvecs[traj_index]
        return vector

        
    def get_all_vectors(self):
        all_vectors = []
        for i in range(0, len(self.model.docvecs)):
            all_vectors.append(self.model.docvecs[i])
        return all_vectors
    
    # converts state representation into xy coordinates
    def _states_to_coord(self):
        self.state_coords = []
        for traj in self.state_sequences:
            traj_temp = []
            for s in traj:
                split_temp = s.split("-")
                traj_temp.append((int(split_temp[0]), int(split_temp[1])))
            self.state_coords.append(traj_temp)
            
            
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # Just for debugging purposes
    # ----------------------------------------------------------------       
        
    def test_parallelogram(self):
        data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]

        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
        max_epochs = 1000
        vec_size = 20
        alpha = 0.025

        model = Doc2Vec(vector_size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm =1)
        
        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
            
        A = word_tokenize("I love chatbots".lower())
        vA = model.infer_vector(A)
        B = word_tokenize("I love cats".lower())
        vB = model.infer_vector(B)
        C = word_tokenize("Aardvarks are nice".lower())
        vC = model.infer_vector(C)
        D = word_tokenize("Antelopes".lower())
        vD = model.infer_vector(D)
        
        pca = PCA(n_components=2) 
        principal_components = pca.fit_transform([vA, vB, vC, vD])

        # Get the coordinates of each point
        pA = principal_components[0]
        pB = principal_components[1]
        pC = principal_components[2]
        pD = principal_components[3]

        # Plot all the points
        plt.scatter([principal_components[i][0] for i in range(0,4)], [principal_components[i][1] for i in range(0,4)], color=['blue', 'black', 'red', 'orange'])
        
        # Draw a parallelogram
        # A -> B
        plt.plot([pA[0], pB[0]], [pA[1], pB[1]], linewidth=1, zorder=1, color="gray") 
        # C -> D
        plt.plot([pC[0], pD[0]], [pC[1], pD[1]], linewidth=1, zorder=1, color="gray")
        
        # to find most similar doc using tags
        similar_doc = model.docvecs.most_similar('3')
        print(similar_doc)
        
        plt.show()  