from simple_rl.tasks.grid_world import GridWorldMDPClass
from simple_rl.agents import QLearningAgent
from simple_rl.run_experiments import run_single_agent_on_mdp
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class EmbeddingSpace:
    def __init__(self, NUM_TRAJECTORIES=50, N_GRAM_TYPE="state-action", FILE_NAME="state-action.txt", MAP_NAME="maps/easygrid.txt"):
        self.NUM_TRAJECTORIES = NUM_TRAJECTORIES
        self.N_GRAM_TYPE = N_GRAM_TYPE
        self.FILE_NAME = FILE_NAME
        self.MAP_NAME = MAP_NAME
        
        self._create_optimal_trajectories(episodes=1000, steps=200, slip_prob=0.1)
        self._save_traj_to_file()
        self._create_ngrams() # train the model
    
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
        

    
    # -------------------------
    # -------------------------
    # get the lists of states and actions
    def _save_traj_to_file(self):
        contents = ""
        s_seq, a_seq = self.generate_sa_sequences()
        
        for traj_num in range(0, len(s_seq)): # go through each trajectory
            for i in range(0, len(a_seq[traj_num])):
                # ----------------------------------------------------------------
                if self.N_GRAM_TYPE == "state-action":
                    # save state-action pairs to the file
                    contents += " " + s_seq[traj_num][i] + "" + a_seq[traj_num][i]
                # ----------------------------------------------------------------        
                if self.N_GRAM_TYPE == "action-reward":
                    # save action-reward pairs to the file
                    contents += " " + a_seq[traj_num][i] + "" + str(self.rewards[traj_num][i])   
                # ----------------------------------------------------------------
                if self.N_GRAM_TYPE == "state-reward":
                    # save state to the file
                    contents += " " + s_seq[traj_num][i] + "" + str(self.rewards[traj_num][i])   
                # ----------------------------------------------------------------
                if self.N_GRAM_TYPE == "states":
                    # save state sequence to the file
                    contents += " " + s_seq[traj_num][i] + ""
                # ----------------------------------------------------------------
                if self.N_GRAM_TYPE == "actions":
                    # save action sequence to the file
                    contents += " " + a_seq[traj_num][i] + ""
                # ----------------------------------------------------------------
                if self.N_GRAM_TYPE == "state-action-reward":
                    # save state-action-reward to the file
                    contents += " " + s_seq[traj_num][i] + "" + a_seq[traj_num][i] + str(self.rewards[traj_num][i]) 
                # ----------------------------------------------------------------  
            contents += "."
        
                
        f = open(self.FILE_NAME, "w")
        f.write(contents)
        f.close()
        
    def generate_sa_sequences(self):
        states = [] 
        actions = []
        
        for traj in self.optimal_trajectories:
            # get the list of states for the current trajectory
            s_sequence = [t[0] for t in traj]
            
            temp_state = []
            # get the string for the current state
            for s in s_sequence:
                temp_state.append("" + str(s[0]) + "-" + str(s[1]))
                
            a_sequence = [t[1] for t in traj]   
            actions.append(a_sequence)
            states.append(temp_state)   
        return states, actions
        
    # NOTE: this function is specific to this gridworld environment
    # def _get_sa_sequences(self):
    #     actions = []
    #     states = []
        
    #     for traj in self.optimal_trajectories:
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
    
    # def _get_sa_sequence(self, trajectory):
    #     actions = []
    #     states = []
        
    #     # get the list of states for the current trajectory
    #     states = [(t[0][0],t[0][1]) for t in trajectory]    
    #     actions = [t[1] for t in trajectory]    
        
    #     return states, actions
    
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
            
    def get_vector(self, traj_index):
        # select one of the vector representations of a sentence
        vector = self.model.infer_vector(self.trajectory_objs[traj_index].words)
        return vector
    
        # model, trajectory_objs = self._create_ngrams(("saved_trajectories/" + N_GRAM_TYPE + ".txt"))
        
    def get_all_vectors(self):
        all_vectors = []
        for i in range(0, len(self.trajectory_objs)):
            all_vectors.append(self.get_vector(i))
        return all_vectors
        
    def _create_ngrams(self):
        #open and read the file of trajectories:
        sample = open(self.FILE_NAME)
        sentences = sample.read()
        
        sentence_words = []

        # iterate through each sentence in the file
        for i in sent_tokenize(sentences):
            temp = []

            # tokenize the sentence into words
            for j in word_tokenize(i):
                if not j == '.': 
                    temp.append(j.lower())

            sentence_words.append(temp) 


        # store all the sentences as TaggedDocument objects
        self.trajectory_objs = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentence_words)]

        # train the model on our sentences
        self.model = Doc2Vec(vector_size=100, min_count=3, epochs=20)
        self.model.build_vocab(self.trajectory_objs)
        self.model.train(self.trajectory_objs, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        
        # return the model and list of document objects
        # return model, trajectory_objs
