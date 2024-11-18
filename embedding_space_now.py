from simple_rl.tasks.taxi import TaxiOOMDPClass
from simple_rl.tasks import TaxiOOMDP
# from simple_rl.tasks.grid_world.GridWorldMDPClass import GridWorldMDP
from simple_rl.agents import QLearningAgent
from simple_rl.run_experiments import run_single_agent_on_mdp
import pickle
from behavior_model import BehaviorModel
from scipy import spatial
import os

class EmbeddingSpace:
    def __init__(self, N_GRAM_TYPE="state-action", MAP_DIRECTORY="maps/"):
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
        agents =[ {"x":1, "y":1, "has_passenger":0}, {"x":1, "y":2, "has_passenger":0}, {"x":2, "y":2, "has_passenger":0}, {"x":3, "y":2, "has_passenger":0}, {"x":3, "y":3, "has_passenger":0}]
        passengers = [{"x":4, "y":3, "dest_x":2, "dest_y":2, "in_taxi":0}]
        walls = []
        # taxi_mdp = TaxiOOMDP(5, 5, agent_loc=agent, walls=walls, passengers=passengers)
        # self.mdps = [TaxiOOMDP(5, 5, agent=a, walls=walls, passengers=passengers) for a in agents]
        # create MDPs for each possible trajectory (i.e. from every possible start position)
        self.mdps = [self.make_grid_world_from_file(file_name="map_bases/easytaxi.txt")]
        
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
        print(contents)
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
    
    def make_grid_world_from_file(file_name, randomize=False, num_goals=1, name=None, goal_num=None, slip_prob=0.0):
        '''
        Args:
            file_name (str)
            randomize (bool): If true, chooses a random agent location and goal location.
            num_goals (int)
            name (str)

        Returns:
            (GridWorldMDP)

        Summary:
            Builds a GridWorldMDP from a file:
                'w' --> wall
                'a' --> agent
                'g' --> goal
                'l' --> lava
                '-' --> empty
        '''
        
# agents =[ {"x":1, "y":1, "has_passenger":0}, {"x":1, "y":2, "has_passenger":0}, {"x":2, "y":2, "has_passenger":0}, {"x":3, "y":2, "has_passenger":0}, {"x":3, "y":3, "has_passenger":0}]
#         passengers = [{"x":4, "y":3, "dest_x":2, "dest_y":2, "in_taxi":0}]
#         walls = []
#         # taxi_mdp = TaxiOOMDP(5, 5, agent_loc=agent, walls=walls, passengers=passengers)
#         self.mdps = [TaxiOOMDP(5, 5, agent=a, walls=walls, passengers=passengers) for a in agents]

    #     agent = {"x":1, "y":1, "has_passenger":0}
    # passengers = [{"x":8, "y":4, "dest_x":2, "dest_y":2, "in_taxi":0}]
    # taxi_world = TaxiOOMDP(10, 10, agent=agent, walls=[], passengers=passengers)

        # if name is None:
        #     name = file_name.split(".")[0]

        # grid_path = os.path.dirname(os.path.realpath(__file__))
        wall_file = open(file_name)
        wall_lines = wall_file.readlines()

        # Get walls, agent, goal loc.
        num_rows = len(wall_lines)
        num_cols = len(wall_lines[0].strip())
        empty_cells = []
        agent_x, agent_y = 1, 1
        walls = []
        goal_locs = []
        passenger_locs = []

        for i, line in enumerate(wall_lines):
            line = line.strip()
            for j, ch in enumerate(line):
                if ch == "w":
                    walls.append((j + 1, num_rows - i))
                elif ch == "g":
                    goal_locs.append((j + 1, num_rows - i))
                elif ch == "p":
                    passenger_locs.append((j + 1, num_rows - i))
                elif ch == "a":
                    agent_x, agent_y = j + 1, num_rows - i
                elif ch == "-":
                    empty_cells.append((j + 1, num_rows - i))

        if goal_num is not None:
            goal_locs = [goal_locs[goal_num % len(goal_locs)]]

        # if randomize:
        #     agent_x, agent_y = random.choice(empty_cells)
        #     if len(goal_locs) == 0:
        #         # Sample @num_goals random goal locations.
        #         goal_locs = random.sample(empty_cells, num_goals)
        #     else:
        #         goal_locs = random.sample(goal_locs, num_goals)

        if len(goal_locs) == 0:
            goal_locs = [(num_cols, num_rows)]

        # return GridWorldMDP(width=num_cols, height=num_rows, init_loc=(agent_x, agent_y), goal_locs=goal_locs, lava_locs=lava_locs, walls=walls, name=name, slip_prob=slip_prob)
        return TaxiOOMDP(width=num_cols, height=num_rows, agent={"x":agent_x, "y":agent_y, "has_passenger":0}, walls=walls, passengers=[{"x":passenger_locs[0][0], "y":passenger_locs[0][1], "dest_x":goal_locs[0][0], "dest_y":goal_locs[0][1], "in_taxi":0}])

            