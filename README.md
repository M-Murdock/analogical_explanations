<!-- ## Usage
``vlm_similarities.py`` -->

## Files
``AE.py`` - Autoencoder for creating embedding space

``Agents.py`` - Creates and generates trajectories for gridworld agents

``generate_maps.py`` - Given a template for a gridworld map (.txt file), generates maps (also.txt) that place the agent in each possible start location

``generate_traj_imgs.py`` - Uses state data to represent each trajectory on a 2D graph  

``gridworld_embedding_space.py`` -  Creates an embedding space of agent trajectories  

``gridworld.py`` - Tests the trajectory embedding space  

``interactivePlotABCD.py`` - Represents trajectories and embeddings in an interactive GUI

``llm_similarities.py`` - Using trajectories formatted as natural language prompts, uses LLM to try to identify analogous trajectories

``vlm_similarities.py`` - Similar to ``llm_similarities.py``, but uses graphical representations of trajectories