1) Sample a bunch of trajectories from the robot’s
policy (we’ll assume that all trajectories are
the same length). Create embeddings of each
trajectory based on the sequences of state-
action pairs and resulting reward. The reward
represents one dimension of the embedding
space.
2) Select a set of trajectory embeddings to be the
”common ground”. Show these behaviors to
the human.
3) Select a trajectory A which is outside of the
common ground. We want to explain this to
the user.
4) Searching the set of trajectory embeddings in
”common ground”, find the ones which are
closest to A along the ”reward” axis (using
euclidean distance) and store them as a set
of ”candidate trajectories”. That is, find the
trajectories which result in a similar reward.
5) Search the ”candidate trajectories” to find one
that is closest to trajectory A
6) This is our analogous trajectory. Display it to
the user.
7) Add the trajectory to the common ground.