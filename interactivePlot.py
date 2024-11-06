# Reference: https://community.plotly.com/t/using-one-slider-to-control-multiple-subplots-not-multiple-traces/13955/4
# Takes a list of vector embeddings as input 

from sklearn.decomposition import PCA
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import numpy as np
from embedding_space import EmbeddingSpace

class InteractivePlot:

    def __init__(self, embedding_space, embedding_indices=[0, 1, 2, 3], abcd_colors=['blue', 'green', 'red' , 'purple'],):
        self.embedding_space = embedding_space
        self.all_vector_embeddings = embedding_space.get_all_vectors()
        self.embedding_indices = embedding_indices
        
        self.abcd_colors = abcd_colors
        
        # perform PCA
        pca = PCA(n_components=2) 
        self.principal_components = pca.fit_transform(self.all_vector_embeddings)

        # create list of all the optimal trajectories (so we only have to do it once)
        self._generate_visual_trajectories()
        
        # make the graphs
        # self.fig, self.ax = plt.subplots(1, 4)
        self.fig, self.ax = plt.subplots(nrows=2, ncols=2)
        self.fig.set_figwidth(15)

        # All potential values for D
        slider_values = [i for i in range(0, len(self.all_vector_embeddings))]
        
        # create subplots for each graph
        self.parallelogram_axis = self.ax[0, 0]
        self.trajectory_axis = self.ax[0, 1]
        self.D_axis = self.ax[1, 0]
        self.reset_axis = self.ax[1, 1]
        
        # create slider
        self.D_slider = Slider(
            self.D_axis, "D", min(slider_values), max(slider_values),
            valinit=-1, valstep=slider_values,
            color="green"
        )
        self.D_slider.on_changed(self.update)
        
        # create reset button
        button = Button(self.reset_axis, 'Reset', hovercolor='0.975') 
        button.on_clicked(self.reset)

        # plot our data
        self.plot_embeddings()
        self.visualize_trajectory()
        plt.show()

    
    def plot_embeddings(self):
    
        # Get the coordinates of each point
        A = self.principal_components[self.embedding_indices[0]]
        B = self.principal_components[self.embedding_indices[1]]
        C = self.principal_components[self.embedding_indices[2]]
        D = self.principal_components[self.embedding_indices[3]]

        # Plot all the points
        self.parallelogram_axis.scatter([self.principal_components[self.embedding_indices[i]][0] for i in range(0,len(self.embedding_indices))], [self.principal_components[self.embedding_indices[i]][1] for i in range(0,len(self.embedding_indices))], color=self.abcd_colors)
        
        # Draw a parallelogram
        # A -> B
        self.parallelogram_axis.plot([A[0], B[0]], [A[1], B[1]], linewidth=1, zorder=1, color="gray") 
        # C -> D
        self.parallelogram_axis.plot([C[0], D[0]], [C[1], D[1]], linewidth=1, zorder=1, color="gray")
        
        
    def _generate_visual_trajectories(self): # NOTE: Fix this to make it more efficient 
        self.all_of_x = []
        self.all_of_y = []
        
        for t in range(0, len(self.principal_components)):
            x = [s[0] for s in self.embedding_space.state_coords[t]]
            y = [s[1] for s in self.embedding_space.state_coords[t]]
            self.all_of_x.append(x)
            self.all_of_y.append(y)


    def visualize_trajectory(self):
        
        # based on the indices for A, B, C,and D 
        for e in range(0, len(self.embedding_indices)):
            self.trajectory_axis.plot(self.all_of_x[self.embedding_indices[e]], self.all_of_y[self.embedding_indices[e]], color=self.abcd_colors[e], linewidth=1, linestyle='--')
            # indicate the star positions with colored dots
            self.trajectory_axis.scatter(self.all_of_x[self.embedding_indices[e]][0],self.all_of_y[self.embedding_indices[e]][0], color=self.abcd_colors[e], s=100)
        
        
        self.trajectory_axis.set_yticklabels([])
        self.trajectory_axis.set_xticklabels([])
        plt.show()   
        
        
    def update(self, val):
        # Clear the current plot
        self.parallelogram_axis.cla() 
        self.trajectory_axis.cla()  
        
        # Set D to be whatever the user specified
        self.embedding_indices[3] = self.D_slider.val
        
        # redraw the graphs
        self.plot_embeddings()
        self.visualize_trajectory()  # re-draw the trajectory graph
        self.fig.canvas.draw_idle()
        
    def reset(self, event):
        # reset the plot to the initial state
        self.embedding_indices[3] = self.embedding_indices[0]
        self.D_slider.reset()
        self.plot_embeddings()
        
        