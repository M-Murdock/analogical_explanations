# Reference: https://community.plotly.com/t/using-one-slider-to-control-multiple-subplots-not-multiple-traces/13955/4

from sklearn.decomposition import PCA
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import numpy as np
from trajectory import _get_sa_sequence

class InteractivePlot:
    def __init__(self, all_vector_embeddings, embedding_indices=[0, 1, 2, 3], labels=["A", "B", "C", "D"], optimal=[]):
        
        self.all_vector_embeddings = all_vector_embeddings
        self.embedding_indices = embedding_indices
        self.labels = labels
        self.optimal = optimal
        
        # perform PCA
        pca = PCA(n_components=2) 
        self.principal_components = pca.fit_transform(all_vector_embeddings)

        # make the graphs
        self.fig, self.ax = plt.subplots(1, 4)
        self.fig.set_figwidth(15)

        # All potential values for D
        slider_values = [i for i in range(0, len(all_vector_embeddings))]
        
        # create subplots for each graph
        self.parallelogram_axis = self.ax[0]
        self.trajectory_axis = self.ax[1]
        self.D_axis = self.ax[2]
        self.reset_axis = self.ax[3]
        
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
        self.parallelogram_axis.scatter([self.principal_components[self.embedding_indices[i]][0] for i in range(0,len(self.embedding_indices))], [self.principal_components[self.embedding_indices[i]][1] for i in range(0,len(self.embedding_indices))], c=(0.1, 0.2, 0.5, 0.3))
        
        # Draw a parallelogram
        # A -> B
        self.parallelogram_axis.plot([A[0], B[0]], [A[1], B[1]], linewidth=1, zorder=1) 
        # C -> D
        self.parallelogram_axis.plot([C[0], D[0]], [C[1], D[1]], linewidth=1, zorder=1)
        
        
    # def visualize_trajectory(self, labels=["A", "B", "C", "D"]):
    def visualize_trajectory(self):
        
        all_of_x = []
        all_of_y = []
        
        # based on the indices for A, B, C,and D 
        optimal_trajectories = [self.optimal[t] for t in self.embedding_indices]
        
        for trajectory in optimal_trajectories:
            states, _ = _get_sa_sequence(trajectory)
        
            x = [s[0] for s in states]
            y = [s[1] for s in states]
            all_of_x.append(x)
            all_of_y.append(y)
        
        for i in range(0, len(optimal_trajectories)):
            self.trajectory_axis.plot(all_of_x[i], all_of_y[i], c=(0.1, 0.2, 0.5, 0.3))
        

        self.trajectory_axis.set(xlabel='X', ylabel='Y',
            title='Trajectory graph')
        self.trajectory_axis.grid()
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
        
        