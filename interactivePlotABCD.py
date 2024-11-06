# Reference: https://community.plotly.com/t/using-one-slider-to-control-multiple-subplots-not-multiple-traces/13955/4

from sklearn.decomposition import PCA
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt

class InteractivePlotABCD:

    def __init__(self, embedding_space, embedding_indices=[0, 0, 0, 0], abcd_colors=['blue', 'green', 'red' , 'purple'],):
        self.embedding_space = embedding_space
        self.vector_embeddings = embedding_space.vectors
        self.embedding_indices = embedding_indices
        self.state_coords = self.embedding_space.state_coords
        
        self.abcd_colors = abcd_colors
        
        # perform PCA
        pca = PCA(n_components=2) 
        self.principal_components = pca.fit_transform(self.vector_embeddings)

        # create list of all the optimal trajectories (so we only have to do it once)
        self._generate_visual_trajectories()
        

        # All potential values for D
        slider_values = [i for i in range(0, len(self.vector_embeddings))]
        
        # make the graphs
        self.fig = plt.figure(constrained_layout=True, figsize=(12, 10))
        widths = [20, 20, 1, 1, 1, 1, 1, 1, 1]
        heights = [20, 5, 2, 2, 5, 2, 5, 5, 1]
        spec = self.fig.add_gridspec(ncols=9, nrows=9, width_ratios=widths, height_ratios=heights)
        
        self.parallelogram_axis = self.fig.add_subplot(spec[0, 0])

        self.trajectory_axis = self.fig.add_subplot(spec[0, 1])

        self.reset_axis = self.fig.add_subplot(spec[1, 0])
        
        self.infer_axis = self.fig.add_subplot(spec[1, 1])
        
        self.A_axis = self.fig.add_subplot(spec[2, 0])
        
        self.B_axis = self.fig.add_subplot(spec[2, 1])

        self.C_axis = self.fig.add_subplot(spec[3, 0])
        
        self.D_axis = self.fig.add_subplot(spec[3, 1])
        
        self.radio_axis = self.fig.add_subplot(spec[4, 0])
        
        
        # create slider A
        self.A_slider = Slider(
            self.A_axis, "A", min(slider_values), max(slider_values),
            valinit=-1, valstep=slider_values,
            color=self.abcd_colors[0]
        )
        self.A_slider.on_changed(self.update)
        
        # create slider B
        self.B_slider = Slider(
            self.B_axis, "B", min(slider_values), max(slider_values),
            valinit=-1, valstep=slider_values,
            color=self.abcd_colors[1]
        )
        self.B_slider.on_changed(self.update)
        
        # create slider C
        self.C_slider = Slider(
            self.C_axis, "C", min(slider_values), max(slider_values),
            valinit=-1, valstep=slider_values,
            color=self.abcd_colors[2]
        )
        self.C_slider.on_changed(self.update)
        
        # create slider D
        self.D_slider = Slider(
            self.D_axis, "D", min(slider_values), max(slider_values),
            valinit=-1, valstep=slider_values,
            color=self.abcd_colors[3]
        )
        self.D_slider.on_changed(self.update)
        
        # create reset button
        self.button = Button(self.reset_axis, 'Reset', hovercolor='0.975') 
        self.button.on_clicked(self.reset)
        
        # create 'infer D' button
        self.infer_button = Button(self.infer_axis, 'Infer D', hovercolor='0.975') 
        self.infer_button.on_clicked(self.infer)

        # create radio buttons
        self.radio_button = RadioButtons(self.radio_axis, ["Inference Mode", "Free Play Mode"], active=0) 
        self.radio_button.on_clicked(self.mode_selection)
        
        self.mode_selection("Inference Mode")
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
        
        for t in range(0, len(self.state_coords)-1):
            x = [s[0] for s in self.state_coords[t]]
            y = [s[1] for s in self.state_coords[t]]
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
        
        # Set A to be whatever the user specified
        self.embedding_indices[0] = self.A_slider.val
        # Set B to be whatever the user specified
        self.embedding_indices[1] = self.B_slider.val
        # Set C to be whatever the user specified
        self.embedding_indices[2] = self.C_slider.val
        # Set D to be whatever the user specified
        self.embedding_indices[3] = self.D_slider.val
        
        # redraw the graphs
        self.plot_embeddings()
        self.visualize_trajectory()  # re-draw the trajectory graph
        self.fig.canvas.draw_idle()
        
    def reset(self, event):
        # reset the plot to the initial state
        self.embedding_indices[0] = self.embedding_indices[0]
        self.embedding_indices[1] = self.embedding_indices[0]
        self.embedding_indices[2] = self.embedding_indices[0]
        self.embedding_indices[3] = self.embedding_indices[0]
        self.A_slider.reset()
        self.B_slider.reset()
        self.C_slider.reset()  
        self.D_slider.reset()
        self.plot_embeddings()
        
    def infer(self, event):
        index = self.embedding_space.infer_D(ABC_indices=self.embedding_indices[0:3])

        self.embedding_indices[3] = self.principal_components[index]
        self.D_slider.set_val(index)

        self.plot_embeddings()
        self.visualize_trajectory()
        
    def mode_selection(self, event):
        if event == "Inference Mode":
            self.D_slider.active = False
            self.infer_button.active = True 
            
            self.D_axis.set_visible(False)
            self.infer_axis.set_visible(True)
        else:
            self.D_slider.active = True
            self.infer_button.active = False 
            
            self.D_axis.set_visible(True)
            self.infer_axis.set_visible(False)

        
        