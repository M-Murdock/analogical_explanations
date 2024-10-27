from sklearn.decomposition import PCA
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import numpy as np

class InteractivePlot:
    def __init__(self, all_vector_embeddings, embedding_indices=[0, 1, 2, 3], labels=["A", "B", "C", "D"]):
        
        self.all_vector_embeddings = all_vector_embeddings
        self.embedding_indices = embedding_indices
        self.labels = labels
        
        # perform PCA
        pca = PCA(n_components=2) 
        self.principal_components = pca.fit_transform(all_vector_embeddings)


        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.25)

        self.plot_embeddings()
        
        D_axis = self.fig.add_axes([0.25, 0.15, 0.65, 0.03])

        slider_values = [i for i in range(0, len(all_vector_embeddings))]
        
        
        self.D_slider = Slider(
            D_axis, "D", min(slider_values), max(slider_values),
            valinit=-1, valstep=slider_values,
            color="green"
        )

        self.D_slider.on_changed(self.update)

        ax_reset = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(ax_reset, 'Reset', hovercolor='0.975')
        
        button.on_clicked(self.reset)
        plt.show()

        
    
    def plot_embeddings(self):
    
        # Get the coordinates of each point
        A = self.principal_components[self.embedding_indices[0]]
        B = self.principal_components[self.embedding_indices[1]]
        C = self.principal_components[self.embedding_indices[2]]
        D = self.principal_components[self.embedding_indices[3]]

        # Plot all the points
        self.ax.scatter([self.principal_components[self.embedding_indices[i]][0] for i in range(0,len(self.embedding_indices))], [self.principal_components[self.embedding_indices[i]][1] for i in range(0,len(self.embedding_indices))], c=(0.1, 0.2, 0.5, 0.3))
        
        # Draw a parallelogram
        # A -> B
        self.ax.plot([A[0], B[0]], [A[1], B[1]], linewidth=1, zorder=1) 
        # C -> D
        self.ax.plot([C[0], D[0]], [C[1], D[1]], linewidth=1, zorder=1)
        
        
        
        
    def update(self, val):
        self.ax.cla()  
        self.embedding_indices[3] = self.D_slider.val
        self.plot_embeddings()
        self.fig.canvas.draw_idle()
        
    def reset(self, event):
        self.embedding_indices[3] = self.embedding_indices[0]
        self.plot_embeddings()
        self.D_slider.reset()