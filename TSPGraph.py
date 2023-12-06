import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from numba import jit

class Graph:
    def __init__(self, dataset: chr):
        self.data_frame = pd.read_csv("Data/TSP{}.csv".format(dataset.capitalize()), names=['x','y','cost'], index_col=False, sep=';')
        self.points = list(zip(self.data_frame['x'],self.data_frame['y']))
        self.dist_matrix = distance.cdist(self.points, self.points, 'euclidean')
        np.rint(self.dist_matrix, out=self.dist_matrix)
        self.dist_matrix = self.dist_matrix.astype(int)
        self.objective = self.dist_matrix + np.array(self.data_frame['cost'])

    def cycle_length(self, cycle):
        cycle_segments = (cycle[:-1], cycle[1:])
        return np.sum(self.dist_matrix[cycle_segments])
    
    def cycle_cost(self, cycle):
        cycle_segments = (cycle[:-1], cycle[1:])
        return np.sum(self.objective[cycle_segments])
        
    def plot(self, cycle=None):
        plt.figure(figsize=(10,7))
        plt.scatter(self.data_frame['x'], self.data_frame['y'], s=self.data_frame['cost']/15)
        if len(cycle) > 0:
            cycle_values = self.data_frame.iloc[cycle]
            plt.plot(cycle_values['x'],cycle_values['y'],'r')
        plt.title("Travelling salesman problem")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
        return plt.figure

    def anim_plot(self,cycles):
        fig, ax = plt.subplots()
        plt.scatter(self.data_frame['x'], self.data_frame['y'], s=self.data_frame['cost']/11)
        plt.title("Travelling salesman problem")
        plt.xlabel("X")
        plt.ylabel("Y")
        line, = ax.plot([], [], 'r', lw=2)
        def animate(frame):
            cycle_values = self.data_frame.iloc[cycles[frame]]
            cycle_values = cycle_values.append(self.data_frame.iloc[cycles[frame][0]])
            line.set_data(cycle_values['x'], cycle_values['y'])

        ani = FuncAnimation(fig, animate, frames=len(cycles))
        ani = HTML(ani.to_jshtml())
        return ani
        
