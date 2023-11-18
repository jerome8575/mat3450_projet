import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


population_params = {
    "num_age_classes": 7,
    "intial_population": [100, 85, 88, 52, 31, 15, 7],
    "fishing_mortality": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "natural_mortality": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "class_survival_rates": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0],
    "class_reproduction_rates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5],
    "class_mean_weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
}

simulation_params = {
    "num_time_steps": 100
}
class Simulation:

    def __init__(self, population_params, simulation_params):
        self.population_params = population_params
        self.simulation_params = simulation_params
        self.projection_matrix = self.build_projection_matrix()

    def run(self):
        x = self.population_params["intial_population"]
        sim_results = np.zeros((self.simulation_params["num_time_steps"], self.population_params["num_age_classes"]))
        sim_results[0, ] = x
        for t in range(self.simulation_params["num_time_steps"]):
            print("Running simulation step: ", t)
            x = self.step(x)
            sim_results[t, ] = x
        return sim_results

    def step(self, x):
        return self.projection_matrix @ x

    def build_projection_matrix(self):

        # create leslie matrix using population_params
        
        projection_matrix = np.zeros((self.population_params["num_age_classes"], self.population_params["num_age_classes"]))
        projection_matrix[0, ] = self.population_params["class_reproduction_rates"]
        projection_matrix[1:, :-1] = np.eye(self.population_params["num_age_classes"] - 1)
        projection_matrix[1:, ] *= self.population_params["class_survival_rates"]

        return projection_matrix

    def plot(self, sim_results):
        plt.plot(sim_results)
        plt.legend(range(self.population_params["num_age_classes"]))
        plt.xlabel("Time Step")
        plt.ylabel("Population")
        plt.show()

class Recruited_simulation():

    def __init__(self, population_params, simulation_params):
        pass

    def run(self):
        pass

    def plot(self):
        pass


simulation_instance = Simulation(population_params, simulation_params)
x = simulation_instance.run()
simulation_instance.plot(x)
print(x)
    