import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


population_params = {
    "num_age_classes": 7,
    "intial_population": [100, 85, 88, 52, 31, 15, 7],
    "fishing_mortality": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "natural_mortality": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    "class_survival_rates": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94,0],
    "class_reproduction_rates": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5],
    #"class_mean_weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "class_mean_weights": [0.4, 0.8, 0.9, 1.2, 1.5, 1.6, 1.9],
    "number_eggs":[0,0,22,22,25,25,25],
    
}

simulation_params = {
    "num_time_steps": 50
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
        self.population_params = population_params
        self.simulation_params = simulation_params
        self.projection_matrix = self.build_projection_matrix()

    def build_projection_matrix(self):
        # create matrix transformation 1
        matrix_A=np.diag(self.population_params["class_survival_rates"][:-1],-1)
        #s_0 = float(self.population_params['class_survival_rates'][0])
        s_0=0.1
        matrix_A[0,]= s_0*np.array(self.population_params["number_eggs"])
        projection_matrix = matrix_A
        print(matrix_A)
        return projection_matrix

    def run(self):
        x = self.population_params["intial_population"]
        sim_results = np.zeros((self.simulation_params["num_time_steps"], self.population_params["num_age_classes"]))
        sim_results[0, ] = x
        for t in range(self.simulation_params["num_time_steps"]):
            #print("Running simulation step: ", t)
            x = self.step(x)
            sim_results[t, ] = x
        return sim_results
    
    def step(self, x):
        mean_mortality = (np.array(self.population_params["fishing_mortality"])+np.array(self.population_params["natural_mortality"]))/2
        mean_mortality=np.append(0.1,mean_mortality)
        matrix_F=np.diag(np.array(mean_mortality)*10)
        matrix_w=np.diag(self.population_params["class_mean_weights"])
        Y = matrix_F @ matrix_w @ self.projection_matrix @ x
        #print("projection prochain t")
        print(matrix_F)
        print(matrix_w)
        print(Y)
        return matrix_F @ matrix_w @ self.projection_matrix @ x
        
        return self.projection_matrix @ x

    def plot(self,sim_results):
        plt.plot(sim_results)
        plt.legend(range(self.population_params["num_age_classes"]))
        plt.xlabel("Time Step")
        plt.ylabel("Population")
        plt.show()


simulation_instance = Simulation(population_params, simulation_params)
x = simulation_instance.run()
simulation_instance.plot(x)
print(x)

simulation_instance = Recruited_simulation(population_params, simulation_params)
y = simulation_instance.run()
simulation_instance.plot(y)
print(y)

