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
    "class_mean_weights": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "number_eggs":[0,0,22,22,25,25,25],
}

walleye_population_params = {
    "num_age_classes": 12,
    "intial_population": [7253, 5715, 3430, 1677, 833, 341, 273, 61, 33, 4, 6, 3],
    "fishing_mortality": [0.007368, 0.110000, 0.289474, 0.372105, 0.365789, 0.323684, 0.463158, 0.463333, 0.598421, 0.610000, 0.459000, 1],
    "natural_mortality": [0.0355, 0.0355, 0.218627, 0.339639, 0.352652, 0.368801, 0.423892, 0.175503, 0.537224, 0.601486, 0.550866, 1],
    "class_survival_rates": [0.9645, 0.9645, 0.781373, 0.660361, 0.647348, 0.631199, 0.576108, 0.824497, 0.462776, 0.398514, 0.449134, 0],
    "class_reproduction_rates": [0, 0, 0, 0, 0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.1, 0],
    "number_eggs": [0, 0, 0, 0, 50000, 50000, 50000, 50000, 30000, 20000, 10000, 0],
    "recruitment_age": 4,
    "egg_survival_rate": 0.00025,
    "egg_survival_to_recruit_rate": 0.00005
}

simulation_params = {
    "num_time_steps": 15,
    "is_fishing": True
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
        nt = self.projection_matrix @ x

        if self.simulation_params["is_fishing"]:
            # apply fishing mortality
            nt = nt - np.array(self.population_params["fishing_mortality"]) * nt

        return nt

    def build_projection_matrix(self):

        # create leslie matrix using population_params
        
        projection_matrix = np.zeros((self.population_params["num_age_classes"], self.population_params["num_age_classes"]))
        projection_matrix[0, ] = np.multiply(self.population_params["class_reproduction_rates"], self.population_params["number_eggs"]) * [self.population_params["egg_survival_rate"]]
        projection_matrix[1:, :-1] = np.eye(self.population_params["num_age_classes"] - 1)
        projection_matrix[1:, ] *= self.population_params["class_survival_rates"]

        return projection_matrix

    def plot(self, sim_results):
        plt.plot(sim_results)
        plt.legend(range(1, self.population_params["num_age_classes"] + 1))
        plt.xlabel("Time Step")
        plt.ylabel("Population")
        plt.show()


class Recruited_population_simmulation:

    def __init__(self, population_params, simulation_params):
        self.population_params = population_params
        self.simulation_params = simulation_params
        self.ind_introduction_matrix = self.build_ind_introduction_matrix()
        self.transition_matrix = self.build_transition_matrix()
    
    def build_ind_introduction_matrix(self):
        num_recruited_age_classes = self.population_params["num_age_classes"] - self.population_params["recruitment_age"]
        matrix_R = np.zeros((num_recruited_age_classes, num_recruited_age_classes))    
        matrix_R[0, ] = np.multiply(self.population_params["class_reproduction_rates"][self.population_params["recruitment_age"]:], 
                                    self.population_params["number_eggs"][self.population_params["recruitment_age"]:]) * [self.population_params["egg_survival_to_recruit_rate"]]
        
        return matrix_R
    
    def build_transition_matrix(self):
        num_recruited_age_classes = self.population_params["num_age_classes"] - self.population_params["recruitment_age"]
        matrix_D = np.zeros((num_recruited_age_classes, num_recruited_age_classes)) 
        matrix_D[1:, :-1] = np.eye(num_recruited_age_classes - 1)
        matrix_D[1:, ] *= self.population_params["class_survival_rates"][self.population_params["recruitment_age"]:]

        return matrix_D
    
    def step(self, N_t_r, N_t_1):
        nt = self.ind_introduction_matrix @ N_t_r + self.transition_matrix @ N_t_1

        # apply fishing mortality
        if self.simulation_params["is_fishing"]:
            nt = nt - np.array(self.population_params["fishing_mortality"][self.population_params["recruitment_age"]:]) * nt
        return nt
    
    def plot(self, sim_results):
        plt.plot(sim_results)
        #plt.legend("Age Class: " + str(i + self.population_params["recruitment_age"]) for i in range(self.population_params["num_age_classes"] - self.population_params["recruitment_age"]))
        plt.legend(range(self.population_params["recruitment_age"], self.population_params["num_age_classes"] + 1))
        plt.xlabel("Time Step")
        plt.ylabel("Population")
        plt.show()

    def run(self):

        # vecteur qui gardera les résultats de la simulation
        sim_results = np.zeros((self.simulation_params["num_time_steps"], self.population_params["num_age_classes"] - self.population_params["recruitment_age"]))

        # Pour la simulation de la population recrutée, on a besoin de générer N_0 jusqu'à N_t-r initialement
        # on le fait avec la simulation classique

        # generate N_0 to N_t-r
        classic_simulation = Simulation(self.population_params, {"num_time_steps": self.population_params["recruitment_age"], "is_fishing": False})
        classic_simulation_results = classic_simulation.run()

        # keep just recruited population
        sim_results[:self.population_params["recruitment_age"], ] = classic_simulation_results[:self.population_params["recruitment_age"], self.population_params["recruitment_age"]:]

        # start recruited population simulation

        for t in range(self.population_params["recruitment_age"], self.simulation_params["num_time_steps"]):
            print("Running simulation step: ", t)
            sim_results[t, ] = self.step(sim_results[t-self.population_params["recruitment_age"], ], sim_results[t- 1, ])
        
        return sim_results

        

            


    




"""class Recruited_simulation():

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
            print("Running simulation step: ", t)
            x = self.step(x)
            sim_results[t, ] = x
        return sim_results
    
    def step(self, x):
        mean_mortality = (np.array(self.population_params["fishing_mortality"])+np.array(self.population_params["natural_mortality"]))/2
        mean_mortality=np.append(0.1,mean_mortality)
        matrix_F=np.diag(mean_mortality)
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
        plt.show()"""


simulation_instance = Simulation(walleye_population_params, simulation_params)
x = simulation_instance.run()
simulation_instance.plot(x)


simulation_instance = Recruited_population_simmulation(walleye_population_params, simulation_params)
results = simulation_instance.run()
simulation_instance.plot(results)



