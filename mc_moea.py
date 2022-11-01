import pickle
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import uuid
from pandarallel import pandarallel
from tqdm.notebook import tqdm
from copy import deepcopy

pandarallel.initialize()

types = ["900 MHz Type I", "900 MHz Type II", "1800 MHz Type I", "1800 MHz Type II", "2600 MHz Type I", "2600 MHz Type II"]
frequency = [900, 900, 1800, 1800, 2600, 2600]
capacity = [800, 1200, 850, 1250, 800 ,1300]
cost = [1150000, 1500000, 880000, 1220000, 950000, 1350000]
station_types = pd.DataFrame({"types": types, "frequency":frequency, "capacity":capacity, "cost":cost})

bound_max = 800
bound_min = -800

def generate_station(n):
    stations = []
    for i in range(n):
        s_info = station_types.sample(n=1).iloc[0]
        pos = np.random.uniform(-800, 800, size=2)
        s = Station(s_info["types"], pos[0], pos[1], s_info["frequency"], s_info["capacity"], s_info["cost"])
        stations.append(s)
    return stations

def distance_scale(p1, p2, scale=1):
    return (((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**(1/2)) * scale

def stations_crossover(s1, s2):
    pos1 = np.random.randint(len(s1)-1) if len(s1) > 1 else 0
    pos2 = np.random.randint(len(s2)-1) if len(s2) > 1 else 0
    s_result = deepcopy(s1)
    s_result[pos1] = s2[pos2]
    return s_result

def fspl(d, f):
    return - ((20*np.log10(d)) + (20*np.log10(f)) - 32.44)

def stations_mutation(s):
    mutation_pos = np.random.randint(len(s)-1)  if len(s) > 1 else 0
    s_result = deepcopy(s)
    s_info = station_types.sample(n=1).iloc[0]
    pos = np.random.uniform(bound_min, bound_max, 2)
    s = Station(s_info["types"], pos[0], pos[1], s_info["frequency"], s_info["capacity"], s_info["cost"])
    s_result[mutation_pos] = s
    return s_result

def assign_new_station(solution, stations):
    solution_local = deepcopy(solution)
    solution_local.stations = deepcopy(stations)
    solution_local.reset()
    return solution_local

def crossover(population, crossover_prop):
    population = deepcopy(population)
    crossover_df1 = population[["solution_obj"]].sample(frac=crossover_prop/2).reset_index(drop=True)
    crossover_df2 = population[["solution_obj"]].sample(frac=crossover_prop/2).reset_index(drop=True)
    crossover_df = pd.concat([crossover_df1, crossover_df2], axis=1)
    crossover_df.columns = ["solution_obj_x","solution_obj_y"]
    crossover_df["new_station"] = crossover_df.parallel_apply(lambda x:stations_crossover(x["solution_obj_x"].stations, x["solution_obj_y"].stations), axis=1)
    crossover_df["solution_obj_z"] = crossover_df.parallel_apply(lambda x:assign_new_station(x["solution_obj_x"], x["new_station"]), axis=1)
    crossover_df.drop(columns=["solution_obj_x", "solution_obj_y", "new_station"], inplace=True)
    crossover_df = crossover_df.rename(columns={"solution_obj_z":"solution_obj"})
    return crossover_df

def mutation(population, mutation_p):
    population = deepcopy(population)
    mutation_df = population[["solution_obj"]].sample(frac=mutation_p).reset_index(drop=True)
    mutation_df["new_station"] = mutation_df.parallel_apply(lambda x:stations_mutation(x["solution_obj"].stations), axis=1)
    mutation_df["solution_obj_z"] = mutation_df.parallel_apply(lambda x:assign_new_station(x["solution_obj"], x["new_station"]), axis=1)
    mutation_df.drop(columns=["solution_obj", "new_station"], inplace=True)
    mutation_df = mutation_df.rename(columns={"solution_obj_z":"solution_obj"})
    return mutation_df

def compute(population, position):
    population = deepcopy(population)
    population["solution_obj"] = population["solution_obj"].parallel_apply(lambda x:x.compute_infos(position))
    population["signal_strength"] = population["solution_obj"].parallel_apply(lambda x:x.signal_strength)
    population["construction_cost"] = population["solution_obj"].parallel_apply(lambda x:x.construction_cost)
    population["num_orphans"] = population["solution_obj"].parallel_apply(lambda x:x.num_orphans)
    population["fitness_value"] = population["solution_obj"].parallel_apply(lambda x:x.fitness)
    
    return population

class Solution(object):
    def __init__(self, stations):
        self.id = str(uuid.uuid4())
        self.stations = deepcopy(stations)
        self.is_compute = False
        self.signal_strength_l = []
        self.num_orphans = 0
        self.signal_strength = np.nan
        self.construction_cost = np.nan
        self.fitness = np.nan
        
    def compute_infos(self, position):
        for sb in position:
            signal_strength = []
            for idx, st in enumerate(self.stations):
                if not st.is_full():
                        d = distance_scale((sb[0], sb[1]), (st.x, st.y), scale=12)
                        p = fspl(d, st.frequency)
                        signal_strength.append((idx, p))
            if signal_strength:        
                signal_strength.sort(key=lambda x:x[1], reverse=True)
                self.stations[signal_strength[0][0]].new_subscriber()
                self.signal_strength_l.append(signal_strength[0][1])
            else:
                self.num_orphans += 1
                self.signal_strength_l.append(np.nan)

        self.signal_strength = np.nanmean(self.signal_strength_l)
        self.construction_cost = sum([x.cost for x in self.stations])
        self.fitness = -(self.signal_strength/110) + (self.construction_cost/40000000) + (self.num_orphans/1000)
        self.is_compute = True
        return self
    
    def reset(self):
        self.id = str(uuid.uuid4())
        self.is_compute = False
        self.signal_strength_l = []
        self.num_orphans = 0
        self.signal_strength = np.nan
        self.construction_cost = np.nan
        self.fitness = np.nan
        for s in self.stations:
            s.id = str(uuid.uuid4())
            s.current_capacity = 0
    
    def __str__(self):
        ret = f"""Solution : {self.id}\n\tComputed?: {self.is_compute}\n\tFitness value: {self.fitness}\n\tAverage signal stength: {self.signal_strength}\n\tOrphans: {self.num_orphans}\n\tConstruction cost: {self.construction_cost}"""
        return ret
    
class Station(object):
    def __init__(self, s_type, x, y, frequency, capacity, cost):
        self.id = str(uuid.uuid4())
        self.type = s_type
        self.x = x
        self.y = y
        self.frequency = frequency
        self.capacity = capacity
        self.current_capacity = 0
        self.cost = cost
    
    def is_full(self):
        return True if self.current_capacity >= self.capacity else False
    
    def new_subscriber(self):
        if not self.is_full():
            self.current_capacity += 1
    
    def __str__(self):
        return f"""\rStation
        \r\tName: {self.id}
        \r\tType: {self.type}
        \r\tPosition: ({self.x}, {self.y})
        \r\tFrequency: {self.frequency}
        \r\tCapacity: {self.current_capacity}/{self.capacity}
        \r\tCost: {self.cost}
        """