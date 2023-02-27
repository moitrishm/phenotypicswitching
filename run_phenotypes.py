'''
Using functions from the Phenotype method and parameters from parameters.json, obtain the population
of the bacterial colony which grows from t=0 till t=time_t.
'''


import networkx as nx   
from scipy.integrate import odeint
import numpy as np
import json

from phenotypes import Phenotypes

if __name__ == '__main__':
  with open('parameters.json') as jsonFile:
        params = json.load(jsonFile)
  p = Phenotypes(params)
  A = p.generate_sparse_matrices(params['edge_weights'], params['proportion'])   
  env = p.generate_environment_sequence()       
  env_seq, indiv_times, total_times = p.generate_durations1(env)
  solf, tf = p.solver_simulate(A, env_seq, indiv_times)
  np.savez('data.npz', matrices=A, population=solf, envs = env_seq, times=indiv_times)