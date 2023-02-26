'''
  Creates a network of phenotypes which can either be a path
  or a sparse graph
'''

import networkx as nx
import matplotlib.pyplot as plt    
from scipy.integrate import odeint
import numpy as np
from numpy import linalg as LA

class Phenotypes(object): 
  
  def set_path_weights(self,nodes,h):     #create a path type network with #nodes = nodes 
    H = nx.path_graph(nodes, create_using=nx.Graph)
    H1 = nx.to_numpy_array(H)
    return H1*h

  def set_edge_weights(self,nodes, h, w):  #creates a fully connected network where two edges are weighted differently
    H = np.full((nodes,nodes),h)
    H[0][nodes-1] = h - w
    H[nodes-1][0] = h - w
    np.fill_diagonal(H,0)
    return H
  
  def set_edge_weights_sparse(self,nodes, h, i):      #random graph with (100*i)% of possible edges drawn
    H=nx.gnm_random_graph(nodes, i*nodes*(nodes-1), directed=True)    
    while(not(nx.is_strongly_connected(H))):
      H=nx.gnm_random_graph(nodes, i*nodes*(nodes-1), directed=True)    #ensure graph is strongly connected
    H1=nx.to_numpy_array(H) #adjacency matrix with all edges = 1
    H2= h* H1             #all edge weights = h
    H2=np.transpose(H2)
    return H2
    
  def rate_calc(self,H):  #calculates sum of edge weights incident to each node
    rates = np.sum(H, axis=0)    
    return rates    
    
  def laplace(self,G,H,rates):        #calculates a laplacian matrix for each network type
    Gn = np.zeros((G.shape[0], G.shape[1]))
    np.fill_diagonal(Gn, G.diagonal() - rates )     
    return Gn+H

  def env_matrix_creator(self, nodes, k, s):      #creates a diagonal matrix with growth rates
    G = np.zeros((nodes,nodes))   #growth rates of every phenotype
    temp = np.full(nodes, 0.1)    #set all growth rates = 0.1
    temp[k] += s/10             #only one of the growth rates is larger by s/10
    np.fill_diagonal(G, temp)            
    return G

  def create_final_path(self,nodes, h, j,s):   #creates final laplacian path matrix where j is the dominant node
    H, Hd = self.set_path_weights(nodes,h)
    G = self.env_matrix_creator(nodes,j,s)
    rate = self.rate_calc(H)
    A = self.laplace(G,H,rate)
    return A
    
  def generate_path_matrices(self,nodes, h, s):     #returns a matrix of path networks of size "nodes" and uniform edge weights "h"
    A = np.zeros((nodes, nodes, nodes))
    for i in range(nodes):
        A[i] = self.create_final_path(nodes, h, i,s)
    return A
    
  def create_final_matrix(self, nodes,h,w,j,s):   #creates final laplacian matrix for fully connected network
    H = self.set_edge_weights(nodes,h,w)
    G = self.env_matrix_creator(nodes,j,s)
    rate = self.rate_calc(H)
    A = self.laplace(G,H,rate)
    return A
    
  def generate_matrices(self,nodes, h, w, s):    #returns a matrix of fully connected networks with two edges weighted with "h-w"
    A = np.zeros((nodes, nodes, nodes))
    for i in range(nodes):
        A[i] = self.create_final_matrix(nodes, h, w, i,s)
    return A
    
  def create_final_smatrix(self, nodes,h, j,i,s):   #creates final laplacian matrix for sparse network
    H = self.set_edge_weights_sparse(nodes,h,i)
    G = self.env_matrix_creator(nodes,j,s)
    rate = self.rate_calc(H)
    A = self.laplace(G,H,rate)
    return A
    
  def generate_sparse_matrices(self, nodes, h, i, s): #returns a matrix of sparse networks with (100-i)% of possible edges drawn
    A = np.zeros((nodes, nodes, nodes))
    for l in range(nodes):
        A[l] = self.create_final_smatrix(nodes, h, l, i,s)
    return A
  
  def generate_environment_sequence(self,nodes):   #generate a sequence of non-repeating integers from 1,2..,nodes
    envs = np.zeros(1000, dtype=int)
    envs[0] = np.random.randint(1,nodes+1)
    for i in range(1,envs.size):
        envs[i] = np.random.randint(1,nodes+1)
        while(envs[i] == envs[i-1]):
            envs[i] = np.random.randint(1,nodes+1)
    return envs

  def generate_durations1(self,envs):  #set a duration for every associated random integer, total duration = TIME_T
    indiv_times = np.zeros(envs.size)
    for i in range(envs.size):
        indiv_times[i] = np.random.exponential(average_durations[(envs[i]-1)],1)[0]
    indiv_times = np.max([indiv_times, np.ones(indiv_times.shape)*0.01], axis=0)
    c_indiv = np.cumsum(indiv_times)
    k = np.argmax(c_indiv>TIME_T)         #cutoff point at which sum of durations exceeds TIME_T
    c_indiv = c_indiv[:k+1] 
    real_envs = envs[:k+1]          #curtailed sequence of environments which add up to TIME_T
    ind_times = indiv_times[:k+1]
    return real_envs, ind_times, c_indiv


  def manual_solver(self,nodes, initials, A, tshift, time):    #manually solves a system of ODEs using the Euler method with step-size= time/100
    t = np.linspace(tshift, tshift+ time, int(100*(time)))
    sol = np.zeros((nodes, t.size), dtype=np.float128)
    sol[:,0] = initials
    for i in range(1,t.size):
        sol[:,i] = sol[:,i-1] + (A @ sol[:,i-1]) * (t[i]-t[i-1])
    return sol, t, sol[:,t.size-1]        #returns solution vector, time vector and solution at final time point
    
  def deriv(self,y, t, A):         #system of ODEs defined 
    dNdt = A @ y
    return dNdt
    
  def solver(self,nodes,initials, A, tshift,time):    #function to solve system at every environment
    t = np.linspace(tshift, tshift+ time, int(100*(time)))            #time including current + sample from exp
    sol = odeint(self.deriv, initials, t, args=(A,))
    return sol, t, sol[t.size-1]
    
  def simulate(self,nodes, A, real_envs, indiv_times):        #solution till TIME_t using manual_solver
    init = np.zeros(nodes, dtype=np.float128)        #vector of initial conditions
    sol_final = np.ones((nodes,1), dtype=np.float128)        #stub for appending solution vector at each step
    t_final = np.ones(1)              #stub for appending time array of every environment 
    t_start=0     #variable to keep track of the relative starting point of every environmental switch   
    tracker = np.ones(1, dtype='int')
    for i in range(nodes):          #initialise population
        init[i]= 5e-60
    for i in range(real_envs.size):             #iterate through random array to decide which environment will occur 
        solution, time_array, init = self.manual_solver(PHENOTYPES,init, A[(real_envs[i]-1)], t_start, indiv_times[i])       #solver outputs solution array, corresponding time array and initial values for next step
        t_start = time_array[time_array.size-1]               #new starting time for next environment
        sol_final = np.append(sol_final, solution, axis=1)      #final solution vector
        t_final = np.append(t_final,time_array)                  #final time vector 
        temp = np.full(time_array.size, real_envs[i])
        tracker = np.append(tracker,temp)
    t_final = t_final[1:]
    tracker = tracker[1:]
    sol_final = sol_final[:,1:]
    return sol_final, t_final 
    
  def simulate_solver(self,nodes, A, real_envs, indiv_times):   #solution till TIME_T using odeint solver
    init = np.zeros(nodes, dtype=np.float64)        #vector of initial conditions
    sol_final = np.ones((1,nodes), dtype=np.float64)        #stub for appending solution vector at each step
    t_final = np.ones(1)              #stub for appending time array of every environment 
    t_start=0     #variable to keep track of the relative starting point of every environmental switch              
    t_total = 0
    for i in range(nodes):          #initialise population
        init[i]= 0.00005
    for i in range(real_envs.size):             #iterate through random array to decide which environment will occur 
        solution, time_array, init = self.solver(PHENOTYPES,init, A[(real_envs[i]-1)], t_start, indiv_times[i])       #solver outputs solution array, corresponding time array and initial values for next step
        t_start = time_array[time_array.size-1]               #new starting time for next environment
        sol_final = np.append(sol_final, solution, axis=0)      #final solution vector
        t_final = np.append(t_final,time_array)                  #final time vector
    sol_final = np.transpose(sol_final)
    return sol_final, t_final 


if __name__ == '__main__':
  #create json file for nodes init, (or PHENOTYPES), AVG, s, time, average_durations = np.full(PHENOTYPES, AVG)
  with open('parameters.json') as jsonFile:
        params = json.load(jsonFile)
  p = Phenotypes()
  A = p.generate_matrices(PHENOTYPES, 0.005, 0.001, 10)
  env = p.generate_environment_sequence(3)
  env_seq, indiv_times, total_times = p.generate_durations1(env)
  solf, tf = p.simulate_solver(PHENOTYPES, A, env_seq, indiv_times)
  np.savez('data.npz', population=solf, time=tf)

  #nx.draw(G)
