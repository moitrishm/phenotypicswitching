'''
  The Phenotype class contains a list of functions
  to create a network of bacterial phenotypes, which can switch among each other and
  contribute to the bacterial colony's population growth. The colony experiences a sequence of environments,
  and a particular phenotype has the largest growth rate in every environment.


  The functions generate a sequence of growth environments and simulate the colony growth using 
  either a scipy ode solver or Euler's method. The population growth is governed by 
  a different coefficient matrix in every environment.
  Thus, the growth equation changes at every environmental switch.
'''

import networkx as nx
from scipy.integrate import odeint
import numpy as np

class Phenotypes(object): 

  def __init__(self, parameters):
        # parse parameters
        for key in parameters:
            setattr(self, key, parameters[key])
        self.average_durations = np.full(self.nodes, self.avg)
  
  def set_path_weights(self,h):     #create a path type network with #nodes = nodes 
    H = nx.path_graph(self.nodes, create_using=nx.Graph)
    H1 = nx.to_numpy_array(H)
    return H1*h

  def set_edge_weights(self,h, w):  #creates a fully connected network where two edges are weighted as "h-w"
    H = np.full((self.nodes,self.nodes),h)
    H[0][self.nodes-1] = h - w
    H[self.nodes-1][0] = h - w
    np.fill_diagonal(H,0)
    return H
  
  def set_edge_weights_sparse(self,h, i):      #random graph with (100*i)% of possible edges drawn
    H=nx.gnm_random_graph(self.nodes, i*self.nodes*(self.nodes-1), directed=True)    
    while(not(nx.is_strongly_connected(H))):
      H=nx.gnm_random_graph(self.nodes, i*self.nodes*(self.nodes-1), directed=True)    #ensure graph is strongly connected
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

  def env_matrix_creator(self, k):      #creates a diagonal matrix with phenotype growth rates
    G = np.zeros((self.nodes,self.nodes))   #growth rates of every phenotype
    temp = np.full(self.nodes, 0.1)    #set all growth rates = 0.1
    temp[k] += self.s/10             #only one of the growth rates is larger by s/10 in each environment
    np.fill_diagonal(G, temp)            
    return G

  def create_final_path(self,h, j):   #creates final laplacian path matrix where j is the dominant node
    H = self.set_path_weights(h)
    G = self.env_matrix_creator(j)
    rate = self.rate_calc(H)
    A = self.laplace(G,H,rate)
    return A
    
  def generate_path_matrices(self,h):     #returns a matrix of path networks of size "nodes" and uniform edge weights "h"
    A = np.zeros((self.nodes, self.nodes, self.nodes))
    for i in range(self.nodes):
        A[i] = self.create_final_path(h, i)
    return A
    
  def create_final_matrix(self,h,w,j):   #creates final laplacian matrix for fully connected network
    H = self.set_edge_weights(h,w)
    G = self.env_matrix_creator(j)
    rate = self.rate_calc(H)
    A = self.laplace(G,H,rate)
    return A
    
  def generate_fc_matrices(self,h, w):    #returns a matrix of fully connected networks with two edges weighted as "h-w"
    A = np.zeros((self.nodes, self.nodes, self.nodes))
    for i in range(self.nodes):
        A[i] = self.create_final_matrix(h, w, i)
    return A
    
  def create_final_smatrix(self,H, j,i):   #creates final laplacian matrix for sparse network
    G = self.env_matrix_creator(j)
    rate = self.rate_calc(H)
    A = self.laplace(G,H,rate)
    return A
    
  def generate_sparse_matrices(self, h, i): #returns a matrix of sparse networks with (100*i)% of possible edges drawn
    A = np.zeros((self.nodes, self.nodes, self.nodes))
    H = self.set_edge_weights_sparse(h,i)
    for l in range(self.nodes):
        A[l] = self.create_final_smatrix(H, l, i)
    return A
  
  def generate_environment_sequence(self):   #generate a sequence of non-repeating integers from 1,2..,nodes
    envs = np.zeros(1000, dtype=int)
    envs[0] = np.random.randint(1,self.nodes+1)
    for i in range(1,envs.size):
        envs[i] = np.random.randint(1,self.nodes+1)
        while(envs[i] == envs[i-1]):
            envs[i] = np.random.randint(1,self.nodes+1)
    return envs

  def generate_durations1(self,envs):  #set a duration associated with each random integer, total growth duration = time_t
    indiv_times = np.zeros(envs.size)
    for i in range(envs.size):
        indiv_times[i] = np.random.exponential(self.average_durations[(envs[i]-1)],1)[0]
    indiv_times = np.max([indiv_times, np.ones(indiv_times.shape)*0.01], axis=0)
    c_indiv = np.cumsum(indiv_times)
    k = np.argmax(c_indiv> self.time_t)         #cutoff point at which sum of durations exceeds TIME_T
    c_indiv = c_indiv[:k+1] 
    real_envs = envs[:k+1]          #curtailed sequence of environments which add up to TIME_T
    ind_times = indiv_times[:k+1]
    return real_envs, ind_times, c_indiv


  def manual_solver(self,initials, A, tshift, time):    #manually solves a system of ODEs using the Euler method with step-size= time/100
    t = np.linspace(tshift, tshift+ time, int(100*(time)))
    sol = np.zeros((self.nodes, t.size), dtype=np.float128)
    sol[:,0] = initials
    for i in range(1,t.size):
        sol[:,i] = sol[:,i-1] + (A @ sol[:,i-1]) * (t[i]-t[i-1])
    return sol, t, sol[:,t.size-1]        #returns solution vector, time vector and solution at final time point
    
  def deriv(self,y, t, A):         #system of ODEs defined 
    dNdt = A @ y
    return dNdt
    
  def solver(self, initials, A, tshift,time):    #solves a system of ODEs using odeint
    t = np.linspace(tshift, tshift+ time, int(100*(time)))            #time including current + duration from list 
    sol = odeint(self.deriv, initials, t, args=(A,))
    return sol, t, sol[t.size-1]
    
  def manual_simulate(self, A, real_envs, indiv_times):        #solution till time_t using manual_solver
    init = np.zeros(self.nodes, dtype=np.float128)        #vector of initial conditions
    sol_final = np.ones((self.nodes,1), dtype=np.float128)        #stub for appending solution vector at each step
    t_final = np.ones(1)              #stub for appending time array of every environment 
    t_start=0     #variable to keep track of the relative starting point of every environmental switch   
    tracker = np.ones(1, dtype='int')
    for i in range(self.nodes):          #initialise population
        init[i]= self.init_n
    for i in range(real_envs.size):             #iterate through random array to decide which environment will occur 
        solution, time_array, init = self.manual_solver(init, A[(real_envs[i]-1)], t_start, indiv_times[i])       #solver outputs solution array, corresponding time array and initial values for next step
        t_start = time_array[time_array.size-1]               #new starting time for next environment
        sol_final = np.append(sol_final, solution, axis=1)      #final solution vector
        t_final = np.append(t_final,time_array)                  #final time vector 
        temp = np.full(time_array.size, real_envs[i])
        tracker = np.append(tracker,temp)
    t_final = t_final[1:]
    tracker = tracker[1:]
    sol_final = sol_final[:,1:]
    return sol_final, t_final 
    
  def solver_simulate(self, A, real_envs, indiv_times):   #solution till time_t using odeint solver
    init = np.zeros(self.nodes, dtype=np.float64)        #vector of initial conditions
    sol_final = np.ones((1,self.nodes), dtype=np.float64)        #stub for appending solution vector at each step
    t_final = np.ones(1)              #stub for appending time array of every environment 
    t_start=0     #variable to keep track of the relative starting point of every environmental switch              
    t_total = 0
    for i in range(self.nodes):          #initialise population
        init[i]= self.init_n
    for i in range(real_envs.size):             #iterate through random array to decide which environment will occur 
        solution, time_array, init = self.solver(init, A[(real_envs[i]-1)], t_start, indiv_times[i])       #solver outputs solution array, corresponding time array and initial values for next step
        t_start = time_array[time_array.size-1]               #new starting time for next environment
        sol_final = np.append(sol_final, solution, axis=0)      #final solution vector
        t_final = np.append(t_final,time_array)                  #final time vector
    sol_final = np.transpose(sol_final)
    return sol_final, t_final 