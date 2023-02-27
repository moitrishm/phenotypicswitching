Python code to simulate the population growth of a bacterial colony. The colony consists of several phenotypes, and individual cells of one phenotype can switch to another phenotype. The colony grows over a certain duration of time, and the growth environment changes at certain points of time. In a given environment, one phenotype has the highest growth rate and every environment has an associated "fittest phenotype" which drives the colony's growth in that environment. The growth is described by a set of ODEs, where every state variable is the population of one phenotype in the colony. The inter-phenotypic switching can be described by a network, and each network has an associated adjacency matrix which also forms the coefficient matrix in the system of ODEs. For more information about this project, please visit moitrishm.github.io/portfolio/portfolio-2/


Necessary python packages: numpy, networkx, json and scipy


1) The phenotypes.py contains the Phenotype() class, which contains the various methods to simulate the growth of the bacterial colony.

2) The run_phenotypes.py file firstly generates the coefficient matrices which describes the set of ODEs. It then generates the sequence of environments over which this system of ODEs evolves (where each environment lasts for a certain duration of time). Finally, it solves the system of ODEs and stores the matrices, environment sequence and associated durations, and the solution array which contains the population of each phenotype at every point of time. 

3) The parameters.json file contains the relevant parameters, which can set the number of nodes in the networks ("nodes"), total duration of the simulation ("time_t") and average duration of every environment ("avg"). The fittest phenotype in every environment has a growth rate that is greater than the growth rates of the other phenotypes by a factor of "s".


The adjacency matrices which are generated can describe path networks, fully connected networks or sparse networks. These networks have directed edges.


>In a fully connected network, there is an edge from every node to every other node. To generate the matrices describing the fully connected networks, use the generate_fc_matrices method which takes two arguments. The first argument specifies the "edge_weights" of all the edges in the network, and the second argument ("reduction") is the amount by which two edge weights are reduced. The system of ODEs corresponding to the fully connected networks with "reduction=0" is a set of coupled ODEs where every equation is dependent on all the state variables.

>To generate the path network matrices, use generate_path_networks which only takes the edge weights as an argument.

>To generate the sparse network matrices, use generate_sparse_networks which takes the network edge weights and the proportion of edges as arguments. The "proportion" argument is a number between 0 and 1, and a sparse network with "proportion = 1" generates a fully connected network.

>After generating an instance of the Phenotype() class, generate the desired coefficient matrix using one of the above methods.

>Then, generate the sequence of environments and durations, and obtain the solution vector. The solution can be obtained using a manual solver (using manual_simulate) or an ODE solver (using solver_simulate) from the scipy package. Both methods take the same arguments: the coefficient matrices describing the ODEs, the sequence of environments and the individual duration of every environment. 




Running run_phenotypes.py gives the matrices, solution vector, environment sequence and individual durations in a data.npz file.


