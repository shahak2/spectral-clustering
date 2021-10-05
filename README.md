# spectral-clustering

**1.Introduction**

This was a project for "Software Project" class in Tel-Aviv university.
In this project we(myself and a partner) implemented a version of the spectral clustering algorithm. It has two interfaces:
•	A C program interface.
•	A Python interface, using C-API for its calculations.

**2.Before using the Python interface:**

Make sure all the files are in the same folder. Run the "build.sh" file via the console, i.e.:

./build.sh

**Remark:** The Python interface works only in Linux environment.

**3. Using the project:**

Python interface – run the command:

./python3 spkmeans.py *K* *CMD* *FILE*

C interface – run the command:

./spkmeans *K* *CMD* *FILE*
Where you substitute:
1.	*K* - Number of required clusters. Alternatively, you may enter (K = )0 for the program to calculate the ideal number of clusters using the eigengap heuristic. 
2.	 *CMD* - can be substitute by:
2.1.	wam – outputs the Weighted Adjacency Matrix of the given datapoints.
2.2.	ddg – outputs the Diagonal Degree Matrix of the given datapoints.
2.3.	lnorm – outputs the Normalized Graph Laplacian Matrix(NGLM) of the given datapoints.
2.4.	spk – Calculates the (NGLM) of the given datapoints. Outputs the final centroids from the K-means algorithm.
2.4.1.	C interface: Initial centroids are the first K datapoints.
2.4.2.	Python Interface: The initial centroids are chosen by the K-means++ algorithm, using the Jacobi matrix.
2.5.	jacobi - outputs the eigenvalues followed by the eigenvectors of the given datapoints.
3.	*FILE* - the input file in txt or csv format. Each line represents a datapoint.

Remark: While wam, ddg, lnorm and spk commands use the calculations of previous commands on the given datapoints, jacobi commands performs its calculations directly on the given datapoints. This was an instruction. For example, lnorm command first calculates wam on the datapoints, then calculates DDG from the result and lastly performs its calculations.




