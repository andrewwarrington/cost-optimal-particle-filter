# Cost Optimal Particle Filter
We present demonstrative tools for exploring the themes described in Warrington and Dhir. "Generalising Cost-Optimal Particle Filtering." arXiv preprint arXiv:1805.00890 (2018).
This repository is very much still in development, and we welcome feedback! 

# Structure 
We include all the required .py files for running the simulations displayed in the above paper. 
Since these are all relatively simple scripts, with relatively limited requirements and dependancies, we suggest just running the scripts in a Python IDE of your choice. 
We also provide Docker files such that the code can be run automatically inside a Docker container. 
Although this approach is currently overkill, future incarnations of the code may have more complex dependencies and hence the portability of Docker is key (and because Docker is awesome and everyone should Dockerise their code for ever more!).

We are currently implementing Jupyter notebook versions of the code, watch this space.

# Regular sampling problem
The first example presented is the regular sampling problem, that is to say, computing the optimal number of equispaced samples to draw to optimally tradeoff localisation accuracy and expenditure.

Code to come soon.

# One sample problem
The second example, termed the one sample problem, is the optimal placement of a single sample, given a state dependent noise model.
To run this experiment, run the script ```oneSampleProblem.py```. 
All parameters are set towards the top of the script, and all outputs are dumped into a directory for subsequent postprocessing.
