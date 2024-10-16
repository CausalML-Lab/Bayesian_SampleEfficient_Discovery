## Sample Efficient Bayesian Learning of Causal Graphs from Interventions
Environment:
Python 3.11, Julia 1.10
Packages are in environment.yml

Python Libraries:
Graphical_Models, PyAgrum, CausalDag, GraphTheory, Causal_learn

Julia Libraries:
LightGraphs, GraphIO, LinkedLists

The CliquePikcing algorithm is from:
https://github.com/mwien/CliquePicking.git

Run our algorithm:

ex: `python run_sample_efficient.py --n 5 --den 1 --n_dag 50 --n_sample 100000`

Run baselines:

ex: `python run_baselines.py`

### References
Wienöbst, M., Bannach, M., & Liśkiewicz, M. (2023). Polynomial-Time Algorithms for Counting and Sampling Markov Equivalent DAGs with Applications. Journal of Machine Learning Research, 24(213), 1-45.