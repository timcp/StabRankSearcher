StabRankSearcher: code for finding (upper bounds to) the stabilizer rank of a quantum state
===========================================================================================

Heuristic algorithm for searching for (an upper bound to) the stabilizer rank of a quantum state, i.e. the minimal number of stabilizer states needed to write the quantum state as linear combination.
The algorithm is by [Bravyi, Smith and Smolin](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.021043).
We used this algorithm for finding the stabilizer rank of Dicke states, see our work

*LIMDD: A Decision Diagram for Simulation of Quantum Computing Including Stabilizer States*
Lieuwe Vinkhuijzen, Tim Coopmans, Vedran Dunjko, David Elkouss, Alfons Laarman


Installation
------------

Before installation, you might want to create a [virtual environment](https://docs.python.org/3/tutorial/venv.html).

The current version requires one to install [NetSquid](www.netsquid.org), for which one can freely register for non-commercial purposes (see [website](www.netsquid.org)).
After registration, add your credentials as local environment variables:

```
export NETSQUIDPYPI_USER=yourusername
export NETSQUIDPYPI_PWD=yourpassword
```

After that, run `make install`.
Then, add the current directory to your Pythonpath:

```
export PYTHONPATH=/path/to/stabranksearcher/
```

To check whether installation worked, run `make tests`.


Usage
-----

The package contains two algorithms for stabilizer search, referred to as `rank searchers`: one with repeated random trials and a random-walk algorithm (the one mentioned above).

A simple example is given in `examples/example.py`, which is a simple script for using the repeated-random searcher to search for the stabilizer rank of an arbitrary input state.

In our work, we applied the random-walk algorithm to Dicke states (equal superposition of computational-basis states with fixed Hamming weight), using the script `examples/dicke_state_analyzer.py`.





