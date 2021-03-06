# Hidden Markov Model

C++ Hidden Markov Model implementation with Python interfacing (using SWIG) developed by [Joobz](https://github.com/Joobz) and [dmg99](https://github.com/dmg99). Most features are currently in development, but the core parts are functional. Updates are not plan, we will work on it when we have some time. The implementation is mostly based in the formulas and notations in [Wikipedia](https://en.wikipedia.org/wiki/Hidden_Markov_model).

Feel free to coment on bugs or possible improvements. We hope it proves useful.

## Current Features

Currently we have implemented the following main classes:

+ Hidden Markov Model (HMM): Standard HMM with gaussian distributions and any possible amount of channels.

+ Hidden Hidden Markov Model (HiddenHMM): Standard HMM with with an extra layer of hidden states. Appart from the usual hypothesis that observations arebeing generated by some hidden states, the transitin matrix of these latter depends on some other hidden states (to which we refer as 'hidden hidden states').

+ Adaptive HHMM: Same as before but here the Hidden states' transition matrices also depend on the current standard hidden states.

## Features to include

Some ideas to add in future versions are:

+ Better python interfacing: assertions and parameters.

+ Add other probability distribution functions for the observation.

+ Multidimensional variables support.

+ Examples!
  
## Basic usage

If you just one to use it, copy the files from dist to your working directory and you should be able to 'import HMM' from python.

Be carefule because currently multi-dimensional inputs must be vectors! Use the HMM.numpy2vec() and HMM.numpy2vec_int() from numpy arrays to transform your variables to C++ vectors. Make sure the data type from the numpy arrays is the right one.

When working with notebooks, sometimes the kernel might unexpectedly stop, this is due to violated assertions in C++, check the command line for any possible error massages.

Finally, you can also directly declare C++ vectors and matrices in python with the corresponding template names (found in src/HMM.i).

### Compilation

If for any reason you need to use the C++ code itself, make modifications or just compile it, check the Makefile a the main directory, in theory you should be able to run the make command with the appropiate modifications on the python and numpy directories. You should also have SWIG installed.
