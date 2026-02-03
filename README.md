# EveSyn: Enhancing Utility of Differentially Private Synthetic Data over Dynamic Database with Efficient Updates

## Introduction

This code base contains two implementation examples of EveSyn, implemented with two state-of-the-art PGM-based data synthesis mechanisms based on graph model as the OriginalSyn function, which are the MWEM+PGM and AIM.

For more details on Private-PGM, please visit [Private-PGM](https://github.com/ryan112358/private-pgm).

These files also have two additional dependencies: [Ektelo](https://github.com/ektelo/ektelo) and [autograd](https://github.com/HIPS/autograd).

## File structure

* mechanisms - Contains EveSyn constructions which are modified for the original data synthesis of EveSyn.
* evmechanisms - Contains EveSyn constructions for the updated data synthesis and some dataset utility tools.
* data - Contains datasets, selected cliques produced in the original data synthesis, and preferred cliques.
* src - Contains some dependencies of PGM-based mechanisms.
* UDF - Contains example UDFs. 
* EveSyn.py - Gives an example of how to organize an EveSyn experiment.

## Usage

1. Before we start, if you are only testing non-UDF parts, you could remove
the ```pycopg2``` and ```sqlalchemy``` in the ```requirements.txt```. Moreover, when you are testing the UDF part in a Linux system like ```Ubuntu```, 
you should check whether ```python-psycopg2``` and ```libpq-dev``` are installed, or you should use ```apt``` to get them before you solve the requirements.

2. Solve the dependencies with ```requirements.txt```. Note that we only support Python 3. 

```
$ pip install -r requirements.txt
```
3. Export the ```src``` file to path. For example, in Windows, you may use:
```
$Env:PYTHONPATH += ";X:\EveSyn\src"
```
4. Run the mechanism in ```mechanisms``` for original data synthesis (OriginalSyn), then run the corresponding mechanism under ```evmechanisms```.
The EveSyn.py also gives an example of how to organize a one-click experiment with ```config.json```.

## UDF usage
See ```README.md``` in ```/UDF```.
