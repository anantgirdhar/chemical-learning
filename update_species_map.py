"""Create a bijection between species and integers

To unique identify species in ML models, this script assigns each specie a
unique integer by storing a list of unqiue species. The index of a specie in
the list is the unique integer it is assigned.

To run this script, pass in a reactions data file where the first four columns
are the reactants and the products.
"""

import pickle
import sys
import os

import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    raise ValueError(f'Usage: python create_species_map.py <reactions_data_file.csv>')
filename = sys.argv[1]
mapfile = 'species_map.p'

# Get a list of unique species
df = pd.read_csv(filename)
df_species = df[['reactant1', 'reactant2', 'product1', 'product2']]
species = pd.concat([df_species[col] for col in df_species.columns]).unique()

# Load the old species_map if it exists
if os.path.isfile(mapfile):
    with open(mapfile, 'rb') as f:
        species_list = pickle.load(f)
else:
    species_list = []

for s in species:
    if s not in species_list:
        species_list.append(s)

# Dump the list back to the file
with open(mapfile, 'wb') as f:
    pickle.dump(species_list, f)
