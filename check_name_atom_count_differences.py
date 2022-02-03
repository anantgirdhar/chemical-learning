import os
import pickle
import sys

species = {}

os.chdir('./rmg_compiled_pickles/')
for fn in os.listdir():
    with open(fn, 'rb') as f:
        entries, sp = pickle.load(f)
    for k, v in sp.items():
        if k not in species:
            # Just add it to the list of species
            species[k] = {
                    'atom_counts': v['atom_counts'],
                    'files': [fn, ],
                    }
        elif species[k]['atom_counts'] == v['atom_counts']:
            # It's the same atom counts and the same name - probably fine
            species[k]['files'].append(fn)
        else:
            # It's different - print stuff
            print(f'Mismatch found for species {k}')
            print('Found so far:')
            print(species[k])
            print(f'Mismatch found in file {fn}:')
            print(v['atom_counts'])
