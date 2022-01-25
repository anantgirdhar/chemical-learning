"""Parse RMG Kinetics Library

Extract reactions and parameters from an RMG kinetics library. This script
converts a kinetics library from RMG format into a pickle file that can later
be read and compiled when aggregating a lot of reactions.

Make sure that the code contained in the library is not malicious. This module
executes the code as python code. So any malicious code will be run without any
error checking.
"""
import importlib
import os
import pickle

# Create the datastructure that will store the reaction data
entries = []
species = []

# Define some functions that can be used to parse the library quickly

def entry(index, label, degeneracy=None, kinetics=None, duplicate=False, reversible=True, elementary_high_p=False, longDesc=None, shortDesc=None, allow_max_rate_violation=None, allow_pdep_route=None, reference=None, referenceType=None):
    """Each reaction is expressed as an "entry" in the library"""
    entries.append({
            'rxn_string': label,
            'kinetics_type': kinetics[0],
            'kinetics': kinetics[1],
            'degeneracy': degeneracy,
            'duplicate': duplicate,
            'elementary_high_p': elementary_high_p,
            })

def Article(*args, **kwargs):
    return None

def Arrhenius(A, n, Ea, T0, Tmin=None, Tmax=None, comment=None, Pmin=None, Pmax=None):
    return 'Arrhenius', {'A': A, 'n': n, 'Ea': Ea, 'T0': T0, 'Tmin': Tmin, 'Tmax': Tmax, 'Pmin': Pmin, 'Pmax': Pmax}

def MultiArrhenius(arrhenius):
    return 'MultiArrhenius', [a[1] for a in arrhenius]

def ThirdBody(arrheniusLow, efficiencies=None, comment=None):
    return 'ThirdBody', {
            'arrheniusLow': arrheniusLow[1],
            'efficiencies': efficiencies,
            }

def Lindemann(arrheniusLow, arrheniusHigh, efficiencies=None, comment=None):
    return 'Lindemann', {
            'arrheniusLow': arrheniusLow,
            'arrheniusHigh': arrheniusHigh,
            'efficiencies': efficiencies,
            }

def Troe(arrheniusLow, arrheniusHigh, efficiencies, alpha=None, T1=None, T3=None, T2=None, comment=None):
    return 'Troe', {
            'arrheniusLow': arrheniusLow,
            'arrheniusHigh': arrheniusHigh,
            'alpha': alpha,
            'T1': T1,
            'T2': T2,
            'T3': T3,
            'efficiencies': efficiencies,
            }

def PDepArrhenius(pressures, arrhenius, comment=None):
    return 'PDepArrhenius', {
            'pressures': pressures,
            'arrhenius': [a[1] for a in arrhenius],
            }

def MultiPDepArrhenius(arrhenius, comment=None):
    return 'MultiPDepArrhenius', {
            'arrhenius': [p[1] for p in arrhenius],
            }

def Chebyshev(coeffs, kunits, Tmin, Tmax, Pmin, Pmax):
    return 'Chebyshev', {
            'coeffs': coeffs,
            'kunits': kunits,
            'Tmin': Tmin,
            'Tmax': Tmax,
            'Pmin': Pmin,
            'Pmax': Pmax,
            }

# Do the parsing

def load_species(dictionary_file):
    """Read an RMG dictionary file and load species information"""
    global species
    species = {}
    with open(dictionary_file, 'r') as f:
        start_new_species = True  # indicates the start of a new species
        for line in f:
            line = line.strip()
            if start_new_species is True:
                name = line
                multiplicity = None
                graph_lines = []
                start_new_species = False
            elif line == "":
                # this is the end of the species
                start_new_species = True
                species[name] = {
                        'multiplicity': multiplicity,
                        'graph': graph_lines,
                        }
                continue
            elif line.startswith("multiplicity"):
                # Add the multiplicity
                multiplicity = int(line.split(" ")[1])
                continue
            else:
                # This is part of the species
                # Check that the line number matches the next one in the series
                line_num = int(line.split(' ')[0])
                if line_num != len(graph_lines) + 1:
                    # Something went wrong
                    raise ValueError(f'Incorrect line number for species {name}')
                graph_lines.append(line)

def parse(library_path):
    """Parse the RMG kinetics library

    The input, library_path, is the path to the folder that contains the
    reaction.py file for the library.
    """
    global entries
    entries = []
    if not os.path.isdir(library_path):
        raise ValueError(f'{library_path} is not a valid directory')
    if not os.path.isfile(os.path.join(library_path, 'reactions.py')):
        raise ValueError(f'reactions.py not found in {library_path}')
    print(f'\nLibrary: {library_path}')
    library_name = os.path.basename(library_path)
    dictionary_file_path = os.path.join(library_path, 'dictionary.txt')
    load_species(dictionary_file_path)
    reactions_file_path = os.path.join(library_path, 'reactions.py')
    with open(reactions_file_path, 'r') as f:
        file_contents = f.read()
    exec(file_contents)
    # Make sure the output folder exists
    if not os.path.exists('./rmg_compiled_pickles'):
        # Create the directory
        os.mkdir('./rmg_compiled_pickles')
    elif not os.path.isdir('./rmg_compiled_pickles'):
        raise ValueError('./rmg_compiled_pickles is not a valid directory')
    # Pickle the data
    with open(f'rmg_compiled_pickles/{library_name}.p', 'wb') as f:
        pickle.dump([entries, species], f)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print(f'Usage: python {sys.argv[0]} kinetics_library_directory')
        sys.exit(0)
    parse(sys.argv[1])
