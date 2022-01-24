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

# Define some functions that can be used to parse the library quickly

def entry(index, label, degeneracy, kinetics, duplicate=False, reversible=True, elementary_high_p=False, longDesc=None, shortDesc=None):
    """Each reaction is expressed as an "entry" in the library"""
    entries.append({
            'rxn_string': label,
            'kinetics_type': kinetics[0],
            'kinetics': kinetics[1],
            'degeneracy': degeneracy,
            'duplicate': duplicate,
            'elementary_high_p': elementary_high_p,
            })

def Arrhenius(A, n, Ea, T0, Tmin=None, Tmax=None, comment=None):
    return 'Arrhenius', {'A': A, 'n': n, 'Ea': Ea, 'T0': T0, 'Tmin': Tmin, 'Tmax': Tmax}

def MultiArrhenius(arrhenius):
    return 'MultiArrhenius', [a[1] for a in arrhenius]

def ThirdBody(arrheniusLow, efficiencies):
    return 'ThirdBody', {
            'arrheniusLow': arrheniusLow[1],
            'efficiencies': efficiencies,
            }

def Lindemann(arrheniusLow, arrheniusHigh, efficiencies, comment=None):
    return 'Lindemann', {
            'arrheniusLow': arrheniusLow,
            'arrheniusHigh': arrheniusHigh,
            'efficiencies': efficiencies,
            }

def Troe(arrheniusLow, arrheniusHigh, efficiencies, alpha=None, T1=None, T3=None, T2=None):
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
    print(f'Library: {library_path}')
    library_name = os.path.basename(library_path)
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
        pickle.dump(entries, f)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print(f'Usage: python {sys.argv[0]} kinetics_library_directory')
        sys.exit(0)
    parse(sys.argv[1])
