"""Extract reactions from compiled pickle files

The pickle files contain reaction and kinetics data. This script helps
extract a subset of those reactions for analysis.
"""
import csv
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

def extract_arrhenius(pickle_files, output_pickle):
    """Extract all Arrhenius reactions from pickle files"""
    subset = []
    # Load data from the pickles
    for pfile in pickle_files:
        with open(pfile, 'rb') as f:
            entries, species = pickle.load(f)
        for e in entries:
            if e['kinetics_type'] == 'Arrhenius':
                subset.append(e)
    with open(output_pickle, 'wb') as f:
        pickle.dump(subset, f)

def arrhenius_to_csv(arrhenius_pickle, output_csv):
    """Write a CSV file containing the Arrhenius kinetics parameters"""
    # Load data from the pickles
    with open(arrhenius_pickle, 'rb') as f:
        entries = pickle.load(f)
    # Create the header
    header = ['reactant1', 'reactant2', 'product1', 'product2', 'A', 'b', 'Ea', 'duplicate', 'degeneracy', 'elementary_high_p']
    rows = [header, ]
    for e in entries:
        # Split the reaction into the different species
        r = e['rxn_string']
        try:
            reactants, products = r.split('=')
            # Split and clean the reactants
            reac1, reac2 = reactants.split('+')
            reac1 = reac1.strip()
            reac2 = reac2.split('<')[0].strip()
            # Split and clean the products
            prod1, prod2 = products.split('+')
            prod1 = prod1.split('>')[-1].strip()
            prod2 = prod2.strip()
        except ValueError:
            print(f'Something went wrong - skipping reaction {r}')
            continue
        # Make sure that the units of everything else are good
        # Get pre exponential factor in cm, mol, s units
        # Get activation energy in cal/mol units
        A = e['kinetics']['A']
        if A[1] in ['cm^3/(mol*s)']:
            A = A[0]
        elif A[1] in ['cm^3/(molecule*s)']:
            print(r'Bad A units - skipping reaction {r}')
            continue
        else:
            raise ValueError(f'Units of A ({A[1]}) unknown')
        n = e['kinetics']['n']
        Ea = e['kinetics']['Ea']
        if Ea[1] == 'cal/mol':
            Ea = Ea[0]
        elif Ea[1] == 'kcal/mol':
            Ea = Ea[0] * 1000.
        elif Ea[1] == 'kJ/mol':
            Ea = Ea[0] * 238.85
        else:
            raise ValueError(f'Units of Ea ({Ea[1]}) unknown')
        degeneracy = e['degeneracy']
        duplicate = e['duplicate']
        elementary_high_p = e['elementary_high_p']
        # Put everything into a list
        row = [reac1, reac2, prod1, prod2, A, n, Ea, duplicate, degeneracy, elementary_high_p]
        # Append to the rows
        rows.append(row)
    # Write the rows to a csv file
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(rows)

def get_statistics(csv_name):
    rows = []
    species = []
    num_reactions = 0
    A = []
    n = []
    Ea = []
    duplicate = []
    degeneracy = []
    elemetary_high_p = []
    with open(csv_name, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # skip the header
        for row in csvreader:
            rows.append(row)
            species.extend(row[0:4])
            num_reactions += 1
            A.append(float(row[4]))
            n.append(float(row[5]))
            Ea.append(float(row[6]))
            duplicate.append(row[7])
            degeneracy.append(row[8])
            elemetary_high_p.append(row[9])
    species = set(species)
    print(f'{num_reactions = }')
    print(f'{len(species) = }')
    print(f'min(A) = {np.min(A)}')
    print(f'max(A) = {np.max(A)}')
    print(f'min(n) = {np.min(n)}')
    print(f'max(n) = {np.max(n)}')
    print(f'min(Ea) = {np.min(Ea)}')
    print(f'max(Ea) = {np.max(Ea)}')
    plt.figure()
    plt.subplot(131)
    plt.plot(sorted(set(A)))
    plt.title('A')
    plt.subplot(132)
    plt.plot(sorted(set(n)))
    plt.title('n')
    plt.subplot(133)
    plt.plot(sorted(set(Ea)))
    plt.title('Ea')
    plt.show()

def main(directory_to_source_pickles, output_name):
    pickle_files = glob.glob(f'{directory_to_source_pickles}/*.p')
    output_pickle = f'{output_name}.p'
    output_csv = f'{output_name}.csv'
    extract_arrhenius(pickle_files, output_pickle)
    arrhenius_to_csv(output_pickle, output_csv)
    get_statistics(output_csv)

if __name__ == '__main__':
    import sys
    directory_to_source_pickles = sys.argv[1]
    output_name = sys.argv[2]
    main(directory_to_source_pickles, output_name)
