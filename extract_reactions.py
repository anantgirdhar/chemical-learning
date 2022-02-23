"""Extract reactions from compiled pickle files

The pickle files contain reaction and kinetics data. This script helps
extract a subset of those reactions for analysis.
"""
import csv
import glob
import pickle

import matplotlib.pyplot as plt
import numpy as np

from cgt import Specie, ChemicalVF2
from reaction import Reaction

def rename_species(old_species, new_species):
    """Map new_species to old_species based on their structure

    When comparing two different chemical libraries, the identifiers used to
    represent species might be different even if the species they represent are
    the same. This function creates a mapping so that they can be mapped when
    combining two different libraries.
    """
    mapping = {}
    for new_name, new_sp in new_species.items():
        for old_name, old_sp in old_species.items():
            GM = ChemicalVF2(new_sp.G, old_sp.G)
            if GM.is_isomorphic():
                # The species are the same
                # Rename the new one to the old one
                # Add it to the mapping if the names don't match
                if new_name != old_name:
                    mapping[new_name] = old_name
                break  # move on to the next new_specie
        else:
            # This specie does not exist in the old database
            # Indicate that with a None
            mapping[new_name] = None
    return mapping

def extract_arrhenius(pickle_files, output_pickle):
    """Extract all Arrhenius reactions from pickle files"""
    subset = []
    species = {}
    # Load data from the pickles
    for pfile in pickle_files:
        with open(pfile, 'rb') as f:
            new_entries, new_species = pickle.load(f)
        # Create a name mapping for consistency with the database
        name_mapping = rename_species(species, new_species)
        # The name mapping contains all species that have a name mismatch or
        # that are not in the overall database. Add in the new species
        # (indicated by None's).
        for name, sp in new_species.items():
            if name in name_mapping and name_mapping[name] == None:
                species[name] = sp
        # Now add all the reactions
        for e in new_entries:
            if e.kinetics_type == 'Arrhenius':
                # Rename the species names as appropriate to match the database
                for sp in e.reactants:
                    if sp.name in name_mapping:
                        rename_to = name_mapping[sp.name]
                        if rename_to:
                            # This means that the species isn't a new specie
                            sp.name = rename_to
                for sp in e.products:
                    if sp.name in name_mapping:
                        rename_to = name_mapping[sp.name]
                        if rename_to:
                            # This means that the species isn't a new specie
                            sp.name = rename_to
                # Now that the species names are consistent, add the reaction
                # to the database
                subset.append(e)
    with open(output_pickle, 'wb') as f:
        pickle.dump([subset, species], f)

def arrhenius_to_csv(arrhenius_pickle, output_csv):
    """Write a CSV file containing the Arrhenius kinetics parameters"""
    # Load data from the pickles
    with open(arrhenius_pickle, 'rb') as f:
        entries, species = pickle.load(f)
    # Create the header
    header = ['reactant1', 'reactant2', 'product1', 'product2', 'A', 'b', 'Ea', 'duplicate', 'degeneracy', 'elementary_high_p']
    atoms = ['C', 'H', 'O', 'N', 'Ar', 'He', 'N', 'Cl', 'S', 'I', 'Si']
    atom_count_headers_prefix = ['r1', 'r2', 'p1', 'p2']
    atom_count_headers_suffix = ['C', 'H', 'O', 'N', 'Ar', 'He', 'Cl', 'S', 'I', 'Si']
    for sp in atom_count_headers_prefix:
        header.extend([sp + atom for atom in atom_count_headers_suffix])
    # Now create the rows
    rows = [header, ]
    for e in entries:
        # Get the species information from the reaction
        if len(e.reactants) != 2 or len(e.products) != 2:
            print(f'Incorrect number of reactants/products - skipping {e}')
            continue
        reac1 = e.reactants[0].name
        reac2 = e.reactants[1].name
        prod1 = e.products[0].name
        prod2 = e.products[1].name
        # Get the atom counts
        overall_atom_counts = {}
        # First initialize all atom counts to 0
        for sp in atom_count_headers_prefix:
            for atom in atom_count_headers_suffix:
                overall_atom_counts[sp + atom] = 0
        # Next get the values from the various species objects
        for sp in atom_count_headers_prefix:
            for atom, count in e.reactants[0].atom_counts.items():
                overall_atom_counts[sp + atom] = count
        # Make sure that the units of everything else are good
        # Get pre exponential factor in cm, mol, s units
        # Get activation energy in cal/mol units
        A = e.kinetics['A']
        if A[1] in ['cm^3/(mol*s)']:
            A = A[0]
        elif A[1] in ['cm^3/(molecule*s)']:
            print(f'Bad A units - skipping reaction {e}')
            continue
        else:
            raise ValueError(f'Units of A ({A[1]}) unknown')
        n = e.kinetics['n']
        Ea = e.kinetics['Ea']
        if Ea[1] == 'cal/mol':
            Ea = Ea[0]
        elif Ea[1] == 'kcal/mol':
            Ea = Ea[0] * 1000.
        elif Ea[1] == 'kJ/mol':
            Ea = Ea[0] * 238.85
        elif Ea[1] == 'J/mol':
            Ea = Ea[0] * 0.23885
        else:
            raise ValueError(f'Units of Ea ({Ea[1]}) unknown')
        degeneracy = e.degeneracy
        duplicate = e.duplicate
        elementary_high_p = e.elementary_high_p
        # Put everything into a list
        row = [
                reac1, reac2, prod1, prod2,
                A, n, Ea,
                duplicate, degeneracy, elementary_high_p,
                ]
        for sp in atom_count_headers_prefix:
            row.extend([overall_atom_counts[sp + atom] for atom in atom_count_headers_suffix])
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
    # get_statistics(output_csv)

if __name__ == '__main__':
    import sys
    directory_to_source_pickles = sys.argv[1]
    output_name = sys.argv[2]
    main(directory_to_source_pickles, output_name)
