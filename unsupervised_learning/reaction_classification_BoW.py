import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import cantera as ct


def get_species_per_reaction(mechanism):
    gas = ct.Solution(mechanism)
    species_list = [sp.name for sp in gas.species()]
    # Get a list of the stoichiometric coefficients for each reaction
    # Each row is a reaction
    # Each column is a species
    # Don't care about whether or not the species are reactants or products
    # So there are no negative numbers in this sparse matrix
    reactions = np.zeros((gas.n_reactions, gas.n_species))
    for rxnidx, r in enumerate(gas.reactions()):
        for sp, coeff in r.reactants.items():
            spidx = species_list.index(sp)
            reactions[rxnidx, spidx] = coeff
        for sp, coeff in r.products.items():
            spidx = species_list.index(sp)
            reactions[rxnidx, spidx] = coeff
    return reactions

def get_atoms_per_reaction(mechanism):
    gas = ct.Solution(mechanism)
    elements = gas.element_names
    n_elements = gas.n_elements
    # Get a list of the compositions of species in each reaction
    # Each row is a reaction
    # Each column is an element in a species
    # There are four copies of each element. Most reactions have four species
    # and so we'll store the atom counts for every species. For reactions that
    # don't have a second reactant / product, we'll just zero the atom counts.
    # Don't care about whether or not the species are reactants or products
    # So there are no negative numbers in this sparse matrix
    # Also don't care about the stoichiometric coefficients for each specie
    reactions = np.zeros((gas.n_reactions, n_elements*4))
    for rxnidx, r in enumerate(gas.reactions()):
        for spidx, sp in enumerate(r.reactants.keys()):
            for elidx, e in enumerate(elements):
                reactions[rxnidx, spidx+elidx] = gas.n_atoms(sp, e)
            # if r.reactants[sp] > 1:
            #     if len(r.reactants) == 1:
            #         for elidx, e in enumerate(elements):
            #             reactions[rxnidx, spidx+1+elidx] = gas.n_atoms(sp, e)
            #     else:
            #         raise NotImplementedError(f'Reaction {r} at index {rxnidx}: don\'t know what to do.')
        for spidx, sp in enumerate(r.products.keys()):
            for elidx, e in enumerate(elements):
                reactions[rxnidx, 2*n_elements + spidx+elidx] = gas.n_atoms(sp, e)
            # if r.products[sp] > 1:
            #     if len(r.products) == 1:
            #         for elidx, e in enmerate(elements):
            #             reactions[rxnidx, spidx+1+elidx] = gas.n_atoms(sp, e)
            #     else:
            #         raise NotImplementedError(f'Reaction {r} at index {rxnidx}: don\'t know what to do.')
    return reactions

def run_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans

def plot_detailed(gas, labels):
    for label in sorted(set(labels)):
        print(f'Label = {label}')
        for reaction in np.array(gas.reactions())[labels == label]:
            print(reaction)
        print()

def run_elbow_method(X, min_clusters=1, max_clusters=10):
    distortions = []
    inertias = []
    cluster_range = list(range(min_clusters, max_clusters+1))
    for n_clusters in cluster_range:
        print(f'{n_clusters = }')
        kmeans = run_kmeans(X, n_clusters)
        distortions.append(sum(np.min(cdist(
            X,
            kmeans.cluster_centers_,
            'euclidean',
            ), axis=1)) / X.shape[0])
        inertias.append(kmeans.inertia_)
        cluster_numbers = kmeans.labels_
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(cluster_range, distortions, 'bo-', label='Distortion')
    ax2.plot(cluster_range, inertias, 'rx-', label='Inertia')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Distortion (blue)')
    ax2.set_ylabel('Inertia (red)')
    plt.show()

def main(mechanism='gri30.yaml'):
    features = get_atoms_per_reaction(mechanism)
    run_elbow_method(features, max_clusters=20)
    return features
