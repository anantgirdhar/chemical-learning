# coding: utf-8
# Dependencies: numpy, and matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data(pickle_filename):
    with open(pickle_filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data

def run_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans

def run_elbow_method(X, state_history_df, min_clusters=1, max_clusters=6):
    """
    The first argument, X, is what is used for clustering.
    The second argument, state_history_df, is a pandas data frame containing
    the time, temperature, density, and mass fractions. This is used for
    plotting the clusters so that some kind of comparisons can be made when
    comparing clustering done on different sets of variables (i.e. X's).
    """
    distortions = []
    inertias = []
    cluster_range = list(range(min_clusters, max_clusters))
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
        # Plot the clusters
        # plot_clusters(state_history_df['t'], state_history_df['T'], cluster_numbers)
        plot_clusters_detailed(state_history_df, cluster_numbers)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(cluster_range, distortions, 'bo-', label='Distortion')
    ax2.plot(cluster_range, inertias, 'rx-', label='Inertia')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('Distortion (blue)')
    ax2.set_ylabel('Inertia (red)')
    plt.show()

def plot_clusters(t, T, cluster_number):
    num_clusters = max(cluster_number) + 1
    plt.figure()
    plt.scatter(t, T, c=cluster_number, cmap='viridis')
    plt.xlabel('t')
    plt.ylabel('T')
    plt.title(f'Number of clusters = {num_clusters}')
    plt.show()

def plot_clusters_detailed(state_history_df, cluster_number):
    num_clusters = max(cluster_number) + 1
    vars_to_plot = ['T', 'Y_c12h26', 'Y_o2', 'Y_co2', 'Y_h2o', 'Y_co', 'Y_h2', 'Y_oh', 'Y_ho2']
    # headers = list(state_history_df.keys())
    # var_indices = [headers.index(var) for var in vars_to_plot]
    plt.subplots(3, 3)
    for i, var in enumerate(vars_to_plot):
        plt.subplot(331 + i)
        plt.scatter(
                state_history_df['t'],
                state_history_df[var],
                c=cluster_number,
                cmap='viridis',
                )
        plt.xlabel('t')
        plt.ylabel(var)
    plt.suptitle(f'Number of clusters = {num_clusters}')
    plt.tight_layout()
    plt.show()

def run_pca(X, n_components):
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

def main(data_pickle_filename='nDodecane_data.p'):
    data = load_data(data_pickle_filename)
    # Create a pandas object with the state history
    var_names = ['t', 'T', 'density']
    var_names.extend(['Y_' + var for var in data['species_names']])
    state_history_df = pd.DataFrame(
            np.concatenate((
                data['time_list'][:, np.newaxis],
                data['state_history']
                ), axis=1),
            columns=var_names,
            )
    # Run the elbow method on some data
    run_elbow_method(data['state_history'], state_history_df)
    # run_elbow_method(data['progress_rate_history'], state_history_df)
    # run_elbow_method(data['production_rate_history'], state_history_df)
    # run_elbow_method(data['heat_production_rate_history'], state_history_df)

#TODO: Add kmeans running and plotting code to main()
#TODO: Run kmeans on net_progress_variable or something similar
