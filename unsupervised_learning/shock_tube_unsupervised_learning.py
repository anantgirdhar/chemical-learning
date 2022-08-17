# coding: utf-8
"""
Ignition delay time computations in a high-pressure reflected shock tube
reactor, comparing ideal gas and Redlich-Kwong real gas models.

In this example we illustrate how to setup and use a constant volume,
adiabatic reactor to simulate reflected shock tube experiments. This reactor
will then be used to compute the ignition delay of a gas at a specified
initial temperature and pressure. The example is written in a general way,
that is, no particular EoS is presumed and ideal and real gas EoS can be used
equally easily.

The reactor (system) is simply an 'insulated box,' and can technically be used
for any number of equations of state and constant-volume, adiabatic reactors.

Other than the typical Cantera dependencies, plotting functions require that
you have matplotlib (https://matplotlib.org/) installed.

Requires: cantera >= 2.5.0, matplotlib >= 2.0
Keywords: combustion, reactor network, non-ideal fluid, ignition delay, plotting
"""

# Dependencies: numpy, and matplotlib
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import time

import cantera as ct
print('Running Cantera version: ' + ct.__version__)


# Define the ignition delay time (IDT). This function computes the ignition
# delay from the occurrence of the peak concentration for the specified
# species.
def ignitionDelay(states, species):
    i_ign = states(species).Y.argmax()
    return states.t[i_ign]

def run_shock_tube(T, p, mech, gas_type):
    gas = ct.Solution(mech, gas_type)
    gas.TP = T, p
    gas.set_equivalence_ratio(phi=1.0, fuel='c12h26',
                                   oxidizer={'o2': 1.0, 'n2': 3.76})
    # Create the reactor and network
    r = ct.Reactor(contents=gas)
    reactorNetwork = ct.ReactorNet([r])
    timeHistory = ct.SolutionArray(gas, extra=['t'])
    time_list = []
    state_history = []
    progress_rate_history = []
    production_rate_history = []
    heat_production_rate_history = []

    t0 = time.time()

    # This is a starting estimate. If you do not get an ignition within this time,
    # increase it
    estimatedIgnitionDelayTime = 0.005
    t = 0
    counter = 1
    while t < estimatedIgnitionDelayTime:
        t = reactorNetwork.step()
        if counter % 20 == 0:
            # We will save only every 20th value. Otherwise, this takes too long
            # Note that the species concentrations are mass fractions
            timeHistory.append(r.thermo.state, t=t)
            time_list.append(t)
            state_history.append(r.thermo.state.tolist())
            progress_rate_history.append(r.kinetics.net_rates_of_progress.tolist())
            production_rate_history.append(r.kinetics.net_production_rates.tolist())
            heat_production_rate_history.append(r.kinetics.heat_production_rates.tolist())
        counter += 1

    # We will use the 'oh' species to compute the ignition delay
    tau = ignitionDelay(timeHistory, 'oh')

    t1 = time.time()
    print("Computed Real Gas Ignition Delay: {:.3e} seconds. "
          "Took {:3.2f}s to compute".format(tau, t1-t0))

    return (
            timeHistory,
            reactorNetwork,
            r, (
                np.array(time_list),
                np.array(state_history),
                np.array(progress_rate_history),
                np.array(production_rate_history),
                np.array(heat_production_rate_history),
                )
            )

def run_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return kmeans

def run_elbow_method(X, time_history_df):
    distortions = []
    inertias = []
    cluster_range = list(range(1, 10))
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
        # plot_clusters(time_history_df['t'], time_history_df['T'], cluster_numbers)
        plot_clusters_detailed(time_history_df, cluster_numbers)
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

def plot_clusters_detailed(time_history_df, cluster_number):
    num_clusters = max(cluster_number) + 1
    vars_to_plot = ['T', 'Y_c12h26', 'Y_o2', 'Y_co2', 'Y_h2o', 'Y_co', 'Y_h2', 'Y_oh', 'Y_ho2']
    # headers = list(time_history_df.keys())
    # var_indices = [headers.index(var) for var in vars_to_plot]
    plt.subplots(3, 3)
    for i, var in enumerate(vars_to_plot):
        plt.subplot(331 + i)
        plt.scatter(
                time_history_df['t'],
                time_history_df[var],
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

def main():
    # Define the reactor temperature and pressure:
    T = 1000  # Kelvin
    p = 40.0*101325.0  # Pascals
    # ideal_gas = ct.Solution('nDodecane_Reitz.yaml', 'nDodecane_IG')
    # gas = ct.Solution('nDodecane_Reitz.yaml', 'nDodecane_RK')
    (
        time_history,
        reactor_network,
        reactor, (
            time_list,
            state_history,
            progress_rate_history,
            production_rate_history,
            heat_production_rate_history,
            ),
        ) = run_shock_tube(T, p, 'nDodecane_Reitz.yaml', 'nDodecane_IG')
    # run_elbow_method(progress_rate_history, time_history.to_pandas())
    # run_elbow_method(production_rate_history, time_history.to_pandas())
    run_elbow_method(heat_production_rate_history, time_history.to_pandas())

#TODO: Add kmeans running and plotting code to main()
#TODO: Run kmeans on net_progress_variable or something similar
