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
import pickle

import time

import cantera as ct
print('Running Cantera version: ' + ct.__version__)


# Define the ignition delay time (IDT). This function computes the ignition
# delay from the occurrence of the peak concentration for the specified
# species.
def ignitionDelay(states, species):
    i_ign = states(species).Y.argmax()
    return states.t[i_ign]

def load_kinetics(mech='nDodecane_Reitz.yaml', gas_type='nDodecane_IG'):
    gas = ct.Solution(mech, gas_type)
    return gas

# def create_shock_tube(T, p, mech='nDodecane_Reitz.yaml', gas_type='nDodecane_IG'):
def create_shock_tube(T, p, gas):
    # gas = ct.Solution(mech, gas_type)
    gas.TP = T, p
    gas.set_equivalence_ratio(phi=1.0, fuel='c12h26',
                                   oxidizer={'o2': 1.0, 'n2': 3.76})
    # Create the reactor and network
    r = ct.Reactor(contents=gas)
    reactorNetwork = ct.ReactorNet([r])
    reactorNetwork.max_time_step = 1e-8
    return r, reactorNetwork

def run_shock_tube(gas, reactor, reactorNetwork, estimatedIgnitionDelayTime=0.001):
    # estimatedIgnitionDelayTime is a starting estimate. If you do not get an
    # ignition within this time, increase it
    timeHistory = ct.SolutionArray(gas, extra=['t'])
    time_list = []
    pressure_history = []
    state_history = []
    progress_rate_history = []
    production_rate_history = []
    heat_production_rate_history = []

    t0 = time.time()

    t = 0
    counter = 1
    while t < estimatedIgnitionDelayTime:
        t = reactorNetwork.step()
        if counter % 20 == 0:
            # We will save only every 20th value. Otherwise, this takes too long
            # Note that the species concentrations are mass fractions
            timeHistory.append(reactor.thermo.state, t=t)
            time_list.append(t)
            pressure_history.append(reactor.thermo.P)
            state_history.append(reactor.thermo.state.tolist())
            progress_rate_history.append(reactor.kinetics.net_rates_of_progress.tolist())
            production_rate_history.append(reactor.kinetics.net_production_rates.tolist())
            heat_production_rate_history.append(reactor.kinetics.heat_production_rates.tolist())
        counter += 1

    # We will use the 'oh' species to compute the ignition delay
    tau = ignitionDelay(timeHistory, 'oh')

    t1 = time.time()
    print("Computed Real Gas Ignition Delay: {:.3e} seconds. "
          "Took {:3.2f}s to compute".format(tau, t1-t0))

    return {
            'time_list': np.array(time_list),
            'pressure_history': np.array(pressure_history),
            'state_history': np.array(state_history),
            'progress_rate_history': np.array(progress_rate_history),
            'production_rate_history': np.array(production_rate_history),
            'heat_production_rate_history': np.array(heat_production_rate_history),
            }

def freeze_sp(gas, sp):
    for i, reaction in enumerate(gas.reactions()):
        if sp in reaction.reactants:
            gas.set_multiplier(0, i)
        elif sp in reaction.products:
            gas.set_multiplier(0, i)
    return gas

def write_data(data, filename):
    with open(filename, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)

def main(frozen_species=[], pickle_filename='nDodecane_data.p', estimatedIgnitionDelayTime=0.001, mechanism='nDodecane_Reitz.yaml', gas_type='nDodecane_IG', T=1000., p=40.0*101325.0):
    gas = load_kinetics(mechanism, gas_type)
    for sp in frozen_species:
        gas = freeze_sp(gas, sp)
    reactor, reactor_network = create_shock_tube(T, p, gas)
    data = run_shock_tube(gas, reactor, reactor_network, estimatedIgnitionDelayTime=estimatedIgnitionDelayTime)
    species_names = [sp.name for sp in reactor.thermo.species()]
    data['species_names'] = species_names
    write_data(data, pickle_filename)
    return data
