# Machine Learning Applications to Chemical Kinetics

This repository contains my trials of applying machine learning to chemical
kinetics. There are multiple pieces of code in this repository. This document
should provide some insight into what each piece does and how to use it.

## Reaction Extraction from RMG

Reaction Mechanism Generator (RMG) is an open source tool developed by MIT to
generate chemical kinetic mechanisms. The code ships with a lot of kinetic
libraries. To create a dataset for our purposes, we can extract this
information to train Neural Networks on.

- `cgt.py`: This file defines the Specie class which stores information about
  a chemical specie. It accepts a chemical adjacency list and extends
  the VF2 algorithm implemented in networkx package to account for atom and
  bond types (semantic feasibility). This can then be used to compare whether
  or not the graphs of two species are the same or not.
- `reaction.py`: This file defines the Reaction class which stores basic
  information about reactions including the reactants, products, and kinetics.
- `parse_rmg_kinetics_library.py`: This utility extracts reactions and
  parameters from an RMG kinetics library and stores it as a pickle file. RMG
  kinetics libraries are valid python code so this utility creates wrappers
  for each class type built into RMG considering only the relevant fields and
  uses that to parse the library file. This file is designed to be run as a
  script from the command line:

    $ python parse_rmg_kinetics_library.py <kinetics_library_directory>

  It stores the output in `rmg_compiled_pickles/`. There will be one pickle
  file per RMG library file (they will have the same name).
- `check_name_atom_count_differences.py`: Within a single RMG library, each of
  the species has a graph structure and an identifier. The identifier is used
  when including the species in chemical reactions. However, across libraries,
  different identifiers could be used to describe the same specie and the same
  identifier might refer to different species. When creating a single database
  of reactions, we need to standardize the names of the species. This short
  utility checks the converted pickle files (see
  `parse_rmg_kinetics_library.py` utility above) for whether or not there are
  multiple species with the same identifier but different atom counts. This is
  not a fool proof way to check that the identifiers are consistent, but it is
  a good, quick, first check.
- `update_species_map.py`: This utility creates a mapping between species and
  integers to be able to uniquely identify each specie with integers later in
  ML codes. Run this from the command line as:

    $ python update_species_map.py <reactions csv file>

  The output of this code is a pickle file (`species_map.p`) that contains a
  list of unique species. The index of each specie in the list gives the unique
  index.
- `extract_reactions.py`: This command line utility is able to extract a
  subset of the reactions (from the equivalent RMG pickle files generated using
  `parse_rmg_kinetics_library.py`) for analysis. For the most initial part of the
  work, I wanted to extract only the Arrhenius type reactions from the entire
  database (I never extended it to do anything else). Run this on the command
  line as:

    $ python update_species_map.py <directory to source pickles> <output csv name>

  This creates a csv file (`arrhenius.csv`) that contains information about
  each of the reactions. This takes into account that different RMG kinetics
  libraries (stored in the different pickle files) might have the same species
  identified by different names and so the names need to be standardized.

## Machine Learning

- `learn_chemistry.py`: This implements the basic machine learning framework to
  try and learn the rate parameters of Arrhenius type elementary reactions. It
  implements cross-validation as well for network architecture and
  hyperparameter selection. This script can be run in several ways from the
  command line to achieve different things:

  - The script defines two functions `run_uniform_models()` and
    `run_nonuniform_models()`. These contain the various combinations of model
    architectures and hyperparameter combinations that can be run. To train
    models with parameters with these different combinations of parameters, run

        $ python learn_chemistry.py train all

    Alternatively, to run cross-validation on these combinations of parameters,
    run

        $ python learn_chemistry.py cv

    Both of these will output a bunch of model files and plots that can be
    used. The model files contain the model weights that can be used to restart
    the model.
  - After running cross-validation, it is possible to extract a list of "top
    models" (using the `process_metrics.py`) script. Then the output of that
    script (which is a pickle file containing a pandas DataFrame) can be passed
    back in to this script to train the models. To do this, run

        $ python learn_chemistry.py train <model_summary_file>

    The output, again, will be a bunch of model files and plots. The difference
    between this and the regular training is that this also runs the testing
    dataset through the model.
- `process_metrics.py`: This script contains two functions that have very
  similar functions.

  - `main_model()` can be used to compile a bunch of model runs and extract
    various quantities of interest (Pearson Correlation coefficients, best fit
    lines between true and predicted output, loss history, etc). The data is
    then returned as pandas DataFrames. Use this function in an interactive
    python shell.
  - `main_folds()` can be used to do the same but for cross-validation output.
    This can be used from an interactive shell. While running, the best models
    (according to various metrics) are saved as a pandas DataFrame and written
    to the file `best_crossvalidation_models.p`. This functionality can also be
    accessed directly from the command line by running

        $ python process_metrics.py

    When run from the command line, there should not be much output to the
    screen (perhaps just some info messages). The data will only be stored in
    the pickle file.
