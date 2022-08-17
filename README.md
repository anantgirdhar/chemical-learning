# Machine Learning Applications to Chemical Stuff

This repository contains my trials of applying machine learning to chemical
stuff. This branch currently deals with my trials applying these techniques to
some fluid flow problems (hopefully ending with applications to shocks and
non-equilibrium type flows). Hopefully this will also tie back in to the bigger
picture of applications to chemical kinetics. This document should include some
description of the code files in here.

## Unsupervised Learning

- `shock_tube_unsupervised_learning.py`:
  - I tried to set up KMeans to run on shock tube like data to see what how the
    algorithm decided to divide up the domain into different pieces based on
    different quantities. This code could probably be cleaned up a bit.
  - Uses the elbow method to find the appropriate number of clusters.
  - Uses the thermodynamic state variables, progress rates, production rates,
    etc. to perform clustering.
  - It would be nice to use something other than the elbow method. Perhaps
    something like Silhouette coefficient would work well. Check out the
    towards data science articles talking about the
    [basics](https://towardsdatascience.com/clustering-metrics-better-than-the-elbow-method-6926e1f723a6)
    and a [deeper dive with
    code](https://towardsdatascience.com/silhouette-method-better-than-elbow-method-to-find-optimal-clusters-378d62ff6891)
    for some inspiration.
  - Perhaps this can also be extended to PCA.
- `reaction_classification_BoW`:
  - I was interested to know if it was possible to cluster the reactions based
    on something. I used the Bag of Words concept treating each reaction based
    on just the molecules that appear in it. The molecules were either
    represented as the molecule strings or as counts of each of the atoms in
    each molecule.
  - I wasn't really sure how to interpret the results so I've left this for
    later.
