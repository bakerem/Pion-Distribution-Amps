# ANL

This contains all of the code used for my data analysis project at ANL in Summer 2023. The code takes Lattice QCD data and analyzes it in order to extract the Mellin moments
and $x$-distribution of the pion distribution amplitude. The workflow is as follows:

1) Perform 1 state fits to check quality of data and find effective mass plateaus (stats/one_state_fits.py)
2) Perfom 2 state fits to extract 1st excited state energies and amplitudes (stats/two_state_fits.py)
3) Use this to calculate the bare matrix elements (data0/raw_el_fits.py)
4) Renormalize bare matrix elements (data0/renorm_jack.py)
5) Calculate Mellin moments for a set of fitting parameters (data0/mellin_moms.py)
6) Calculate $alpha$ for the one-parameter Ansatz ($\phi(u)=u^\alpha(1-u)^\alpha$) to determine which basis to expand DA in (data0/x_dep_alpha.py)
7) Calculate coefficients for expansion of DA (data0/x_dep_sns.py)

Modules for plotting are generally labeled as such and auxilliary functions are found in data0/functions.py

There are a couple conventions that are uninituitve. First, the quality labelled as $a$ is not the lattice spacing, but actually the inverse of the lattice spacing
in GeV. This is due to an inital misunderstanding that has been easier to keep instead of fixing everywhere. This means that in the code where a $\times a$ should appear,
a $/a$ appears instead. 

Below is an outline of the code:

![code_outline](https://github.com/bakerem/ANL/assets/96547711/f1e5cd09-384c-40b4-8258-6916747c2e44)
