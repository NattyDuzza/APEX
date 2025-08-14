<img src="https://github.com/NattyDuzza/APEX/blob/main/Cosmic%20Web%20Logo%20for%20APEX.png" width="250">

# APEX
Angular Power Spectrum Explorer - A Python 'library' for extracting angular power spectrum data from sacc files, as well as overseeing their creation and use in modelling. Created originally to implement a model by Sara Maluebre (University of Oxford), but intended to expand to cover a wider range of applications and models.


## Installation

The files needed to run the full suite of APEX functions are currently:

- APEX.py
- Plots.py
- InputHandler.py

These can be sourced from this github page. Use the development version at your own risk... I try to make the 'main' version 99% functional before publishing, but the development branch will often be a temporary image half way through a fix.
The import of the latter two of these files is handled automatically by APEX.py. Import APEX.py as usual:

```python
import APEX as ap
```

## Usage

The current main branch of APEX has the following features:


- Read SACC files (see https://sacc.readthedocs.io/) and retrieve data from them. This can be done in the usual way, or one can choose to define the range of multipoles the data returned is given at
- Create runtime PYCCL (see https://ccl.readthedocs.io/) tracer objects for use in cosmological modelling
- Allows for analysis on maps with different naming conventions through using aliases
- Generates tracer workspaces by type to aid with use in iterative coding
- Allows a customised cobaya (see https://cobaya.readthedocs.io/) MCMC set-up experience as well as the native one
- Can run a full MCMC cobaya, or a Minimize, on a given set of maps to obtain measurements of galaxy bias and bias weighted star formation rate density

For examples of how to use APEX, please refer to the examples folder on the github.
