# Winnie

Python package focusing on PSF-subtraction and forward modeling of circumstellar disks in high-contrast imaging data. An initial release (~ May 2024) will focus on JWST coronagraphic observations and will support RDI PSF-subtraction in various flavors, disk model convolution using synthetic PSF grids, and forward modeling for RDI.

#### Notes for early testing: 

1) the tutorial notebooks (in Winnie/tutorials) will temporarily add the code to your path, so no installation is needed for now. Just clone the repo, install dependencies, and open the first tutorial notebook to get started

2) included in the tutorials directory are some NIRCam AU Mic + reference observations to use in the tutorials

3) the first two tutorial notebooks provide examples for the main uses (with the second providing a full example for forward modeling). The last two tutorials can be considered more advanced, and you can probably skip them if you're either a) already very familiar with PSF subtraction / good at parsing code and docstrings, or b) unfamiliar but just want to run the code without thinking too much about it

4) There is no setup file yet, so you'll need to verify that you have the dependencies on your own (most of them you should already have)

**Dependencies:**

astropy

jwst

joblib

matplotlib

numpy

pyklip

scipy

spaceKLIP (develop branch)

webbpsf (develop branch)

webbpsf-ext (develop branch)



**And a few extra dependencies for the tutorial:**

lmfit

synphot (with data files for the ck04 library)

vip_hci