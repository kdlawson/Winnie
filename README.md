# Winnie

Python package focusing on PSF-subtraction and forward modeling of circumstellar disks in high-contrast imaging data using reference star differential imaging (RDI). This initial release focuses on JWST coronagraphic observations and supports 1) RDI PSF-subtraction in various flavors, 2) disk model convolution using synthetic PSF grids, and 3) forward modeling for RDI.

RDI PSF subtraction and forward modeling should work for any JWST coronagraphy. However, model convolution is only directly supported for JWST round mask data (for now) — so, excluding MIRI FQPM data and NIRCam bar mask data. For forward modeling of these data, you will need to provide one of a) your own PSF grid, b) PSF grid generation functions, or c) already-convolved models. See documentation for winnie.SpaceRDI.prepare_convolution and winnie.SpaceRDI.set_circumstellar_model for more information.

Open an issue if you have any problems or want to suggest new features.

## Installation

#### Clone this repository:

```
git clone https://github.com/kdlawson/Winnie.git
```

#### (Optional) Create a conda environment for Winnie and its dependencies, then activate it:

Note: if you already have SpaceKLIP installed, you can install Winnie in that environment instead to save some trouble. In this case, consider adding the --dry-run keyword after "pip install" in the commands below to be sure that the changes that pip will make are acceptable to you. If they are, rerun the command without --dry-run. If they are not, create a new environment and proceed.

```
conda create -n winnie python=3.11
conda activate winnie
```

#### If it isn't already installed, install SpaceKLIP:

E.g., 

```
pip install git+https://github.com/spacetelescope/spaceKLIP.git
```

Or follow instructions in the [SpaceKLIP
documentation](https://spaceklip.readthedocs.io/en/latest/Installation-and-dependencies.html).


#### Change directory to the cloned repository:

```
cd Winnie
```

#### Install Winnie and its dependencies:

Installation with optional dependencies needed to run the tutorials:

```
pip install ".[tutorials]"
```

Minimal installation:

```
pip install .
```

#### Install data files for dependencies:

Before using Winnie, you'll need to install data files and set up environment variables required by a number of dependencies. Instructions for this are provided at the bottom of the [SpaceKLIP installation instructions](https://spaceklip.readthedocs.io/en/latest/Installation-and-dependencies.html).

## Usage

### Input data:

The code assumes your data were processed through the imagetools shift_frames
step from SpaceKLIP (align_frames in older versions), such that the position of the star is the same in all
images. I'd also strongly recommend coadding your integrations for each
exposure before proceeding. The code will do this anyway, but is much less
careful about updating header info. For a SpaceKLIP imagetools object called
ImageTools, this is just:

> ImageTools.coadd_frames()

For NIRCam coronagraphy, I would also suggest running SpaceKLIP's
NIRCam background subtraction routine as a final step:

> ImageTools.subtract_nircam_coron_background()

### Example:

```
from winnie import SpaceRDI
from spaceKLIP import database
import astropy.units as u
import glob

db = database.Database(output_dir='./psfsub/')  # Initialize the SpaceKLIP database

db.read_jwst_s012_data(datapaths=sorted(glob.glob('./coadded/*calints.fits')))  # Load calints from local dir 'coadded'

wdb = SpaceRDI(db,                    # Initialize the SpaceRDI object from the spaceKLIP database
               overwrite=True,        # Overwrite any existing SpaceRDI outputs
               r_opt=[0,4]*u.arcsec)  # Optimize the RDI PSF model over the region 0-4 arcsec from the star

wdb.load_concat('JWST_NIRCAM_NRCALONG_F356W_MASKRND_MASK335R_SUB320A335R')  # Load data for the desired concatenation

reduc = wdb.run_rdi(save_products=True)  # Carry out a basic RDI reduction and save products to output_dir
```

### Tutorials:

Included in the tutorials directory are some NIRCam observations of AU Mic and reference stars (GTO 1184; PI Josh Schlieder) for use with the [tutorial notebooks](https://github.com/kdlawson/Winnie/tree/main/tutorials).

The first two tutorial notebooks provide examples for the main uses (with the second providing a full example for forward modeling). The last two tutorials can be considered more advanced, and you can probably skip them if you're either a) already very familiar with PSF subtraction / good at parsing code and docstrings, or b) not already familiar but just want to run the code without thinking too much about it.

### Current limitations:

- Many of the operations are written to optionally run on a GPU (by passing use_gpu=True when initializing winnie.SpaceRDI). However, I'm still working through updating these functions in a few cases, so the code will currently override this option with use_gpu=False.

- winnie.SpaceRDI should propagate pixel-wise uncertainties from the ERR FITS extensions by default. However, a) these neglect speckle noise / speckle residuals, and b) the current SpaceKLIP ramp fitting with pseudo reference pixels significantly reduces noise but doesn't change the noise maps versus the standard pipeline — so these (generally) overestimate the noise. In other words, these are not statistically robust currently, but may still be useful for relative comparisons.

- Explicit support for other observatories / observing modes is not yet included. However, most of the underlying functions are generalized for use with other data (e.g., winnie.rdi.rdi_residuals). They simply lack the automated loading / book-keeping / etc. provided for JWST coronagraphy. Please reach out if you would like guidance in using these tools with other data.

## Citation

There is no dedicated reference for Winnie yet. However, the majority of the methods used by Winnie have been published. Some suggested citations for specific procedures implemented in the code are provided below.

For use of the Model Constrained RDI (MCRDI), please cite:
[Lawson et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...935L..25L/abstract)

For the JWST convolution procedure, please cite:
[Lawson et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023AJ....166..150L/abstract)

Please also consider including citations to Winnie's dependencies — especially those that are under active development. Whether you're directly importing these packages or not, they are critical building blocks for the existence of this tool. I can assure you that there is no world in which I started working on this, noticed that POPPY didn't exist, and said to myself "Oh, I guess I'll just have to write some code to simulate optical propagation for telescopes". This package simply wouldn't exist and you wouldn't be using it for your science. For the continued existence of this package, these actively-developed dependencies require funding, and the would-be funding sources require evidence that these dependencies are being used. <font color='red'>**Let funding sources know that you have an interest in the continued development of these dependencies by citing them in your publications that make use of Winnie.**</font>

Among these dependencies are:  

    Astropy (Astropy Collaboration et al. 2013, 2018, 2022)  
    CuPy (Okuta et al. 2017)  
    JWST Calibration Pipeline (Bushouse et al. 2022)
    LMFIT (Newville et al. 2022)  
    Matplotlib (Hunter 2007; Caswell et al. 2021)  
    NumPy (Harris et al. 2020)  
    POPPY (Perrin et al. 2012)  
    pyKLIP (Wang et al. 2015)  
    SciPy (Virtanen et al. 2020)  
    SpaceKLIP (Kammerer et al. 2022)  
    Vortex Image Processing (Gomez Gonzalez et al. 2017)  
    WebbPSF (Perrin et al. 2014)  
    WebbPSF-ext (Leisenring 2021)  

For publications drafted with aastex, these can simply be included in the acknowledgments using the \software tag:
> \software{Astropy \citep{astropy2013, astropy2018, astropy2022},...}  

Finally: the namesake of this package is a small dog named Winnie. Much of this code was written during COVID-19 lockdowns and the aftermath with Winnie sitting on my lap (indeed, I often worked hours later than I intended because she was asleep, and what sort of monster would wake a sleeping puppy?). If companionship was regarded as a criterion for authorship, Winnie would be the first author on the publication for this software. Feel free to include an acknowledgement of Winnie's contributions to this tool instead of or in addition to citing my silly little papers. E.g.,  
> We acknowledge a humble dog named Winnie, whose companionship contributed significantly to the creation of the software called \texttt{Winnie}, which is utilized herein.

<figure>
<img src="winnie.jpeg" alt="A picture of a small black dog with white markings looking upwards while wearing a frog hoodie and standing in front of the JWST image of the Carina Nebula" width="500"/>
<figcaption>Fig.1 - a small dog named Winnie visiting the Carina Nebula.</figcaption>
</figure>