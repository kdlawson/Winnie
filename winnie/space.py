import numpy as np
from astropy.io import fits
from astropy import wcs, convolution
from pyklip.klip import _rotate_wcs_hdr
import numbers
from copy import (copy, deepcopy)
import astropy.units as u
import os
import webbpsf
import webbpsf_ext
import pickle
from spaceKLIP.database import Database as spaceklip_database
from tqdm.auto import tqdm

from .rdi import (rdi_residuals, build_annular_rdi_zones)

from .plot import (mpl, plt, quick_implot, mpl_centered_extent)

from .utils import (robust_mean_combine, median_combine,
                    ang_size_to_px_size, px_size_to_ang_size,
                    high_pass_filter_sequence, pad_and_rotate_hypercube, rotate_hypercube,
                    xy_polar_ang_displacement, rotate_image, gaussian_filter_sequence, crop_data,
                    c_to_c_osamp, pad_or_crop_image, dist_to_pt, compute_derot_padding, 
                    nan_median_absolute_deviation)

from .convolution import (convolve_with_spatial_psfs,
                          get_jwst_psf_grid_inds,
                          get_jwst_coron_transmission_map,
                          generate_lyot_psf_grid,
                          get_webbpsf_model_center_offset)

from .deconvolution import (coronagraphic_richardson_lucy)


class SpaceRDI:
    def __init__(self, database=None, output_basedir=None,
                 output_subdir='WinnieRDI', data_ext=None,
                 ncores=-1, use_gpu=False, verbose=True, show_plots=False,
                 overwrite=False, prop_err=True, show_progress=False,
                 use_robust_mean=False, robust_clip_nsig=3, pad_data='auto', 
                 pad_before_derot=False, r_opt=3*u.arcsec, r_sub=None, 
                 save_coron_transmission=True, save_instance=False, 
                 efficient_saving=True, from_fits=None):
        """
        Initialize the Winnie class for carrying out RDI on JWST data.

        Parameters
        ----------
         database: spaceKLIP.database.Database
            SpaceKLIP database containing stage 2 observations to work with.

        output_basedir: str
            Base directory for saving reduction products. If not provided, the
            output directory will be the same as that of the input database.
        
        output_subdir: str
            Subdirectory within output_basedir to save reduction products. If
            not provided, the subdirectory will be 'WinnieRDI'.

        data_ext: str
            The file extension for the input FITS files (e.g., 'calints'). If
            not provided, the file extension will be assumed to be the text
            following the final underscore in the name of the first file of the
            first concatenation (excluding the '.fits' extension).

        ncores: int
            Number of processor cores to use where applicable.

        use_gpu: bool
            Use GPU operations on a CUDA-capable GPU in place of some CPU
            operations. Note: this is currently not used. Still working on
            updating the GPU code.

        verbose: bool
            Provides some reduction info in print statements in a few places.

        show_plots: bool
            Shows some diagnostic plots when verbose is True.

        overwrite: bool
            When saving reduction products, overwrite existing products if
            True.

        prop_err: bool
            If True, will load ERR array extension and propagate error through
            any RDI reduc of the data.

        show_progress: bool
            If True, will show progress bars for the RDI procedure. Note:
            usually unnecessary (and uninformative) for JWST data, where the
            small number of frames makes PSF subtraction very fast.

        use_robust_mean: bool
            If data are not already coadded, will combine integrations using a
            sigma clipped mean rather than the median.

        robust_clip_nsig: int or float
            If use_robust_mean=True, the number of median absolute deviations
            above or below the median for a value to be clipped.

        pad_data: int or str
            If pad_data is an int, pads the data (both science and reference)
            with pad_data pixels along each end of the spatial axes. E.g.,
            input data of shape (320,320) with pad_data=5 will yield data of
            shape (330,330). If pad_data is 'auto', automatically pads the data
            such that no data is lost when derotating by the position angles of
            the science data. Setting pad_data = None will prevent padding.

        pad_before_derot: bool
            If True, pads data at the derotation step to avoid cutting off
            pixels during rotation. Redundant if used with pad_data='auto'.
            Technically slightly more resource efficient than pad_data='auto',
            but makes some details of forward modeling a lot trickier.

        r_opt: array, float, None
            The optimization zone radii to pass as 'r_opt' to
            winnie.rdi.build_annular_rdi_zones when loading each concatenation.
            See winnie.rdi.build_annular_rdi_zones doctstring for more info on
            permitted formats. Defaults to 3*u.arcsec (producing a single
            optimization zone spanning 0-3 arcsec from the star).

        r_sub: array, float, None
            The subtraction zone radii to pass as 'r_sub' to
            winnie.rdi.build_annular_rdi_zones when loading each concatenation.
            See winnie.rdi.build_annular_rdi_zones doctstring for more info on
            permitted formats. Defaults to None (producing a single subtraction
            zone spanning the field of view).

        save_coron_transmission: bool
            If True, when executing run_rdi with save_products=True and
            forward_model=False, saves the derotated and averaged PSF mask for
            each reduction for compatability with SpaceKLIP (as long as masks
            have been set).

        save_instance: bool
            If True, writes the current instance of the SpaceRDI object as an
            HDU when saving reduction products to disk. This instance can be
            reconstructed by passing the fits filename as the from_fits
            argument when initializing a SpaceRDI object.

        efficient_saving: bool
            If True and if save_instance, stores the SpaceRDI object with only
            the data that should not already be stored on disk (e.g., not the
            data arrays themselves). This significantly reduces output file
            sizes, but does not guarantee that the object can be precisely
            reconstructed later (e.g., if the data files are changed, deleted,
            or moved). With efficient_saving=False, expect file sizes to be
            large (~hundreds of MB without a PSF grid loaded, ~1-2 GB if a PSF
            grid is loaded). The benefit of efficient_saving=False is that the
            reconstruction is relatively robust and includes all of the data
            needed in a single file.

        from_fits: str
            Path to a Winnie reduction FITS file created with
            save_instance=True from which the SpaceRDI object will be
            recreated. If from_fits is provided, all other arguments are
            ignored.
        """
        if from_fits is not None:
            self._from_fits(from_fits)
        else:
            self.concat = None
            self.convolver = None
            self.database = database
            output_basedir = self.database.output_dir if output_basedir is None else output_basedir
            if not os.path.isdir(output_basedir):
                os.makedirs(output_basedir)
            self.output_dir = f'{output_basedir}{output_subdir}/'
            self.output_subdir = output_subdir
            self.output_basedir = output_basedir
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)
            if data_ext is None:
                self.data_ext = self.database.obs[list(self.database.obs.keys())[0]]['FITSFILE'][0].split('_')[-1].removesuffix('.fits')
            else:
                self.data_ext = data_ext
            self.ncores = ncores
            self.use_gpu = use_gpu
            self.verbose = verbose
            self.show_plots = show_plots
            self.overwrite = overwrite
            self.show_progress = show_progress
            self.prop_err = prop_err
            self.use_robust_mean = use_robust_mean
            self.robust_clip_nsig = robust_clip_nsig
            self.pad_data = pad_data
            self.pad_before_derot = pad_before_derot
            self.r_opt = r_opt
            self.r_sub = r_sub
            self.save_coron_transmission = save_coron_transmission
            self.save_instance = save_instance
            self.efficient_saving = efficient_saving
            if self.use_gpu:
                print("Warning! GPU implementation still in progress; setting use_gpu to False.")
                self.use_gpu = False
       
        
    def load_concat(self, concat, coron_offsets=None, cropped_shape=None):
        """
        Load data for an indicated concatenation in preparation for an RDI
        reduction. By default, this prepares simple annular optimization zones
        for PSF-subtraction. After running load_concat, these zones can be
        changed as you prefer using the set_zones method.
        
        Parameters
        ----------
        concat: str or int
            Either the full concatenation string or the index of the desired
            concatenation. 

        coron_offsets: array
            Array of shape (Nobs, 2), providing the offset of the coronagraph
            from the star (in pixels) for each file contained in the
            concatenation table. To be used for data that have been aligned but
            lack the spaceklip alignment header info.

        cropped_shape: array or tuple
            Spatial dimensions to which data should be cropped, as [ny,nx]. Can
            be set later or returned to original shape using the set_crop
            method. Cropping is primarily for improving runtimes during forward
            modeling.
        """
        if isinstance(concat, numbers.Number):
            concat_str = list(self.database.obs.keys())[concat]
        else:
            concat_str = concat
        
        db_tab = self.database.obs[concat_str]

        sci = db_tab['TYPE'] == 'SCI'
        ref = db_tab['TYPE'] == 'REF'

        files = db_tab['FITSFILE']

        self._c_star = np.array([db_tab['CRPIX1'][0], db_tab['CRPIX2'][0]])-1 # Position of the star in the data before any cropping
        self.pxscale = db_tab[0]['PIXSCALE']*u.arcsec/u.pixel # pixel scale with astropy units
        self._lam = db_tab['CWAVEL'][0]*u.micron
        self._blurfwhm = np.nan_to_num(db_tab['BLURFWHM'][0])
        
        if db_tab['EXP_TYPE'][0] == 'NRC_CORON':
            self._d_eff = 5.2*u.meter
        else:
            self._d_eff = 6.603464*u.meter
            
        self._instfwhm = ang_size_to_px_size(np.rad2deg((self._lam/(self._d_eff.to(u.micron))).value)*u.deg, self.pxscale).value
        self._fwhm = np.hypot(self._instfwhm, self._blurfwhm)
        self._sigma = self._fwhm/np.sqrt(8.*np.log(2.))
        self.concat = concat_str
        
        imcube = []
        errcube = []
        posangs = []
        visit_ids = []
        c_coron = []
        dates = []
        self._coron_offsets = []
        for i,f in enumerate(files):
            with fits.open(f, lazy_load_hdus=True) as hdul:
                ints = hdul[1].data
                errs = hdul[2].data if self.prop_err else None
                h0, h1 = hdul[0].header, hdul[1].header
                if coron_offsets is None:
                    if 'MASKOFFS' in hdul:
                        offset = np.mean(hdul['MASKOFFS'].data, axis=0)
                    else:
                        if i==0:
                            print("WARNING: No alignment header info found. Assuming perfect coronagraph alignment.")
                        offset = np.array([0,0])
                else:
                    offset = coron_offsets[i]
            if np.ndim(ints.squeeze()) != 2:
                if self.use_robust_mean:
                    im, err = robust_mean_combine(ints, errs, self.robust_clip_nsig)
                else: 
                    im, err = median_combine(ints, errs)
            else:
                im, err = ints.squeeze(), (None if not self.prop_err else errs.squeeze())
                    
            imcube.append(im)
            errcube.append(err)
            posangs.append(h1['PA_APER'])
            visit_ids.append(h0['VISIT_ID'])
            c_coron.append(self._c_star-offset)
            dates.append(h0['DATE-BEG'])
            self._coron_offsets.append(offset)
        
        self._coron_offsets = np.array(self._coron_offsets)
        self._image_mask = h0['CORONMSK'].replace('MASKA', 'MASK')
        if 'PUPIL' in h0:
            self._pupil_mask = h0['PUPIL']
        self._aperturename = h0['APERNAME']
        self.filt = h0['FILTER']

        self._imcube_sci = np.array(imcube)[sci]
        self._errcube_sci = np.array(errcube)[sci] if self.prop_err else None
        self._posangs_sci = np.array(posangs)[sci]
        self._visit_ids_sci = np.array(visit_ids)[sci]
        self._c_coron_sci = np.array(c_coron)[sci]
        self._dates_sci = np.array(dates)[sci]
        self._files_sci = np.array(files)[sci]

        self._imcube_ref = np.array(imcube)[ref]
        self._errcube_ref = np.array(errcube)[ref] if self.prop_err else None
        self._posangs_ref = np.array(posangs)[ref]
        self._visit_ids_ref = np.array(visit_ids)[ref]
        self._c_coron_ref = np.array(c_coron)[ref]
        self._dates_ref = np.array(dates)[ref]
        self._files_ref = np.array(files)[ref]
        
        self._ny, self._nx = self._imcube_sci.shape[-2:]
        
        if self.pad_data is not None: self._apply_padding()

        self._imcube_css = None
        self.cropped_shape = cropped_shape
        
        self.rdi_settings = {}
        self.fixed_rdi_settings = {}

        # Setting initial annular zones for RDI procedure based on set defaults.
        self.update_annular_zones(exclude_opt_nans=True)

        self.set_crop(cropped_shape)

        self.rdi_presets()

        if self.convolver is not None:
            self.convolver.load_concat(self.concat, **self.convolver_args)

        if self.save_coron_transmission:
            outfile = self.output_dir + self.concat + '_psfmask.fits'
            if not os.path.exists(outfile) or np.shape(fits.getdata(outfile)) != [self.ny, self.nx]:
                _ = self.make_derot_coron_maps(collapse_rolls=False, save_products=True)


    def _apply_padding(self):
        if self.pad_data == 'auto':
            dymin_pad, dymax_pad, dxmin_pad, dxmax_pad = compute_derot_padding(self._nx, self._ny, -self._posangs_sci, cent=self._c_star)
        else:
            dymin_pad = dymax_pad = dxmin_pad = dxmax_pad = self.pad_data

        imc_padding = [[0,0], [dymin_pad, dymax_pad], [dxmin_pad, dxmax_pad]]
        cent_adj = np.array([dxmin_pad, dymin_pad])

        self._imc_padding = imc_padding
        self._imcube_sci = np.pad(self._imcube_sci, imc_padding, constant_values=np.nan)
        self._errcube_sci = np.pad(self._errcube_sci, imc_padding, constant_values=np.nan) if self.prop_err else None
        self._imcube_ref = np.pad(self._imcube_ref, imc_padding, constant_values=np.nan)
        self._errcube_ref = np.pad(self._errcube_ref, imc_padding, constant_values=np.nan) if self.prop_err else None

        self._c_star += cent_adj
        self._c_coron_sci += cent_adj
        self._c_coron_ref += cent_adj

        self._ny, self._nx = self._imcube_sci.shape[1:]


    def set_crop(self, cropped_shape=None, auto_pad_nfwhm=5):
        """
        If cropped_shape is a tuple or two-element array: the desired spatial dimensions
        of the cropped data as [ny, nx].

        For cropped_shape = None, returns data to the original uncropped dimensions

        For cropped_shape = 'auto':
        - calculates the largest separation of pixels included in the optimization zones,
          and adds auto_pad_nfwhm times the effective FWHM in pixels
        - sets the cropped shape such that those separations are included in the FOV.
        """
        if cropped_shape == 'auto':
            rmax = np.max(dist_to_pt(self._c_star, nx=self._nx, ny=self._ny)[np.any(self._optzones, axis=0)])
            new_nx = new_ny = int(rmax*2 + auto_pad_nfwhm*self._fwhm)
            cropped_shape = [new_ny, new_nx]

        if cropped_shape is not None:
            self.cropped_shape = np.asarray(cropped_shape)
            self.imcube_sci, self.c_star, self._crop_indices = crop_data(self._imcube_sci, self._c_star, self.cropped_shape, return_indices=True, copy=False)
            y1, y2, x1, x2 = self._crop_indices
            self.errcube_sci = (None if self._errcube_sci is None else self._errcube_sci[..., y1:y2, x1:x2])
            self.imcube_ref = self._imcube_ref[..., y1:y2, x1:x2]
            self.errcube_ref = (None if self._errcube_ref is None else self._errcube_ref[..., y1:y2, x1:x2])
            self.optzones = self._optzones[..., y1:y2, x1:x2]
            self.subzones = self._subzones[..., y1:y2, x1:x2]
            self.c_coron_sci = self._c_coron_sci - np.array([x1,y1])
            self.c_coron_ref = self._c_coron_ref - np.array([x1,y1])
            self.ny, self.nx = self.cropped_shape
            if self._imcube_css is not None:
                self.imcube_css = self._imcube_css[..., y1:y2, x1:x2]
            else:
                self.imcube_css = self._imcube_css

        else: # Set cropped data to non-cropped
            self.imcube_sci = self._imcube_sci
            self.errcube_sci = self._errcube_sci
            self.imcube_ref = self._imcube_ref
            self.errcube_ref = self._errcube_ref
            self.c_star = self._c_star
            self.c_coron_sci = self._c_coron_sci
            self.c_coron_ref = self._c_coron_ref
            self.optzones = self._optzones
            self.subzones = self._subzones
            self.ny, self.nx = self._ny, self._nx
            self._crop_indices = [0, self._ny+1, 0, self._nx+1]
            self.cropped_shape = cropped_shape
            self.imcube_css = self._imcube_css
    
        if self.convolver is not None:
            self.convolver.set_crop(cropped_shape)


    def run_rdi(self, save_products=False, return_res_only=False,
                forward_model=False, collapse_rolls=True, derotate=True,
                **extra_rdi_settings):
        """
        Runs winnie.rdi.rdi_residuals using settings stored in the SpaceRDI
        object's rdi_settings property. Returns a SpaceReduction object
        containing the residuals and other products. If save_products is True,
        the products are saved to a FITS file in the output directory specified
        in the input SpaceKLIP database object.

        Parameters
        ----------
        save_products: bool
            If True, saves the output products to a FITS file.
        
        return_res_only: bool
            If True, directly returns the result of rdi_residuals (useful for
            debugging or other advanced analysis).

        forward_model: bool
            If True, runs forward modeling on the circumstellar model set using
            set_circumstellar_model and with the current RDI configuration.

        collapse_rolls: bool
            If True, returns the residuals for each distinct pointing
            (derotated if derotate=True). In most cases, this just means
            returning both of the PSF-subtracted exposures. However, for NIRCam
            dual channel obs where one filter is paired with two or more
            filters for the other channel (e.g., F210M + F335M followed by
            F210M + F444W), the result is multiple exposures per roll for a
            given filter. In this case, the residuals are averaged together for
            each roll.
        
        derotate: bool
            If True, derotates the residuals to a common north up orientation.
            If False, returns residuals in the original orientation.
        
        extra_rdi_settings:
            Additional settings to pass to winnie.rdi.rdi_residuals. If any of
            these is already contained in the rdi_settings attribute, an error
            will be raised.

        Returns
        -------
        products: SpaceReduction object
            Object containing the residuals and other products from the RDI
            procedure. If collapse_rolls is True, the object will contain
            residuals for each roll (products.rolls) as well as a
            median-combined residual image (products.im). If the prop_err
            attribute is True, the object will also contain the propagated
            error arrays for the residuals (products.err and
            products.err_rolls).
        """
        if self.concat is None:
            raise ValueError("""
                Prior to executing "run_rdi", you must load a concatenation
                using the load_concat method.
                             """)
        output_ext = copy(self.output_ext)
        reduc_label = copy(self.reduc_label)
        if forward_model:
            if (self.rdi_settings.get('coeffs_in', None) is not None) or (extra_rdi_settings.get('coeffs_in', None) is not None):
                raise ValueError("""
                    Forward modeling with run_rdi is not valid when using fixed
                    RDI coefficients. For classical RDI, the output from the
                    derotate_and_combine_cssmodel method is likely more
                    appropriate.
                    """)
            if self.imcube_css is None:
                raise ValueError("""
                    Prior to executing "run_rdi" with forward_model=True you
                    must first set a circumstellar model using the
                    set_circumstellar_model method.
                                 """)
            imcube_sci = self.imcube_css
            prop_err = False # Never propagate error when forward modeling
            output_ext = output_ext + '_fwdmod'
            reduc_label = f'FM {reduc_label}'
        else:
            imcube_sci = self.imcube_sci
            prop_err = self.prop_err
            
        pad_before_derot = self.pad_before_derot
        
        if not prop_err:# or return_res_only:
            err_hcube = err_hcube_ref = None
        else:
            err_hcube = self.errcube_sci[:, np.newaxis] 
            err_hcube_ref = self.errcube_ref[:, np.newaxis]
        
        if derotate:
            posangs = self._posangs_sci
        else:
            posangs = None
            
        res = rdi_residuals(hcube=imcube_sci[:, np.newaxis],
                            hcube_ref=self.imcube_ref[:, np.newaxis],
                            optzones=self.optzones, subzones=self.subzones,
                            posangs=posangs, cent=self.c_star,
                            use_gpu=self.use_gpu, ncores=self.ncores,
                            err_hcube=err_hcube,
                            err_hcube_ref=err_hcube_ref,
                            pad_before_derot=pad_before_derot,
                            show_progress=self.show_progress,
                            **self.rdi_settings, **extra_rdi_settings)
        
        if return_res_only:
            return res
                
        residuals, residuals_err, c_derot = res
        
        residuals = residuals[:,0] # dropping unused wavelength axis
        if residuals_err is not None: 
            residuals_err = residuals_err[:,0]
        
        if derotate:
            im_col, err_col = median_combine(residuals, residuals_err)
        else:
            im_col, err_col = None, None
            
        if collapse_rolls:
            im_rolls, err_rolls = [],[]
            uni_visit_ids, uni_visit_inds = np.unique(self._visit_ids_sci, return_index=True)
            uni_visit_ids = uni_visit_ids[np.argsort(uni_visit_inds)]
            for visit_id in uni_visit_ids:
                visit = self._visit_ids_sci == visit_id
                im_roll, err_roll = median_combine(residuals[visit], (residuals_err[visit] if prop_err else None))
                im_rolls.append(im_roll)
                err_rolls.append(err_roll)
            im_rolls = np.asarray(im_rolls)
            err_rolls = np.asarray(err_rolls) if prop_err else None
        else:
            im_rolls = err_rolls = None

        products = SpaceReduction(spacerdi=self,
                                  im=im_col,
                                  rolls=im_rolls,
                                  err=err_col,
                                  err_rolls=err_rolls,
                                  c_star_out=c_derot, 
                                  output_ext=output_ext,
                                  derotated=derotate,
                                  reduc_label=self.reduc_label)
        if save_products:
            if self.save_instance and not forward_model:
                extra_hdus = [self._to_fits()]
            else:
                extra_hdus = []
            try:
                products.save(overwrite=self.overwrite, extra_hdus=extra_hdus)
            except OSError:
                raise OSError("""
                      A FITS file for this output_ext + output_dir + concat
                      already exists! To overwrite existing files, set the
                      overwrite attribute for your Winnie SpaceRDI instance to
                      True. Alternatively, either change the output_ext
                      attribute for your SpaceRDI instance, or select a
                      different output directory when initializing your
                      SpaceKLIP database object.
                      """)
            if not forward_model:
                if self.concat in self.database.red and products.filename in self.database.red[self.concat]['FITSFILE']:
                    self.database.red[self.concat].remove_row(np.where(self.database.red[self.concat]['FITSFILE'] == products.filename)[0][0])
                self.database.read_jwst_s3_data(products.filename)
        return products

    
    def set_zones(self, optzones, subzones, exclude_opt_nans=True):
        """
        Set the optimization and subtraction zones for RDI PSF-subtraction. See
        winnie.rdi.rdi_residuals for more information on the format and
        function of optzones and subzones. 

        ___________
        Parameters:

        optzones: ndarray
            Optimization zones for the reduction. 
        subzones: ndarray
            Subtraction zones for the reduction.
        exclude_opt_nans: bool
            If True, excludes from the optimization zones any pixels that are
            NaN in either the science or reference data.

        Raises:

            ValueError: If some pixels are included in multiple subtraction zones.

        Notes: 

        - If the spatial dimensions of the zones match those of the uncropped
          data, the zones are directly assigned.
         
        - If the zones are already cropped, the corresponding uncropped zones
          are constructed from them.
        
        """
        if np.any(subzones.sum(axis=0) > 1):
            raise ValueError("Subtraction zones are invalid; some pixels are included in multiple subtraction zones.")
        
        # If spatial dims of zones match those of the uncropped data:
        if np.all(np.asarray(optzones.shape[-2:]) == np.asarray(self._imcube_sci.shape[-2:])):
            self._optzones = np.asarray(optzones)
            self._subzones = np.asarray(subzones)
            
            if self.cropped_shape is not None: # For changing zones after cropping
                y1, y2, x1, x2 = self._crop_indices
                self.optzones = self._optzones[..., y1:y2, x1:x2]
                self.subzones = self._subzones[..., y1:y2, x1:x2]
            else:
                self.optzones = self._optzones
                self.subzones = self._subzones
        # Else, assume zones are already cropped and construct corresponding uncropped zones from them
        # In this case, we should assume set_crop has already been run.
        else: 
            self.optzones = np.asarray(optzones)
            self.subzones = np.asarray(subzones)
            y1, y2, x1, x2 = self._crop_indices
            
            self._optzones = np.zeros((self._ny, self._nx), dtype='bool')
            self._subzones = np.zeros((self._ny, self._nx), dtype='bool')
            self._optzones[..., y1:y2, x1:x2] = self.optzones
            self._subzones[..., y1:y2, x1:x2] = self.subzones

        if exclude_opt_nans:
            # First apply to uncropped zones
            nans_sci = np.any(np.isnan(self._imcube_sci), axis=0)
            nans_ref = np.any(np.isnan(self._imcube_ref), axis=0)
            self._optzones = np.where(np.logical_or(nans_sci, nans_ref), False, self._optzones)

            # and now the cropped zones (if the cropped data exists)
            if self.cropped_shape is not None:
                nans_sci = np.any(np.isnan(self.imcube_sci), axis=0)
                nans_ref = np.any(np.isnan(self.imcube_ref), axis=0)
                self.optzones = np.where(np.logical_or(nans_sci, nans_ref), False, self.optzones)
            else:
                self.optzones = self._optzones


    def update_annular_zones(self, exclude_opt_nans=True):
        """
        Set annular RDI zones based on current values for self.r_opt and
        self.r_sub.

        ___________
        Parameters:

        exclude_opt_nans: bool
            If True, excludes from the optimization zones any pixels that are
            NaN in either the science or reference data.
        """
        optzones, subzones = build_annular_rdi_zones(self._nx, self._ny, self._c_star, r_opt=self.r_opt, r_sub=self.r_sub, pxscale=self.pxscale)
        self.set_zones(optzones, subzones, exclude_opt_nans=exclude_opt_nans)
        self._check_smoothed_nans()


    def report_current_config(self, show_plots=None):
        """
        Print a summary of the current configuration of the SpaceRDI instance.
        If show_plots is True, also plots the current optimization and
        subtraction zones for the first science exposure.
        """
        if show_plots is None:
            show_plots = self.show_plots
        print(self.concat)
        print(f'Science data:   {self.imcube_sci.shape[0]} exposures of shape ({self.ny},{self.nx})')
        print(f'Reference data: {self.imcube_ref.shape[0]} exposures of shape ({self.ny},{self.nx})\n')
        print(f'RDI Settings:')
        print(f'Mode: {self.reduc_label}')
        for key in self.rdi_settings:
            if isinstance(self.rdi_settings[key], np.ndarray):
                desc = f'{type(self.rdi_settings[key])} of shape {self.rdi_settings[key].shape}'
            else:
                desc = self.rdi_settings[key]
            print(f"'{key}': {desc}")
        print(f"Extension for output files: '{self.output_ext}'")
        print(f"{len(self.optzones)} optimization zone(s)")
        print(f"{len(self.subzones)} subtraction zone(s)")
        if show_plots:
            fig,axes = quick_implot(np.array([np.where(self.optzones, self.imcube_sci[0], np.nan),
                                              np.where(self.subzones, self.imcube_sci[0], np.nan)]
                                             ).transpose((1,0,2,3)),
                                    norm=mpl.colors.LogNorm,
                                    norm_kwargs=dict(clip=True),
                                    clim='0.001*99.99%, 99.99%',
                                    extent=mpl_centered_extent((self.ny, self.nx), self.c_star, self.pxscale),
                                    show_ticks=True,
                                    show=False, panelsize=(4,4))
            axes[0].set_title('Optimization Zones')
            axes[1].set_title('Subtraction Zones')
            for i,ax in enumerate(axes):
                for axis in [ax.xaxis, ax.yaxis]:
                    axis.set_major_formatter("${x:0.0f}''$")
            plt.show()
    
    
    def set_fixed_rdi_settings(self, **settings):
        """
        Set RDI settings that will be added to (and overwrite where duplicated)
        any settings managed by set_presets, rdi_presets, hpfrdi_presets, or
        mcrdi_presets. These must be re-set if a new concatenation is loaded.

        Some settings that may be useful:
        
        ref_mask: ndarray
            2D boolean array of shape (len(optzones), len(self.imcube_ref))
            that indicates which reference images should be considered for
            which optimization regions. E.g., if ref_mask[i,j] is False, then
            for the ith optimization zone (optzones[i]), the jth reference
            image (imcube_ref[j]) will NOT be used for construction of the PSF
            model. This can be useful if some reference exposures have
            anomalous features that make them problematic for some regions
            while still being suitable for others; e.g., an image with a bright
            background source near the edge of the FOV may still be useful for
            nulling the PSF near the inner working angle.

        zero_nans: bool
            If True, any nans in the optimization zones will be replaced with
            zeros for the procedure.

        return_coeffs: bool
            If True, returns only the array of PSF model coefficients.

        coeffs_in: ndarray
            If provided, these coefficients will be used to construct the PSF
            model instead of computing coefficients.

        opt_smoothing_fn: callable or None
            If not None, this argument indicates the function with which to
            smooth the sequences. This should be a function that takes a
            hypercube along with some keyword arguments and returns a smoothed
            hypercube, i.e.: hcube_filt = opt_smoothing_fn(hcube,
            **opt_smoothing_kwargs).
            
        opt_smoothing_kwargs: dict
            If opt_smoothing_fn is not None, arguments to pass to
            opt_smoothing_fn when it is called.

        large_arrs: bool
            If True (and if use_gpu=False), PSF model reconstruction will use
            smaller parallelized calculations. If False, larger vectorized
            calculations will be used instead. See the docstring for
            winnie.rdi.reconstruct_psf_model_cpu for more information. Default
            is False.
        """
        self.fixed_rdi_settings = settings
        self.rdi_settings.update(self.fixed_rdi_settings)
    
    
    def set_presets(self, presets={}, output_ext='psfsub', reduc_label='Custom RDI (Winnie)'):
        """
        Generic method to quickly assign a set of arguments to use for
        winnie.rdi.rdi_residuals, while also setting the extension for saved
        files, repopulating any settings in self.fixed_rdi_settings, and
        reporting the configuration if verbose is True.
        """
        self.output_ext = output_ext
        self.rdi_settings = presets
        self.rdi_settings.update(self.fixed_rdi_settings)
        self.reduc_label = reduc_label
        self._check_smoothed_nans()
        if self.verbose:
            self.report_current_config()
    
    
    def rdi_presets(self, output_ext='rdi_psfsub', reduc_label='RDI (Winnie)'):
        """
        Set presets to perform a standard RDI reduction.
        ___________
        Parameters:
            output_ext (str, optional): Output file extension for FITS
                products. Defaults to 'rdi_psfsub'.
        """
        self.set_presets(presets={}, output_ext=output_ext, reduc_label=reduc_label)
    
    
    def hpfrdi_presets(self, filter_size=None, filter_size_adj=1, output_ext='hpfrdi_psfsub', reduc_label='HPFRDI (Winnie)'):
        """
        Set presets for High-Pass Filtering RDI (HPFRDI), in which coefficients
        are computed by comparing high-pass filtered science and reference
        data.

        ___________
        Parameters:
            filter_size (float, optional): Size of the high-pass filter. If not
                provided, it defaults to the value of self._sigma.
            filter_size_adj (float, optional): Adjustment factor for the filter
                size. Defaults to 1. output_ext (str, optional): Output file
                extension for FITS products. Defaults to 'hpfrdi_psfsub'.
        """
        if filter_size is None:
            filter_size = self._sigma
        presets = {}
        presets['opt_smoothing_fn'] = high_pass_filter_sequence
        presets['opt_smoothing_kwargs'] = dict(filtersize=filter_size_adj*filter_size)
        self.set_presets(presets=presets, output_ext=output_ext, reduc_label=reduc_label)
    
    
    def mcrdi_presets(self, output_ext='mcrdi_psfsub', reduc_label='MCRDI (Winnie)'):
        """
        Set presets for Model Constrained RDI (MCRDI), in which coefficients
        are computed by comparing reference data to science data from which an
        estimate of the circumstellar scene has been subtracted.

        ___________
        Parameters:
            output_ext (str, optional): Output file extension for FITS
                products. Defaults to 'mcrdi_psfsub'.

        Raises:
            ValueError: If a circumstellar model has not been set using the
                set_circumstellar_model method.
        """
        if self.imcube_css is None:
            raise ValueError(
                """
                Prior to executing mcrdi_presets,
                you must first set a circumstellar model using 
                set_circumstellar_model.
                """)

        self.set_presets(presets={'hcube_css': self.imcube_css[:, np.newaxis]},
                         output_ext=output_ext, reduc_label=reduc_label)
    

    def prepare_convolution(self, source_spectrum=None, reference_index=0, fov_pixels=151, osamp=2,
                            output_ext='psfs', prefetch_psf_grid=True, recalc_psf_grid=False,
                            convolver_basedir=None, convolver_subdir='psfgrids', fetch_opd_by_date=True, 
                            grid_fn=generate_lyot_psf_grid, grid_kwargs={}, 
                            grid_inds_fn=get_jwst_psf_grid_inds, grid_inds_kwargs={},
                            transmission_map_fn=get_jwst_coron_transmission_map,
                            transmission_map_kwargs={}):
        """
        Sets up the SpaceRDI instance to enable convolution of circumstellar
        scene models by preparing an instance of the SpaceConvolution class and
        assigning it to the convolver attribute. Arguments are stored in the
        convolver_args attribute, where they can be altered as preferred before
        loading a new concatenation.

        ___________
        Parameters:

        source_spectrum : synphot.spectrum.SourceSpectrum, optional
            The source spectrum to use for generating PSFs for convolution.
        reference_index : int, optional
            The index of the science exposure that will be used to initialize
            the WebbPSF instrument object with WebbPSF's
            setup_sim_to_match_file function.
        fov_pixels : int, optional
            The number of pixels in the field of view. Default is 151.
        osamp : int, optional
            The oversampling factor to use for generating PSFs. Default is 2.
        output_ext : str, optional
            The file extension for the output PSFs. Default is 'psfs'.
        prefetch_psf_grid : bool, optional
            Whether to fetch the PSF grid immediately after a concatenation is
            loaded, or wait for manual fetching by the user. Default is True.
        recalc_psf_grid : bool, optional
            Whether to recalculate the PSF grid if it has already been
            generated and saved. Default is False.
        psfgrids_output_dir : str, optional
            The directory for saving the PSF grids (relative to the output
            direction for the initial SpaceKLIP database). Default is
            'psfgrids'.
        fetch_opd_by_date : bool, optional
            Whether to fetch the OPD map for the date of the observations or
            use the default. Default is True.
        grid_fn : callable, optional
            The function to use for generating the PSF grid. Default is
            winnie.convolution.generate_lyot_psf_grid.
        grid_kwargs : dict, optional
            Additional keyword arguments to be passed to grid_fn.
        grid_inds_fn : callable, optional
            The function to use for matching model pixels to PSF grid samples.
            Default is winnie.convolution.get_jwst_psf_grid_inds.
        grid_inds_kwargs : dict, optional
            Additional keyword arguments to be passed to grid_inds_fn.
        transmission_map_fn : callable, optional
            The function to use for generating the coronagraph transmission
            map. Default is winnie.convolution.get_jwst_coron_transmission_map.
        transmission_map_kwargs : dict, optional
            Additional keyword arguments to be passed to transmission_map_fn.

        Notes:
        - grid_fn should return three objects: 1) a 3D array of PSF samples, 2)
          the 2D array of shape (2,N) containing the polar coordinates of those
          samples (relative to the coronagraph center; units of [arcsec,
          degrees]), and 3) the 2D array of shape (2,N) containing the
          cartesian coordinates of those samples (relative to the coronagraph
          center; units of [arcsec, arcsec]). It should take a WebbPSF
          instrument object as the first argument, and must have the following
          signature: 
                grid_fn(inst_webbpsf, source_spectrum=None, shift=None,
                osamp=2, fov_pixels=151, show_progress=True, **grid_kwargs)
          See winnie.convolution.generate_lyot_psf_grid for documentation
          regarding these arguments.

        - grid_inds_fn should return a 2D integer array of shape (ny*osamp,
          nx*osamp) where each pixel provides the index of the PSF sample that
          should be used for convolution of that pixel. It should have the
          following signature:
                grid_inds_fn(c_coron, psf_offsets_polar, osamp=2, shape=None,
                pxscale=None, **grid_inds_kwargs)
          See winnie.convolution.get_jwst_psf_grid_inds for documentation
          regarding these arguments.

        - transmission_map_fn should return a 2D array of shape (ny*osamp,
          nx*osamp) where each pixel provides the transmission value of the
          coronagraph at that location. It should have the following signature:
                transmission_map_fn(inst_webbpsfext, c_coron, osamp=2,
                shape=None, **transmission_map_kwargs)
          See winnie.convolution.get_jwst_coron_transmission_map for
          documentation regarding these arguments.

        - If even more customization is needed: set prefetch_psf_grid to False
          here and use the set_custom_grid method for the convolver (accessed
          via the convolver attribute for the SpaceRDI object) to directly
          set the PSF grid and related objects.
        """
        convolver_args = dict(reference_index=reference_index, fov_pixels=fov_pixels, osamp=osamp, output_ext=output_ext,
                              prefetch_psf_grid=prefetch_psf_grid, recalc_psf_grid=recalc_psf_grid, grid_fn=grid_fn, grid_kwargs=grid_kwargs,
                              grid_inds_fn=grid_inds_fn, grid_inds_kwargs=grid_inds_kwargs, transmission_map_fn=transmission_map_fn,
                              transmission_map_kwargs=transmission_map_kwargs)
        
        if self.convolver is None or convolver_args != self.convolver_args: # If the convolver is not already set up or the args have changed
            if convolver_basedir is None:
                convolver_basedir = self.output_dir
            self.convolver = SpaceConvolution(database=self.database, source_spectrum=source_spectrum, ncores=self.ncores, use_gpu=self.use_gpu,
                                              verbose=self.verbose, show_plots=self.show_plots, show_progress=True, overwrite=self.overwrite,
                                              output_basedir=convolver_basedir, output_subdir=convolver_subdir, fetch_opd_by_date=fetch_opd_by_date,
                                              pad_data=self.pad_data, efficient_saving=self.efficient_saving)
            self.convolver_args = convolver_args

        if self.concat != self.convolver.concat:
            self.convolver.load_concat(self.concat, cropped_shape=self.cropped_shape, coron_offsets=self._coron_offsets, **self.convolver_args)

    
    def set_circumstellar_model(self, model_cube=None, model_files=None, model_dir=None, model_ext='cssmodel',
                                raw_model=None, raw_model_pxscale=None, raw_model_center=None):
        """
        Sets a circumstellar scene model to be used in various procedures
        (e.g., RDI forward modeling or MCRDI.) 

        ______________ 
        Input options:

        raw_model, raw_model_pxscale, and raw_model_center: a north-up image of
            the unconvolved circumstellar scene, the pixel scale of the model
            (assumed arcsec/pixel if unitless, else should be astropy units
            castable to arcsec/pixel), and the center position for the raw
            model in pixel coordinates ([x,y]; i.e., where the star would be
            located in the image). The raw model will be convolved with the
            prepared convolution setup.
        
        model_cube: a 3D array matching the shape of either the cropped or
            uncropped science observations and containing PSF-convolved model
            images at the appropriate roll angles to match the data.

        model_files: a list of filenames to load into a 3D array as model_cube.

        model_ext and model_dir: the filename extension and directory to use
            for populating a list of model files. model_ext defaults to
            'cssmodel' and model_dir defaults to the output directory of the
            SpaceKLIP database object.

        If none of these are provided, the method will attempt to load saved
        model images from the output directory of the SpaceKLIP database object
        assuming an extension of 'cssmodel'. 
        
        If multiple options are provided, the list above indicates descending
        priority.
        """
        if raw_model is not None:
            if self.convolver is None or self.convolver.psfs is None:
                    raise ValueError("""
                        To run set_circumstellar_model with a raw model as
                        input, you must first execute prepare_convolution.
                                     """)
            model_cube = self.convolver.convolve_model(raw_model, pxscale_in=raw_model_pxscale, c_star_in=raw_model_center)

        if model_cube is None:
            if model_files is None:
                if model_dir is None:
                    model_dir = self.output_dir
                model_files = np.array([model_dir+os.path.basename(os.path.normpath(f)).replace(self.data_ext, model_ext) for f in self._files_sci])
            model_cube = np.array([fits.getdata(f, ext=1).squeeze() for f in model_files])
        
        y1,y2,x1,x2 = self._crop_indices
        if np.all(np.asarray(model_cube.shape[-2:]) == np.asarray(self.imcube_sci.shape[-2:])): # Model provided in cropped shape
            self._imcube_css = np.zeros_like(self._imcube_sci)
            self._imcube_css[..., y1:y2, x1:x2] = model_cube

        else: # Model provided in uncropped shape
            self._imcube_css = model_cube

        self._imcube_css[np.isnan(self._imcube_sci)] = np.nan
        self.imcube_css = self._imcube_css[..., y1:y2, x1:x2]


    def save_circumstellar_model(self, output_ext='cssmodel', model_dict={}):
        """
        Saves the current circumstellar model images to FITS files so they can
        be loaded later. The files will be saved to the output directory of the
        SpaceRDI database object, with the same filename as the science data
        files but with the input data_ext replaced with the specified
        output_ext ('cssmodel' by default). Any entries in model_dict will be
        appended to the image header. If a circumstellar model is not set, this
        method will raise a ValueError.
        """
        if self.imcube_css is None:
            raise ValueError(
                    """
                    Prior to executing save_circumstellar_model, you must first set a
                    circumstellar model using set_circumstellar_model.
                    """)
        for i,f in enumerate(self._files_sci):
            fout = self.output_dir+os.path.basename(os.path.normpath(f)).replace(self.data_ext, output_ext)
            if fout == f:
                raise ValueError(f"""
                    The output file path for the circumstellar model is the
                    same as the input data file: 
                        {fout}
                    This is either because the specified data_ext is not
                    contained in the data filename, or because the value for
                    data_ext is the same as the specified output_ext. Choose a
                    different output_ext or change the data_ext attribute for
                    your SpaceRDI instance to avoid overwriting the input data
                    files.
                    """)
            with fits.open(f) as hdul:
                hdul_out = fits.HDUList([hdul[0], hdul[1]])
                h1 = hdul_out[1].header
                h1.update(NAXIS1=self.nx, NAXIS2=self.ny, CRPIX1=self.c_star[0]+1, CRPIX2=self.c_star[1]+1)
                h1.update(model_dict)
                hdul_out[1].header = h1
                hdul_out[1].data = self.imcube_css[i]
                hdul_out.writeto(fout, overwrite=self.overwrite)


    def derotate_and_combine_circumstellar_model(self, collapse_rolls=True, output_ext='cssmodel', save_products=False):
        """
        Derotates the current circumstellar model and averages over all rolls;
        provides output in a SpaceReduction object to match the output of
        run_rdi. If a circumstellar model is not set, this method will raise a
        ValueError.
        """
        if self.imcube_css is None:
            raise ValueError(
                    """
                    Prior to executing derotate_and_combine_circumstellar_model, you must
                    first set a circumstellar model using
                    set_circumstellar_model.
                    """)
        
        pad_before_derot = self.pad_before_derot
        
        csscube = self.imcube_css
        
        if pad_before_derot:
            residuals, c_derot = pad_and_rotate_hypercube(csscube, -self._posangs_sci,
                                                          cent = self.c_star, ncores = self.ncores, 
                                                          use_gpu = self.use_gpu, cval0=np.nan)
        else:
            residuals, c_derot = rotate_hypercube(csscube, -self._posangs_sci,
                                                  cent = self.c_star, ncores = self.ncores, 
                                                  use_gpu = self.use_gpu, cval0=np.nan), self.c_star
                    
        im_col, _ = median_combine(residuals)
        
        if collapse_rolls:
            im_rolls = []
            uni_visit_ids, uni_visit_inds = np.unique(self._visit_ids_sci, return_index=True)
            uni_visit_ids = uni_visit_ids[np.argsort(uni_visit_inds)]
            for visit_id in uni_visit_ids:
                visit_filt = self._visit_ids_sci == visit_id
                im_roll, _ = median_combine(residuals[visit_filt])
                im_rolls.append(im_roll)
            im_rolls = np.asarray(im_rolls)
        else:
            im_rolls = None
        
        products = SpaceReduction(spacerdi=self, im=im_col, rolls=im_rolls,
                                  c_star_out=c_derot, output_ext=output_ext)
        
        if save_products:
            try:
                products.save(overwrite=self.overwrite)
            
            except OSError:
                raise OSError("""
                      A FITS file for this output_ext + output_dir + concat
                      already exists! To overwrite existing files, set the
                      overwrite attribute for your Winnie SpaceRDI instance to
                      True. Alternatively, either change the output_ext
                      attribute for your SpaceRDI instance, or select a
                      different output directory when initializing your
                      SpaceKLIP database object.
                      """)

        return products


    def make_derot_coron_maps(self, collapse_rolls=False, save_products=False):
        coron_tmaps = np.zeros_like(self.imcube_sci)
        maskfiles = self.database.obs[self.concat]['MASKFILE'][self.database.obs[self.concat]['TYPE']=='SCI']
        inst = None
        for i, c_coron in enumerate(self.c_coron_sci):
            if maskfiles[i] == 'NONE' or fits.getval(maskfiles[i], 'NAXIS', ext=1) == 0:
                if self.convolver is not None and self.convolver.coron_tmaps_osamp is not None: # Use the already-loaded transmission map
                    coron_tmaps[i] = webbpsf_ext.image_manip.frebin(self.convolver.coron_tmaps_osamp[i], scale=1./self.convolver.osamp, total=False)
                else: # generate a transmission map image
                    if inst is None: # On first iteration, initialize our WebbPSF-ext object
                        # Swap to prelaunch equivalent mask names for WebbPSF-ext
                        if self._pupil_mask.endswith('RND'):
                            pupil_mask_ext = 'CIRCLYOT'
                        elif self._pupil_mask.endswith('WB'):
                            pupil_mask_ext = 'WEDGELYOT'
                        else:
                            pupil_mask_ext = self._pupil_mask
                        if self.database.obs[self.concat]['INSTRUME'][0] == 'NIRCAM':
                            inst = webbpsf_ext.NIRCam_ext(filter=self.filt, pupil_mask=pupil_mask_ext, image_mask=self._image_mask, oversample=3)
                        else:
                            inst = webbpsf_ext.MIRI_ext(filter=self.filt, pupil_mask=pupil_mask_ext, image_mask=self._image_mask, oversample=3)
                        inst.aperturename = self._aperturename
                    coron_tmaps[i] = get_jwst_coron_transmission_map(inst, c_coron, osamp=3, shape=(self.ny, self.nx), return_oversample=False, nd_squares=True)
            else: # Load the provided transmission map
                coron_tmap = fits.getdata(maskfiles[i])
                if self.pad_data is not None:
                    coron_tmap = np.pad(coron_tmap, self._imc_padding[1:], constant_values=np.nan)
                coron_tmaps[i] = coron_tmap
        if self.pad_before_derot:
            coron_tmaps_derot, c_derot = pad_and_rotate_hypercube(coron_tmaps, -self._posangs_sci,
                                                          cent=self.c_star, ncores=self.ncores, 
                                                          use_gpu=self.use_gpu, cval0=np.nan)
        else:
            coron_tmaps_derot, c_derot = rotate_hypercube(coron_tmaps, -self._posangs_sci,
                                                  cent=self.c_star, ncores=self.ncores, 
                                                  use_gpu=self.use_gpu, cval0=np.nan), self.c_star
                    
        im_col, _ = median_combine(coron_tmaps_derot)
        
        if collapse_rolls:
            im_rolls = []
            uni_visit_ids, uni_visit_inds = np.unique(self._visit_ids_sci, return_index=True)
            uni_visit_ids = uni_visit_ids[np.argsort(uni_visit_inds)]
            for visit_id in uni_visit_ids:
                visit_filt = self._visit_ids_sci == visit_id
                im_roll, _ = median_combine(coron_tmaps_derot[visit_filt])
                im_rolls.append(im_roll)
            im_rolls = np.asarray(im_rolls)
        else:
            im_rolls = None

        products = SpaceReduction(spacerdi=self, im=im_col, rolls=im_rolls,
                                  c_star_out=c_derot, output_ext='psfmask')

        products.filename = products.filename.replace('psfmask_i2d.fits', 'psfmask.fits')

        if save_products:
            products.save(overwrite=True)
        return products

    
    def make_css_subtracted_residuals(self, data_reduc=None, model_reduc=None, save_products=True):
        if data_reduc is None:
            data_reduc = self.run_rdi(save_products=False)
            reduc_label = f'CSS-sub {self.reduc_label}'
        else:
            reduc_label = f'CSS-sub {data_reduc.reduc_label}'
        if model_reduc is None:
            model_reduc = self.run_rdi(save_products=False, forward_model=True)
        output_ext = data_reduc.output_ext+'_csssub'
        products = SpaceReduction(spacerdi=self, im=data_reduc.im-model_reduc.im, rolls=data_reduc.rolls-model_reduc.rolls, c_star_out=data_reduc.c_star, output_ext=output_ext, reduc_label=reduc_label)
        if save_products:
            try:
                products.save(overwrite=self.overwrite)
            except OSError:
                raise OSError("""
                      A FITS file for this output_ext + output_dir + concat
                      already exists! To overwrite existing files, set the
                      overwrite attribute for your Winnie SpaceRDI instance to
                      True. Alternatively, either change the output_ext
                      attribute for your SpaceRDI instance, or select a
                      different output directory when initializing your
                      SpaceKLIP database object.
                      """)
            if self.concat in self.database.red and products.filename in self.database.red[self.concat]['FITSFILE']:
                self.database.red[self.concat].remove_row(np.where(self.database.red[self.concat]['FITSFILE'] == products.filename)[0][0])
            self.database.read_jwst_s3_data(products.filename)
        return products


    def make_css_subtracted_stage2_products(self, subdir='css_subtracted', update_database=False):
        """
        Creates a new subdirectory ('css_subtracted' by default) and saves the
        science data with the current circumstellar model subtracted, along with
        reference data and maskfiles. Optionally updates the entries in the
        SpaceKLIP database. The outputs of this method can then be used as inputs
        for a SpaceKLIP reduction, if preferred — e.g., to extract candidate
        photometry. 
        
        WARNING: candidate photometry extracted with SpaceKLIP will only be valid
        if the candidate source was excluded from the optimization zones used for
        Winnie PSF subtraction, as well as from the region used for optimizing the
        disk model. Otherwise, the source will affect the resulting disk model and
        the source photometry will be biased.
        """
        if self.imcube_css is None:
            raise ValueError(
                    """
                    Prior to executing make_css_subtracted_stage2_products, you
                    must first set a circumstellar model using
                    set_circumstellar_model.
                    """)
        output_dir = self.output_basedir+subdir+'/'
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        for i,f in enumerate(self._files_sci):
            j = np.where(self.database.obs[self.concat]['FITSFILE']==f)[0][0]
            im_css = self.imcube_css[i][self._imc_padding[1][0]:-self._imc_padding[1][1], self._imc_padding[2][0]:-self._imc_padding[2][1]]
            fcss_sub = output_dir+os.path.basename(os.path.normpath(f))
            with fits.open(f) as hdul:
                hdul['SCI'].data -= im_css
                hdul.writeto(fcss_sub, overwrite=True)
            fmask = self.database.obs[self.concat]['MASKFILE'][j]
            if fmask is not None:
                fmask_out = output_dir+os.path.basename(os.path.normpath(fmask))
                with fits.open(fmask) as hdul:
                    hdul.writeto(fmask_out, overwrite=True)
            else:
                fmask_out = None
            if update_database:
                self.database.update_obs(self.concat, j, fcss_sub, fmask_out)
        for i,f in enumerate(self._files_ref):
            j = np.where(self.database.obs[self.concat]['FITSFILE']==f)[0][0]
            fcss_sub = output_dir+os.path.basename(os.path.normpath(f))
            with fits.open(f) as hdul:
                hdul.writeto(fcss_sub, overwrite=True)
                
            fmask = self.database.obs[self.concat]['MASKFILE'][j]
            if fmask is not None:
                fmask_out = output_dir+os.path.basename(os.path.normpath(fmask))
                with fits.open(fmask) as hdul:
                    hdul.writeto(fmask_out, overwrite=True)
            else:
                fmask_out = None
            if update_database:
                self.database.update_obs(self.concat, j, fcss_sub, fmask_out)
        

    def jackknife_references(self, derotate=False, exclude_by_visitid=True, show_progress=True):
        """
        If exclude_by_visitid is True, carries out reductions using jackknife
        resampling over each unique visit ID, leaving out all reference images
        with a given visit ID in turn. For a reference target with multiple
        rolls, this will exclude only one roll at a time. For a reference
        target with multiple dithers, this will exclude all of the dithers at
        once. For nVis visit IDs, this results in nVis reductions.

        If exclude_by_visitid is False, for nRef reference images, carries out
        jackknife resampling by running nRef reductions leaving out each
        reference image in turn.

        The former is more useful for identifying the images contributing
        off-axis sources to the PSF model, while the latter may be useful for
        identifying uncorrected cosmic rays. 
        
        ________ 
        Returns:
            inds_JK: ndarray
                1D array of indices (length nRef), where entries of i indicate
                the reference images excluded from rescube_JK[i].
            rescube_JK: ndarray
                4D array of shape (nJK, nRolls, ny, nx), where nJK=nVis if
                exclude_by_visitid and nJK=nRef otherwise, containing the
                residuals from each jackknife reduction. If derotate is True,
                the residuals are derotated

        if self.prop_err is True, also returns:
            errcube_JK: ndarray
                4D array of propagated uncertainties from the ERR FITS
                extensions for each jackknife reduction. 
        """
        nRef = self.imcube_ref.shape[0]
        nOpt = self.optzones.shape[0]

        init_refmask = self.rdi_settings.get('ref_mask')

        jk_refmask = np.ones((nOpt, nRef), dtype='bool')

        if exclude_by_visitid:
            uVisIDs, inds_JK = np.unique(self._visit_ids_ref, return_inverse=True)
            nJK = len(uVisIDs)
        else:
            inds_JK = np.arange(nRef)
            nJK = nRef
        
        rescube_JK = np.zeros((nJK, *self.imcube_sci.shape), self.imcube_sci.dtype)
        if self.prop_err:
            errcube_JK = np.zeros_like(rescube_JK)

        for i in (tqdm(np.unique(inds_JK), leave=False) if show_progress else np.unique(inds_JK)):
            jk_refmask[:,:] = np.invert(inds_JK == i)
            self.rdi_settings['ref_mask'] = jk_refmask
            hcube_res, err_hcube_res, _ = self.run_rdi(derotate=derotate, return_res_only=True)
            rescube_JK[i] = hcube_res[:,0]
            if self.prop_err:
                errcube_JK[i] = err_hcube_res[:,0]

        out = [inds_JK, rescube_JK]
        if self.prop_err:
            out.append(errcube_JK)
        if init_refmask is None:
            del self.rdi_settings['ref_mask']
        else:
            self.rdi_settings['ref_mask'] = init_refmask
        return out


    def run_deconvolution(self, reduc_in=None, num_iter=500, epsilon='auto', auto_eps_errtol=1e-3, return_iters=None,
                          init_from_reduc=False, excl_mask_in=None, output_suffix='deconv', save_products=False, show_progress=True):
        """
        Perform deconvolution using a variant of the Richardson Lucy algorithm,
        adapted for shift-variant coronagraphic observations. Requires that
        prepare_convolution has been run. 
        
        ___________ 
        Parameters:
        
        reduc_in: SpaceReduction, optional 
            If provided, it must be a derotated reduction for the currently
            loaded concatenation. If not provided, an HPFRDI reduction will be
            run using the current opt/sub zones.

        num_iter: int, optional
            The number of deconvolution iterations to run. Default is 500.

        epsilon: float or 'auto', optional
            The correction threshold for deconvolution. If not None, no
            corrections will be applied to any pixels with absolute values less
            than epsilon. Larger values of epsilon will decrease noise
            (resulting from division by values very near zero) in final
            results, but may also decrease the accuracy of the deconvolution.
            If 'auto' and if self.prop_err, epsilon will be set to
            auto_eps_errtol times the median pixel error from reduc.err_rolls.
            If 'auto' but not self.prop_err, epsilon will be set to
            auto_eps_errtol times the median absolute deviation for
            reduc.rolls. Default is 'auto'.

        auto_eps_errtol: float, optional
            When epsilon is 'auto', this factor will be used to set the
            threshold for deconvolution; effectively the number of standard
            deviations below which no corrections should be considered. Default
            is 1e-3.

        return_iters: array-like, optional
            If provided, an array of integers indicating the iteration numbers
            that should be returned (in addition to the final result). Helpful
            for diagnosing issues (and for making neat animations!). Default is
            None.
            
        init_from_reduc: bool, optional
            If True, the 0th iteration for deconvolution is the input image
            being deconvolved. Otherwise, a uniform image of values equaling
            the median of all positive pixels is used. Can speed convergence
            but is less robust to artifacts in the input image. Should be used
            only with particularly clean reductions. Default is False. 

        excl_mask_in: ndarray, optional
            A boolean mask of the same shape as the input image, where values
            of True indicate pixels that should NOT be corrected during
            deconvolution (in addition to any NaNs in the original image and
            any values excluded by epsilon). Default is None.

        output_suffix: str, optional
            String appended to the reduction's output_ext attribute for output
            of deconvolved results. Default is 'deconv'.

        save_products: bool, optional
            Whether to save the deconvolved products to FITS files. Default is
            False.

        show_progress: bool, optional
            Whether to show progress bars during the deconvolution. Default is
            True.

        _____
        Notes:

        If custom functions are being used for transmission_map_fn and/or
        grid_inds_fn, they must take additional arguments: the position angle
        by which to derotate and and the position of the star. See
        convolution.get_jwst_psf_grid_inds and
        convolution.get_jwst_coron_transmission_map for reference.

        ________
        Returns:
            products: SpaceReduction
                A SpaceReduction object containing the deconvolved image and
                residuals. If save_products is True, the result will be saved
                to a FITS file as well.
            
            if reduc_in is None:
                reduc: SpaceReduction
                    The HPFRDI reduction used as input for the deconvolution.

            if return_iters is not None:
                deconv_iters: ndarray
                    A 3D array of shape (len(return_iters), ny, nx) containing
                    the deconvolved images at the iteration numbers specified by
                    return_iters.
        """
        if reduc_in is None:
            self.hpfrdi_presets()
            reduc = self.run_rdi(save_products=False)
        else:
            reduc = reduc_in

        reduc_label = f'Deconv {reduc.reduc_label}'

        if self.convolver is None or self.convolver.psfs is None:
                raise ValueError("""
                    To run deconvolution, you must first execute
                    prepare_convolution.
                                """)

        deconv_psfs = np.array([webbpsf_ext.image_manip.frebin(rotate_image(self.convolver.psfs, -posang, cval0=0.), scale=1/self.convolver.osamp, total=True) for posang in self._posangs_sci])
        deconv_coron_tmaps = np.zeros((len(self.c_coron_sci), self.ny, self.nx), dtype=self.convolver.coron_tmaps_osamp.dtype)
        deconv_psf_inds = np.zeros((len(self.c_coron_sci), self.ny, self.nx), dtype=self.convolver.psf_inds_osamp.dtype)
        for i,c_coron in enumerate(self.c_coron_sci):
            deconv_psf_inds[i] = self.convolver.grid_inds_fn(c_coron, self.convolver.psf_offsets_polar, 1, shape=(self.ny, self.nx), 
                                                                pxscale=self.convolver.pxscale, posang=self._posangs_sci[i], c_star=reduc.c_star)

            deconv_coron_tmaps[i] = self.convolver.transmission_map_fn(self.convolver.inst_webbpsfext, c_coron, return_oversample=False,
                                                                            osamp=self.convolver.osamp, nd_squares=False, 
                                                                            shape=(self.ny, self.nx), posang=self._posangs_sci[i],
                                                                            c_star=reduc.c_star)

        if epsilon == 'auto':
            if reduc.err_rolls is None:
                epsilon = auto_eps_errtol*np.nanmedian(reduc.err_rolls)
            else:
                epsilon = auto_eps_errtol*nan_median_absolute_deviation(reduc.rolls)

        rolls_deconv = np.zeros_like(reduc.rolls)
        if return_iters is not None:
            deconv_iters = np.zeros((len(return_iters), *reduc.rolls.shape), dtype=reduc.rolls.dtype)

        iterator = reduc.rolls
        if show_progress:
            iterator = tqdm(iterator, leave=False)
        for i,image in enumerate(iterator):
            init_nans = np.isnan(image)
            image_in, nans = image.copy(), init_nans.copy()
            while np.any(nans):
                image_in = convolution.interpolate_replace_nans(image_in, np.ones((5,5)), convolution.convolve_fft)
                nans = np.isnan(image_in)
            excl_mask = init_nans.copy()
            if excl_mask_in is not None:
                excl_mask |= excl_mask_in

            if init_from_reduc:
                im_deconv_in = image_in.copy()
            else:
                im_deconv_in = None

            out = coronagraphic_richardson_lucy(image_in, deconv_psfs[i], psf_inds=deconv_psf_inds[i], im_mask=deconv_coron_tmaps[i], num_iter=num_iter,
                                                im_deconv_in=im_deconv_in, epsilon=epsilon, return_iters=return_iters, use_gpu=self.use_gpu, ncores=self.ncores, excl_mask=excl_mask,
                                                show_progress=show_progress)
            if return_iters is None:
                rolls_deconv[i] = np.where(excl_mask_in, image, out)
                rolls_deconv[i] = np.where(init_nans, np.nan, rolls_deconv[i])
            else:
                rolls_deconv[i], deconv_iters[:,i] = np.where(excl_mask_in, image, out[0]), np.where(excl_mask_in, image, out[1])
                rolls_deconv[i], deconv_iters[:,i] = np.where(init_nans, np.nan, rolls_deconv[i]), np.where(init_nans, np.nan, deconv_iters[:,i])

        im_deconv = np.nanmedian(rolls_deconv, axis=0)
        products = SpaceReduction(spacerdi=self, im=im_deconv, rolls=rolls_deconv, c_star_out=reduc.c_star, output_ext=f'{reduc.output_ext}_{output_suffix}', reduc_label=reduc_label)
        if save_products:
            try:
                products.save(overwrite=self.overwrite)
            except OSError:
                raise OSError("""
                      A FITS file for this output_ext + output_dir + concat
                      already exists! To overwrite existing files, set the
                      overwrite attribute for your Winnie SpaceRDI instance to
                      True. Alternatively, either change the output_ext
                      attribute for your SpaceRDI instance, or select a
                      different output directory when initializing your
                      SpaceKLIP database object.
                      """)
                      
        if return_iters is None and reduc_in is not None:
            return products

        out = [products]
        if reduc_in is None:
            out.append(reduc)
        if return_iters is not None:
            out.append(deconv_iters)
        return out


    def _check_smoothed_nans(self):
        """
        If we're using smoothing during optimization and zero_nans isn't
        already set to True, see if there's any NaNs in our optzones after smoothing. 
        If so, add zero_nans=True to our settings to avoid all-NaN results.
        """
        if 'opt_smoothing_fn' in self.rdi_settings and not self.rdi_settings.get('zero_nans', False):
            sci_filt = self.rdi_settings['opt_smoothing_fn'](self.imcube_sci, **self.rdi_settings['opt_smoothing_kwargs'])
            ref_filt = self.rdi_settings['opt_smoothing_fn'](self.imcube_ref, **self.rdi_settings['opt_smoothing_kwargs'])
            allopt = np.any(self.optzones, axis=0)
            nans = np.any([*np.isnan(sci_filt[..., allopt]), *np.isnan(ref_filt[..., allopt])])
            if nans: 
                self.rdi_settings['zero_nans'] = True


    def _from_fits(self, input):
        if isinstance(input, fits.ImageHDU):
            p_in = input.data.astype('int64').tolist()
        else:
            with fits.open(input) as hdul:
                p_in = hdul['winnie_pickled'].data.astype('int64').tolist()
        self.__dict__.update(pickle.loads(bytes(p_in)).__dict__)


    def _to_fits(self):
        """
        Convert the SpaceRDI object to a FITS HDU for saving to a FITS file.
        """
        return fits.ImageHDU(list(pickle.dumps(self, pickle.HIGHEST_PROTOCOL)), name='winnie_pickled')


    def __getstate__(self):
        state = self.__dict__.copy()
        for key in list(state.keys()):
            if f'_{key}' in state: # Delete cropped aliases of data
                del state[key]
        if self.efficient_saving:
            for key in ['_imcube_sci', '_imcube_ref', '_errcube_sci', '_errcube_ref']:
                if key in state:
                    del state[key]
            if 'hcube_css' in state['rdi_settings']:
                state['rdi_settings']['hcube_css'] = None
        return state


    def __setstate__(self, state):
        self.__dict__.update(state) # Restore attributes
        # Reload the data if efficient saving was used.
        if self.efficient_saving: 
            imcube_sci, errcube_sci = [],[]
            for f in self._files_sci:
                ints, errs = fits.getdata(f, ext=1), (fits.getdata(f, ext=2) if self.prop_err else None)
                if np.ndim(ints.squeeze()) != 2:
                    if self.use_robust_mean:
                        im, err = robust_mean_combine(ints, errs, self.robust_clip_nsig)
                    else: 
                        im, err = median_combine(ints, errs)
                else:
                    im, err = ints.squeeze(), (None if not self.prop_err else errs.squeeze())
                imcube_sci.append(im)
                errcube_sci.append(err)
            imcube_ref, errcube_ref = [],[]
            for f in self._files_ref:
                ints, errs = fits.getdata(f, ext=1), (fits.getdata(f, ext=2) if self.prop_err else None)
                if np.ndim(ints.squeeze()) != 2:
                    if self.use_robust_mean:
                        im, err = robust_mean_combine(ints, errs, self.robust_clip_nsig)
                    else: 
                        im, err = median_combine(ints, errs)
                else:
                    im, err = ints.squeeze(), (None if not self.prop_err else errs.squeeze())
                imcube_ref.append(im)
                errcube_ref.append(err)
            self._imcube_sci = np.array(imcube_sci)
            self._imcube_ref = np.array(imcube_ref)
            self._errcube_sci = np.array(errcube_sci)
            self._errcube_ref = np.array(errcube_ref)
            if self.pad_data is not None:
                self._imcube_sci = np.pad(self._imcube_sci, self._imc_padding, constant_values=np.nan)
                self._imcube_ref = np.pad(self._imcube_ref, self._imc_padding, constant_values=np.nan)
                self._errcube_sci = np.pad(self._errcube_sci, self._imc_padding, constant_values=np.nan) if self.prop_err else None
                self._errcube_ref = np.pad(self._errcube_ref, self._imc_padding, constant_values=np.nan) if self.prop_err else None
        self.set_crop(self.cropped_shape)
        if self.efficient_saving and 'hcube_css' in self.rdi_settings:
            self.rdi_settings['hcube_css'] = self.imcube_css[:, np.newaxis]


class SpaceConvolution:
    def __init__(self, database, source_spectrum=None,
                 ncores=-1, use_gpu=False, verbose=True,
                 show_plots=False, show_progress=True,
                 overwrite=True, fetch_opd_by_date=True,
                 pad_data='auto', output_basedir=None,
                 output_subdir='psfgrids',
                 efficient_saving=True):
        
        self.database = database
        self.source_spectrum = source_spectrum
        self.ncores = ncores
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.show_plots = show_plots
        self.overwrite = overwrite
        self.show_progress = show_progress
        self.concat = None
        self.inst_webbpsf = None
        self.inst_webbpsfext = None
        self._grid_fetched = False
        self.fetch_opd_by_date = fetch_opd_by_date
        self.pad_data = pad_data
        self.efficient_saving = efficient_saving
        self.output_basedir = self.database.output_dir if output_basedir is None else output_basedir
        self.output_dir = f'{output_basedir}{output_subdir}/'
        if not os.path.isdir(self.output_basedir):
            os.makedirs(self.output_basedir)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if self.use_gpu:
            print("Warning! GPU implementation still in progress; setting use_gpu to False.")
            self.use_gpu = False


    def load_concat(self, concat, reference_index=None, coron_offsets=None, fov_pixels=151,
                    osamp=2, output_ext='psfs', prefetch_psf_grid=True, recalc_psf_grid=False,
                    cropped_shape=None, grid_fn=generate_lyot_psf_grid, grid_kwargs={}, 
                    grid_inds_fn=get_jwst_psf_grid_inds, grid_inds_kwargs={},
                    transmission_map_fn=get_jwst_coron_transmission_map, transmission_map_kwargs={}):

        if isinstance(concat, numbers.Number):
            concat_str = list(self.database.obs.keys())[concat]
        else:
            concat_str = concat

        db_tab = self.database.obs[concat_str]

        self.osamp=int(osamp)
        self.fov_pixels=int(fov_pixels)
        
        sci = db_tab['TYPE'] == 'SCI'
        ref =  db_tab['TYPE'] == 'REF'

        if reference_index is None: # Take the first sci entry to set the date for fetching an OPD file
            reference_index = np.where(sci)[0][0]
            
        files = db_tab['FITSFILE']

        self._c_star = np.array([db_tab['CRPIX1'][0], db_tab['CRPIX2'][0]])-1 # Position of the star in the data
        self.pxscale = db_tab[0]['PIXSCALE']*u.arcsec/u.pixel # pixel scale with astropy units
        self.blurfwhm = np.nan_to_num(db_tab['BLURFWHM'][0])
        self.blursigma = self.blurfwhm/np.sqrt(8.*np.log(2.))
        
        self.concat = concat_str
        
        posangs = []
        visit_ids = []
        c_coron = []

        for i,f in enumerate(files):
            with fits.open(f, lazy_load_hdus=True) as hdul:
                h0, h1 = hdul[0].header, hdul[1].header
                if coron_offsets is None:
                    if 'MASKOFFS' in hdul:
                        offset = np.mean(hdul['MASKOFFS'].data, axis=0)
                    else:
                        offset = np.array([0,0])
                else:
                    offset = coron_offsets[i]
                if i == reference_index:
                    header0, header1 = h0.copy(), h1.copy()

            posangs.append(h1['PA_APER'])
            visit_ids.append(h0['VISIT_ID'])
            c_coron.append(self._c_star-offset)
        
        self.reference_file = files[reference_index]

        self.image_mask = db_tab[0]['CORONMSK'].replace('MASKA', 'MASK').replace('4QPM_', 'FQPM').replace('LYOT_2300', 'LYOT2300')
        self.aperturename = db_tab[0]['APERNAME']
        self.pps_aper = db_tab[0]['PPS_APER']
        self.filt = db_tab[0]['FILTER']
        self.channel = header0.get('CHANNEL', None)
        self.instrument = db_tab[0]['INSTRUME']
        self.date = header0['DATE-BEG']
        if 'PUPIL' in header0:
            self.pupil_mask = header0['PUPIL']
        else:
            if self.filt == 'F2300C':
                self.pupil_mask = 'MASKLYOT'
            elif self.filt.endswith('C'):
                self.pupil_mask='MASKFQPM'
            else:
                self.pupil_mask = None

        posangs, visit_ids, c_coron, files = (np.asarray(i) for i in [posangs, visit_ids, c_coron, files])

        self.posangs_sci = posangs[sci]
        self.visit_ids_sci = visit_ids[sci]
        self._c_coron_sci = c_coron[sci]
        self.files_sci = files[sci]

        self.posangs_ref = posangs[ref]
        self.visit_ids_ref = visit_ids[ref]
        self._c_coron_ref = c_coron[ref]
        self.files_ref = files[ref]
        
        self._ny = header1['NAXIS2']
        self._nx = header1['NAXIS1']

        if self.pad_data is not None: self._apply_padding()

        # Will eventually store these data in a more mutable format
        if self.image_mask == 'MASK210R':
            webbpsf_options = dict(
                pupil_shift_x = -0.0045,
                pupil_shift_y = -0.0022,
                pupil_rotation = -0.38)
        elif self.image_mask == 'MASK335R':
            webbpsf_options = dict(
                pupil_shift_x = -0.0125,
                pupil_shift_y = -0.008,
                pupil_rotation = -0.595)
        elif self.image_mask == 'FQPM1140':
            webbpsf_options = dict(
                pupil_shift_x = 0.00957944,
                pupil_shift_y = 0.01387777,
                pupil_rotation = -0.10441008,
                defocus_waves = 0.01478258)
        else:
            webbpsf_options = {}

        if self.pupil_mask == 'MASKBAR':
            self.pupil_mask = self.pps_aper.split('_')[1]

        self.prepare_webbpsf_instance(options=webbpsf_options)

        self.prepare_webbpsf_ext_instance(options=webbpsf_options)

        self.psf_file = f'{self.output_dir}{concat}_fov{self.fov_pixels}_os{self.osamp}_{output_ext}.fits'

        self._psf_shift = None
        self._coron_tmaps_osamp = None
        self._psf_inds_osamp = None

        self.grid_fn = grid_fn
        self.grid_kwargs = grid_kwargs
        self.transmission_map_fn = transmission_map_fn
        self.transmission_map_kwargs = transmission_map_kwargs
        self.grid_inds_fn = grid_inds_fn
        self.grid_inds_kwargs = grid_inds_kwargs

        self.cropped_shape = cropped_shape
        if prefetch_psf_grid:
            self.fetch_psf_grid(recalc_psf_grid=recalc_psf_grid)
        else:
            self._grid_fetched = False
        

    def _apply_padding(self):
        if self.pad_data == 'auto':
            dymin_pad, dymax_pad, dxmin_pad, dxmax_pad = compute_derot_padding(self._nx, self._ny, -self.posangs_sci, cent=self._c_star)
        else:
            dymin_pad = dymax_pad = dxmin_pad = dxmax_pad = self.pad_data

        cent_adj = np.array([dxmin_pad, dymin_pad])

        self._c_star += cent_adj
        self._c_coron_sci += cent_adj
        self._c_coron_ref += cent_adj

        self._ny = self._ny + dymin_pad + dymax_pad
        self._nx = self._nx + dxmin_pad + dxmax_pad


    def prepare_webbpsf_instance(self, options):
        # If we're setting OPD based on date and the WebbPSF instance is either
        # not initialized or the date has changed:
        if self.fetch_opd_by_date and (self.inst_webbpsf is None or self.opd_query_date.split('T')[0] != self.date.split('T')[0]):
            if self.pupil_mask.endswith('WB'):
                reference_file = fits.open(self.reference_file)
                reference_file[0].header['PUPIL'] = self.pupil_mask
            else:
                reference_file = self.reference_file
            self.inst_webbpsf = webbpsf.setup_sim_to_match_file(reference_file)
            self.inst_webbpsf.options.update(options)
            self.opd_query_date = self.date
        # Otherwise, update the non-OPD elements; This is more or less borrowed
        # from webbpsf.setup_sim_to_match_file()
        else: 
            if self.inst_webbpsf is None: # if needed, initialize the WebbPSF instance
                self.inst_webbpsf = webbpsf.instrument(self.instrument)
            self.inst_webbpsf.filter = self.filt
            self.inst_webbpsf.set_position_from_aperture_name(self.aperturename)
            if self.inst_webbpsf.name == 'NIRCam':
                if self.pupil_mask.startswith('MASK'):
                    self.inst_webbpsf.pupil_mask = self.pupil_mask
                    self.inst_webbpsf.image_mask = self.image_mask                                                   
                    self.inst_webbpsf.set_position_from_aperture_name(self.aperturename)
            elif self.inst_webbpsf.name == 'MIRI':
                if self.inst_webbpsf.filter in ['F1065C', 'F1140C', 'F1550C']:
                    self.inst_webbpsf.image_mask = 'FQPM'+self.filt[1:5]
                elif self.inst_webbpsf.filter == 'F2300C':
                    self.inst_webbpsf.image_mask = 'LYOT2300'
            self.inst_webbpsf.options.update(options)
             # In case we're initializing with fetch_opd_by_date=False, we need
             # to set a null OPD date so that if self.fetch_opd_by_date is
             # later set to True, the OPD will be loaded when executing this
             # method.
            if not self.fetch_opd_by_date:
                self.opd_query_date = '' # null date

    
    def prepare_webbpsf_ext_instance(self, options):
        # Swap to prelaunch equivalent mask names for WebbPSF-ext
        if self.pupil_mask.endswith('RND'):
            pupil_mask_ext = 'CIRCLYOT'
        elif self.pupil_mask.endswith('WB'):
            pupil_mask_ext = 'WEDGELYOT'
        else:
            pupil_mask_ext = self.pupil_mask

        # Currently we only use webbpsf_ext for transmission maps, which aren't OPD dependent.
        if self.instrument == 'NIRCAM':
            inst = webbpsf_ext.NIRCam_ext(filter=self.filt, pupil_mask=pupil_mask_ext, image_mask=self.image_mask,
                                         fov_pix=self.fov_pixels, oversample=self.osamp)
        else:
            inst = webbpsf_ext.MIRI_ext(filter=self.filt, pupil_mask=pupil_mask_ext, image_mask=self.image_mask,
                                        fov_pix=self.fov_pixels, oversample=self.osamp)
        inst.aperturename = self.aperturename
        inst.oversample = self.osamp
        inst.options.update(options)
        self.inst_webbpsfext = inst
    

    def calc_psf_shift(self):
        inst_off = deepcopy(self.inst_webbpsf)
        inst_off.image_mask = None
        psf_off = inst_off.calc_psf(source=None,
                                    oversample=4,
                                    fov_pixels=35)[2].data
        self._psf_shift = get_webbpsf_model_center_offset(psf_off, osamp=4)
    
    
    def fetch_psf_grid(self, recalc_psf_grid=False):
        """
        ___________
        Parameters:

        recalc_psf_grid : bool, optional
            Whether to recalculate the PSF grid if it has already been
            generated and saved. Default is False.
        grid_inds_fn : callable, optional
            The function to use for matching model pixels to PSF grid samples.
            Default is winnie.convolution.get_jwst_psf_grid_inds.
        grid_inds_kwargs : dict, optional
            Additional keyword arguments to be passed to grid_inds_fn.
        transmission_map_fn : callable, optional
            The function to use for generating the coronagraph transmission
            map. Default is winnie.convolution.get_jwst_coron_transmission_map.
        transmission_map_kwargs : dict, optional
            Additional keyword arguments to be passed to transmission_map_fn.
        grid_fn : callable, optional
            The function to use for generating the PSF grid. Default is
            winnie.convolution.generate_lyot_psf_grid.
        grid_kwargs : dict, optional
            Additional keyword arguments to be passed to grid_fn.

        Notes:

        - grid_fn should return three objects: 1) a 3D array of PSF samples, 2)
          the 2D array of shape (2,N) containing the polar coordinates of those
          samples (relative to the coronagraph center; units of [arcsec,
          degrees]), and 3) the 2D array of shape (2,N) containing the
          cartesian coordinates of those samples (relative to the coronagraph
          center; units of [arcsec, arcsec]). It should take a WebbPSF
          instrument object as the first argument, and must have the following
          signature: 
                grid_fn(inst_webbpsf, source_spectrum=None, shift=None,
                osamp=2, fov_pixels=151, show_progress=True, **grid_kwargs)
          See winnie.convolution.generate_lyot_psf_grid for documentation
          regarding these arguments.

        - grid_inds_fn should return a 2D integer array of shape (ny*osamp,
          nx*osamp) where each pixel provides the index of the PSF sample that
          should be used for convolution of that pixel. It should have the
          following signature:
                grid_inds_fn(c_coron, psf_offsets_polar, osamp=2, shape=None,
                pxscale=None, **grid_inds_kwargs)
          See winnie.convolution.get_jwst_psf_grid_inds for documentation
          regarding these arguments.

        - transmission_map_fn should return a 2D array of shape (ny*osamp,
          nx*osamp) where each pixel provides the transmission value of the
          coronagraph at that location. It should have the following signature:
                transmission_map_fn(inst_webbpsfext, c_coron, osamp=2,
                shape=None, **transmission_map_kwargs)
          See winnie.convolution.get_jwst_coron_transmission_map for
          documentation regarding these arguments.
        """
        if os.path.isfile(self.psf_file) and not recalc_psf_grid:
            with fits.open(self.psf_file) as hdul:
                self.psfs = hdul[0].data
                self.psf_offsets_polar = hdul[1].data
                self.psf_offsets = hdul[2].data
        else:
            if self.grid_fn is not None:
                if self._psf_shift is None:
                    self.calc_psf_shift()
                out = self.grid_fn(self.inst_webbpsf, source_spectrum=self.source_spectrum,
                            shift=self._psf_shift, osamp=self.osamp, fov_pixels=self.fov_pixels,
                            show_progress=self.show_progress, **self.grid_kwargs)
                
                self.psfs, self.psf_offsets_polar, self.psf_offsets = out
                hdul = fits.HDUList(hdus=[fits.PrimaryHDU(self.psfs), 
                                          fits.ImageHDU(self.psf_offsets_polar),
                                          fits.ImageHDU(self.psf_offsets)])
                hdul.writeto(self.psf_file, overwrite=True)
            else:
                self.psfs, self.psf_offsets_polar, self.psf_offsets = None, None, None
            
        if self.blursigma != 0 and self.psfs is not None:
            self.psfs = gaussian_filter_sequence(self.psfs, self.blursigma*self.osamp)
            
        self._fetch_grid_inds()
        self._fetch_transmission_maps()

        self._grid_fetched = True
        self.set_crop(self.cropped_shape)


    def _fetch_grid_inds(self):
        if self.grid_inds_fn is not None:
            psf_inds_osamp = []
            for c_coron in self._c_coron_sci:
                psf_inds = self.grid_inds_fn(c_coron, self.psf_offsets_polar, osamp=self.osamp, shape=(self._ny, self._nx), pxscale=self.pxscale, **self.grid_inds_kwargs)
                psf_inds_osamp.append(psf_inds)
            self._psf_inds_osamp = np.asarray(psf_inds_osamp)
        else:
            self._psf_inds_osamp = None


    def _fetch_transmission_maps(self):
        if self.transmission_map_fn is not None:
            coron_tmaps_osamp = []
            for c_coron in self._c_coron_sci:
                coron_tmap = self.transmission_map_fn(self.inst_webbpsfext, c_coron, osamp=self.osamp, shape=(self._ny, self._nx), **self.transmission_map_kwargs)
                coron_tmaps_osamp.append(coron_tmap)
            self._coron_tmaps_osamp = np.asarray(coron_tmaps_osamp)
        else:
            self._coron_tmaps_osamp = np.ones((len(self._c_coron_sci), self._ny*self.osamp, self._nx*self.osamp))


    def set_crop(self, cropped_shape=None):
        self.cropped_shape = (np.asarray(cropped_shape) if cropped_shape is not None else cropped_shape)
        if self._grid_fetched:
            if cropped_shape is not None:
                cr_ny, cr_nx = self.cropped_shape
                x0, y0 = self._c_star
                x1, y1 = max(0, int(np.round(x0-(cr_nx-1.)/2.))), max(0, int(np.round(y0-(cr_ny-1.)/2.)))
                cr_cent = np.array([x0-x1, y0-y1])
                self.c_star = cr_cent
                if self._coron_tmaps_osamp is not None:
                    self.coron_tmaps_osamp, _, self.crop_indices_osamp = crop_data(self._coron_tmaps_osamp,
                                                                    c_to_c_osamp(self._c_star, self.osamp),
                                                                    self.cropped_shape*self.osamp,
                                                                    return_indices=True, copy=False)
                else:
                    self.coron_tmaps_osamp = None
                    _, _, self.crop_indices_osamp = crop_data(np.zeros((self._ny*self.osamp, self._nx*self.osamp)),
                                                              c_to_c_osamp(self._c_star, self.osamp),
                                                              self.cropped_shape*self.osamp,
                                                              return_indices=True, copy=False)
                y1o, y2o, x1o, x2o = self.crop_indices_osamp
                if self._psf_inds_osamp is not None:
                    self.psf_inds_osamp = self._psf_inds_osamp[..., y1o:y2o, x1o:x2o]
                else:
                    self.psf_inds_osamp = None
                self.ny, self.nx = self.cropped_shape
                self.c_coron_sci = self._c_coron_sci - np.array([x1,y1])
                self.c_coron_ref = self._c_coron_ref - np.array([x1,y1])
            else:
                self.crop_indices_osamp = None
                self.c_star = self._c_star
                self.c_coron_sci = self._c_coron_sci
                self.c_coron_ref = self._c_coron_ref
                self.coron_tmaps_osamp = self._coron_tmaps_osamp
                self.psf_inds_osamp = self._psf_inds_osamp
                self.ny, self.nx = self._ny, self._nx


    def set_custom_grid(self, psfs, psf_inds_osamp, coron_tmaps_osamp):
        self.psfs = psfs
        self._psf_inds_osamp = psf_inds_osamp
        self._coron_tmaps_osamp = coron_tmaps_osamp
        self._grid_fetched = True
        self.set_crop(self.cropped_shape)


    def convolve_model(self, model, pxscale_in, c_star_in=None):
        if c_star_in is None:
            ny_in, nx_in = model.shape
            c_star_in = (np.array([nx_in, ny_in])-1.)/2.

        pxscale_osamp = self.pxscale.value/self.osamp
        shape_osamp = (self.ny*self.osamp, self.nx*self.osamp)
        c_star_osamp = c_to_c_osamp(self.c_star, self.osamp)

        sfac = (pxscale_in << u.arcsec/u.pixel).value / pxscale_osamp

        # crop the input model to spec in order to avoid resampling/rotating where unnecessary
        # could be improved slightly by considering the rotations involved. This aims to preserve
        # the entire rotated+cropped FOV for the model in the worst case scenario.
        model_cr_ny, model_cr_nx = np.ceil(np.array([self.ny, self.nx])/sfac*self.osamp).astype(int)
        model_cr_nh = np.ceil(np.hypot(model_cr_nx, model_cr_ny)).astype(int)

        model_cr, c_star_in_cr = crop_data(model, c_star_in, [model_cr_nh, model_cr_nh], copy=True)

        if sfac != 1:
            model_osamp = webbpsf_ext.image_manip.frebin(model_cr, scale=sfac, total=False)
        else:
            model_osamp = model.copy()
            
        c_star_in_cr_osamp = c_to_c_osamp(c_star_in_cr, sfac)
        model_con_out = np.zeros((len(self.posangs_sci), self.ny, self.nx))
        for i,posang in enumerate(self.posangs_sci):
            model_rot = rotate_image(model_osamp, posang, cent=c_star_in_cr_osamp, use_gpu=self.use_gpu, cval0=0)
            model_rot = pad_or_crop_image(model_rot, shape_osamp, cent=c_star_in_cr_osamp, new_cent=c_star_osamp, cval0=0.)

            model_con = convolve_with_spatial_psfs(model_rot, self.psfs, self.psf_inds_osamp[i],
                                                coron_tmap=self.coron_tmaps_osamp[i],
                                                use_gpu=self.use_gpu, ncores=self.ncores)

            model_con_out[i] = webbpsf_ext.image_manip.frebin(model_con, scale=1/self.osamp, total=False)

        return model_con_out


    def __getstate__(self):
        state = self.__dict__.copy()
        for key in list(state.keys()):
            if f'_{key}' in state: # Delete cropped aliases of data
                del state[key]
        if self.efficient_saving:
            for key in ['psfs', 'psf_offsets_polar', 'psf_offsets', '_coron_tmaps_osamp', '_psf_inds_osamp']:
                if key in state:
                    del state[key]
        if state['inst_webbpsf'] is not None and isinstance(state['inst_webbpsf'].pupilopd, fits.hdu.hdulist.HDUList):
            state['inst_webbpsf'].pupilopd = None
        return state


    def __setstate__(self, state):
        self.__dict__.update(state) # Restore attributes
        if (self.inst_webbpsf.pupilopd is None) and self.fetch_opd_by_date:
            self.inst_webbpsf.load_wss_opd_by_date(self.opd_query_date)
        if self._grid_fetched:
            if self.efficient_saving: # If the grid had been fetched before, load it again
                self.fetch_psf_grid(recalc_psf_grid=False)
            else:
                self.set_crop(self.cropped_shape)


class SpaceReduction:
    def __init__(self, file_to_load=None, spacerdi=None, output_ext=None, im=None, rolls=None, err=None,
                 err_rolls=None, c_star_out=None, concat=None, derotated=True, reduc_label=''):
        """
        Generates a new set of reduction products or load one that was previously saved.
        
        Options for loading saved output:
        a) provide file_to_load, a complete or relative path to a FITS file saved by Winnie
        b) provide spacerdi, output_ext, and concat — where spacerdi can be a SpaceRDI object WITHOUT
           a concatenation loaded to save memory.
        c) provide spacerdi and output_ext, where spacerdi is a SpaceRDI instance WITH a concatenation loaded
           and where output_ext is the extension that was used when saving the file previously.
        """
        if spacerdi is not None and np.logical_or(im is not None, rolls is not None):
            self.im = im
            self.rolls = rolls
            self.err = err
            self.err_rolls = err_rolls
            self.c_star = c_star_out
            self.pxscale = spacerdi.pxscale
            self.reduc_label = reduc_label

            if self.im is not None:
                self.ny, self.nx = self.im.shape
            else:
                self.ny, self.nx = self.rolls.shape[1:]

            image_headers = []
            c_coron_out = []
            uni_visit_inds = np.sort(np.unique(spacerdi._visit_ids_sci, return_index=True)[1])
            for i in uni_visit_inds: # For multi-exposure rolls, retain header info for 1st exposure of each roll
                with fits.open(spacerdi._files_sci[i]) as hdul:
                    h0 = hdul[0].header
                    h1 = hdul[1].header
                    
                h1['PXSCALE'] = ((self.pxscale << u.arcsec/u.pixel).value, 'average pixel scale in arcsec/pixel')
                h1.update(NAXIS1=self.nx, NAXIS2=self.ny, CRPIX1=c_star_out[0]+1, CRPIX2=c_star_out[1]+1)

                c_coron_i = spacerdi.c_coron_sci[i]
                if derotated:
                    w = wcs.WCS(h1, naxis=[1,2])
                    _rotate_wcs_hdr(w, h1['ROLL_REF'])
                    h1['CD1_1'] = w.wcs.cd[0, 0]
                    h1['CD1_2'] = w.wcs.cd[0, 1]
                    h1['CD2_1'] = w.wcs.cd[1, 0]
                    h1['CD2_2'] = w.wcs.cd[1, 1]

                    c_coron_i = xy_polar_ang_displacement(*(c_coron_i-spacerdi.c_star),
                                                            spacerdi._posangs_sci[i])+c_star_out
                c_coron_out.append(c_coron_i)
                
                h1['CCORON1'], h1['CCORON2'] = zip(c_coron_i+1,
                                                   ['axis 1 coordinate of the coron center',
                                                    'axis 2 coordinate of the coron center'])
                
                image_headers.append(h1)
                if i == uni_visit_inds[0]:
                    self.primary_header = h0
            
            self.primary_header['MODE'] = reduc_label
            self.primary_header['output_ext'] = output_ext

            self.primary_header['ANNULI'] = spacerdi.optzones.shape[0]
            self.primary_header['CRPIX1'] = c_star_out[0]+1
            self.primary_header['CRPIX2'] = c_star_out[1]+1

            self.filename = f'{spacerdi.output_dir}{spacerdi.concat}_{output_ext}_i2d.fits'
            self.image_headers = image_headers
            self.c_coron = np.array(c_coron_out)
            self.output_ext = output_ext

            h1_combined = self.image_headers[0].copy()
            h1_combined['CCORON1'], h1_combined['CCORON2'] = zip(np.mean(c_coron_out, axis=0)+1,
                                                                 ['axis 1 coordinate of the coron center',
                                                                  'axis 2 coordinate of the coron center'])
            self.combined_image_header = h1_combined
            

        else: # load the products from a file
            if file_to_load is None:
                if spacerdi is not None and output_ext is not None and concat is not None:
                    if isinstance(concat, numbers.Number):
                        concat_str = list(spacerdi.database.obs.keys())[concat]
                    else:
                        concat_str = concat

                    db_tab = spacerdi.database.obs[concat_str]
                    file = db_tab[db_tab['TYPE'] == 'SCI']['FITSFILE'][0]
                    h0 = fits.getheader(file, ext=0)
                    
                elif spacerdi is not None and output_ext is not None and spacerdi.concat is not None:
                    h0 = fits.getheader(spacerdi._files_sci[0], ext=0)
                    concat_str = spacerdi.concat

                else:
                    raise ValueError("""
                    To load a saved SpaceReduction object, you must provide a) a file_to_load path, 
                    b) a spacerdi object, output_ext, and concat, or c) a spacerdi object with a
                                     concatenation already loaded and output_ext.      
                    """)
                file_to_load = f'{spacerdi.output_dir}{concat_str}_{output_ext}_i2d.fits'
                if not os.path.exists(file_to_load): # For backwards compatability
                    file_to_load = '{}JWST_{}_{}_{}_{}_{}_{}.fits'.format(spacerdi.output_dir,
                                                                            h0['INSTRUME'],
                                                                            h0['DETECTOR'],
                                                                            h0['FILTER'],
                                                                            h0['CORONMSK'],
                                                                            h0['SUBARRAY'],
                                                                            output_ext)
            self.filename = file_to_load
            hdul = fits.open(file_to_load)
            self.primary_header = hdul[0].header
            self.im = (hdul['SCI'].data if 'SCI' in hdul else None)
            self.err = (hdul['ERR'].data if 'ERR' in hdul else None)
            self.reduc_label = self.primary_header.get('MODE', 'UNKNOWN (Winnie)') # backwards compatability
            if output_ext is None:
                self.output_ext = self.primary_header.get('output_ext', self.filename.split(concat_str)[-1][1:].replace('.fits', '')) # backwards compatability
            else:
                self.output_ext = output_ext
            self.rolls = []
            self.err_rolls = []
            self.image_headers = []
            
            for hdu in hdul:
                if hdu.name.startswith('SCI_ROLL'):
                    self.rolls.append(hdu.data)
                    self.image_headers.append(hdu.header)
                elif hdu.name.startswith('ERR_ROLL'):
                    self.err_rolls.append(hdu.data)

            if len(self.rolls) == 0:
                self.rolls = None
            else:
                self.rolls = np.asarray(self.rolls)
                
            if len(self.err_rolls) == 0:
                self.err_rolls = None
            else:
                self.err_rolls = np.asarray(self.err_rolls)
                
            self.combined_image_header = hdul[1].header
            self.c_star = np.array([self.combined_image_header['CRPIX1'], self.combined_image_header['CRPIX2']])-1
            self.c_coron = np.array([[h1['CCORON1'], h1['CCORON2']] for h1 in self.image_headers])-1
            self.pxscale = self.image_headers[0]['PXSCALE']*u.arcsec/u.pixel
            
            if self.im is not None:
                self.ny, self.nx = self.im.shape
            else:
                self.ny, self.nx = self.rolls.shape[1:]

            hdul.close()
        self.extent = mpl_centered_extent([self.ny, self.nx], self.c_star, self.pxscale)
    

    def generate_rmaps(self):
        self.rmap = dist_to_pt(self.c_star, self.nx, self.ny)
        self.rmap_arcsec = px_size_to_ang_size(self.rmap, self.pxscale).value


    def to_hdulist(self):
        hdul = fits.HDUList(hdus=[fits.PrimaryHDU(header=self.primary_header)])
        if self.im is not None:
            h1 = self.combined_image_header
            hdul.append(fits.ImageHDU(data=self.im, header=h1, name='SCI'))
            if self.err is not None: 
                hdul.append(fits.ImageHDU(data=self.err, header=h1, name='ERR'))
            
        if self.rolls is not None:
            for i,image in enumerate(self.rolls):
                h1 = self.image_headers[i]
                hdul.append(fits.ImageHDU(data=image, header=h1, name=f'SCI_ROLL{i+1}'))
                if self.err_rolls is not None: 
                    hdul.append(fits.ImageHDU(data=self.err_rolls[i], header=h1, name=f'ERR_ROLL{i+1}'))
        else:
            # Save extensions with just the header info
            for i,h1 in enumerate(copy(self.image_headers)):
                hdul.append(fits.ImageHDU(data=None, header=h1, name=f'SCI_ROLL{i+1}'))
        
        return hdul


    def save(self, overwrite=False, extra_hdus=[]):
        hdul = self.to_hdulist()
        hdul += fits.HDUList(extra_hdus)
        try:
            hdul.writeto(self.filename, overwrite=overwrite)
        except OSError:
            raise OSError("""
                  The file you are attempting to write already exists! Use overwrite=True to overwrite
                  the existing file, or assign a different file name by changing the filename attribute.
                  """)