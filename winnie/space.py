import numpy as np
from astropy.io import fits
from astropy import wcs
from pyklip.klip import _rotate_wcs_hdr
import numbers
from copy import (copy, deepcopy)
import astropy.units as u
import os
import webbpsf
import webbpsf_ext

from .rdi import (rdi_residuals, build_annular_rdi_zones)

from .plot import (mpl, plt, quick_implot, mpl_centered_extent)

from .utils import (robust_mean_combine, median_combine,
                    ang_size_to_px_size, px_size_to_ang_size,
                    high_pass_filter_sequence, pad_and_rotate_hypercube, rotate_hypercube,
                    xy_polar_ang_displacement, rotate_image, gaussian_filter_sequence, crop_data,
                    c_to_c_osamp, pad_or_crop_image, dist_to_pt, compute_derot_padding)

from .convolution import (convolve_with_spatial_psfs,
                          get_jwst_psf_grid_inds,
                          get_jwst_coron_transmission_map,
                          generate_lyot_psf_grid,
                          get_webbpsf_model_center_offset)

class SpaceRDI:
    def __init__(self, database, data_ext=None, ncores=-1, use_gpu=False, 
                 verbose=True, show_plots=False, overwrite=False,
                 prop_err=True, show_progress=False, use_robust_mean=False,
                 robust_clip_nsig=3, pad_data='auto', pad_before_derot=False,
                 r_opt=3*u.arcsec, r_sub=None):
        """
        Initialize the Winnie class for carrying out RDI on JWST data.

        Parameters
        ----------
         database: spaceklip.Database
            SpaceKLIP database containing stage 2 observations to work with.

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
        """
        self.concat = None
        self.convolver = None
        self.database = database
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
        
        imcube, errcube, posangs, visit_ids, c_coron, dates, files = (np.asarray(i) for i in [imcube, errcube, posangs, visit_ids, c_coron, dates, files])

        self._imcube_sci = imcube[sci]
        self._errcube_sci = errcube[sci] if self.prop_err else None
        self._posangs_sci = posangs[sci]
        self._visit_ids_sci = visit_ids[sci]
        self._c_coron_sci = c_coron[sci]
        self._dates_sci = dates[sci]
        self._files_sci = files[sci]

        self._imcube_ref = imcube[ref]
        self._errcube_ref = errcube[ref] if self.prop_err else None
        self._posangs_ref = posangs[ref]
        self._visit_ids_ref = visit_ids[ref]
        self._c_coron_ref = c_coron[ref]
        self._dates_ref = dates[ref]
        self._files_ref = files[ref]
        
        self._ny, self._nx = imcube.shape[-2:]
        
        if self.pad_data is not None: self._apply_padding()

        self._imcube_css = None
        self.imcube_css = None
        self.cropped_shape = None
        
        self.fixed_rdi_settings = {}

        # Setting initial annular zones for RDI procedure based on set defaults.
        self.update_annular_zones(exclude_opt_nans=True)

        self.set_crop(cropped_shape)

        self.rdi_presets()

        if self.convolver is not None:
            self.convolver.load_concat(self.concat, **self.convolver_args)


    def _apply_padding(self):
        if self.pad_data == 'auto':
            dymin_pad, dymax_pad, dxmin_pad, dxmax_pad = compute_derot_padding(self._nx, self._ny, -self._posangs_sci, cent=self._c_star)
        else:
            dymin_pad = dymax_pad = dxmin_pad = dxmax_pad = self.pad_data

        imc_padding = [[0,0], [dymin_pad, dymax_pad], [dxmin_pad, dxmax_pad]]
        cent_adj = np.array([dxmin_pad, dymin_pad])
        
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
            if self._imcube_css is not None:
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
        else:
            imcube_sci = self.imcube_sci
            prop_err = self.prop_err
            
        pad_before_derot = self.pad_before_derot
        
        if not prop_err or return_res_only:
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
                                  derotated=derotate)
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
    
    
    def set_presets(self, presets={}, output_ext='psfsub'):
        """
        Generic method to quickly assign a set of arguments to use for
        winnie.rdi.rdi_residuals, while also setting the extension for saved
        files, repopulating any settings in self.fixed_rdi_settings, and
        reporting the configuration if verbose is True.
        """
        self.output_ext = output_ext
        self.rdi_settings = presets
        self.rdi_settings.update(self.fixed_rdi_settings)
        if self.verbose:
            self.report_current_config()
    
    
    def rdi_presets(self, output_ext='rdi_psfsub'):
        """
        Set presets to perform a standard RDI reduction.

        ___________
        Parameters:
            output_ext (str, optional): Output file extension for FITS
                products. Defaults to 'rdi_psfsub'.
        """
        self.set_presets(presets={}, output_ext=output_ext)
    
    
    def hpfrdi_presets(self, filter_size=None, filter_size_adj=1, output_ext='hpfrdi_psfsub'):
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

        Notes:
            - This method checks if there will be any NaN values in the
              optimization zones after applying the specified filtering.
            - If so, it also sets 'zero_nans' to True to avoid a crash when
              run_rdi is called.
        """
        if filter_size is None:
            filter_size = self._sigma
        presets = {}
        presets['opt_smoothing_fn'] = high_pass_filter_sequence
        presets['opt_smoothing_kwargs'] = dict(filtersize=filter_size_adj*filter_size)
        # See if there's any NaNs in our optzones after filtering is applied. 
        # If so, add zero_nans=True to our settings to avoid a crash.
        sci_filt = high_pass_filter_sequence(self.imcube_sci, filter_size)
        ref_filt = high_pass_filter_sequence(self.imcube_ref, filter_size)
        allopt = np.any(self.optzones, axis=0)
        nans = np.any([*np.isnan(sci_filt[..., allopt]), *np.isnan(ref_filt[..., allopt])])
        if nans:
            presets['zero_nans'] = True
        self.set_presets(presets=presets, output_ext=output_ext)
    
    
    def mcrdi_presets(self, output_ext='mcrdi_psfsub'):
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
                         output_ext=output_ext)
    

    def prepare_convolution(self, source_spectrum=None, reference_index=0, fov_pixels=151, osamp=2,
                            output_ext='psfs', prefetch_psf_grid=True, recalc_psf_grid=False,
                            psfgrids_output_dir='psfgrids', fetch_opd_by_date=True, 
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
            self.convolver = SpaceConvolution(database=self.database, source_spectrum=source_spectrum, ncores=self.ncores, use_gpu=self.use_gpu,
                                              verbose=self.verbose, show_plots=self.show_plots, show_progress=True, overwrite=self.overwrite,
                                              psfgrids_output_dir=psfgrids_output_dir, fetch_opd_by_date=fetch_opd_by_date, pad_data=self.pad_data)
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
            if self.convolver is None:
                    raise ValueError("""
                        To run set_circumstellar_model with a raw model as
                        input, you must first execute prepare_convolution.
                                     """)
            model_cube = self.convolver.convolve_model(raw_model, pxscale_in=raw_model_pxscale, c_star_in=raw_model_center)

        if model_cube is None:
            if model_files is None:
                if model_dir is None:
                    model_dir = self.database.output_dir
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
            fout = self.database.output_dir+os.path.basename(os.path.normpath(f)).replace(self.data_ext, output_ext)
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


class SpaceConvolution:
    def __init__(self, database, source_spectrum=None,
                 ncores=-1, use_gpu=False, verbose=True,
                 show_plots=False, show_progress=True,
                 overwrite=True, fetch_opd_by_date=True,
                 pad_data='auto', psfgrids_output_dir='psfgrids'):
        
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
        self.psfgrids_output_dir = f'{database.output_dir}{psfgrids_output_dir}/'
        if not os.path.isdir(self.psfgrids_output_dir):
            os.makedirs(self.psfgrids_output_dir)
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
        
        reference_file = files[reference_index]

        self.image_mask = header0['CORONMSK'].replace('MASKA', 'MASK').replace('4QPM_', 'FQPM')
        self.aperturename = header0['APERNAME']
        self.pps_aper = header0['PPS_APER']
        self.filt = header0['FILTER']
        self.channel = header0.get('CHANNEL', None)
        self.instrument = header0['INSTRUME']
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
        if self.channel == 'SHORT' and self.image_mask == 'MASK335R':
            webbpsf_options = dict(
                pupil_shift_x = -0.0045,
                pupil_shift_y = -0.0022,
                pupil_rotation = -0.38)
        elif self.channel == 'LONG' and self.image_mask == 'MASK335R':
            webbpsf_options = dict(
                pupil_shift_x = -0.0125,
                pupil_shift_y = -0.008,
                pupil_rotation = -0.595)
        elif self.channel is None and self.image_mask == 'FQPM1140':
            webbpsf_options = dict(
                pupil_shift_x = 0.00957944,
                pupil_shift_y = 0.01387777,
                pupil_rotation = -0.10441008,
                defocus_waves = 0.01478258)
        else:
            webbpsf_options = {}

        if self.pupil_mask == 'MASKBAR':
            self.pupil_mask = self.pps_aper.split('_')[1]

        if self.fetch_opd_by_date and (self.inst_webbpsf is None or self.inst_webbpsf.opd_query_date.split('T')[0] != self.date.split('T')[0]):
            if self.pupil_mask.endswith('WB'):
                reference_file = fits.open(reference_file)
                reference_file[0].header['PUPIL'] = self.pupil_mask
            self.initialize_webbpsf_instance(file=reference_file, options=webbpsf_options)
        else: # Otherwise, update the non-OPD elements; This is more or less borrowed from webbpsf.setup_sim_to_match_file()
            if self.inst_webbpsf is None:
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
            self.inst_webbpsf.options.update(webbpsf_options)

        self.inst_webbpsf.opd_query_date = self.date
        
        self.initialize_webbpsf_ext_instance(options=webbpsf_options)

        self.psf_file = '{}JWST_{}_{}_{}_{}_{}_fov{}_os{}_{}.fits'.format(self.psfgrids_output_dir,
                                                                                header0['INSTRUME'],
                                                                                header0['DETECTOR'],
                                                                                header0['filter'],
                                                                                header0['coronmsk'],
                                                                                header0['SUBARRAY'],
                                                                                self.fov_pixels,
                                                                                self.osamp,
                                                                                output_ext)
        
        self._psf_shift = None
        self._coron_tmaps_osamp = None
        self._psf_inds_osamp = None

        self.cropped_shape = cropped_shape
        if prefetch_psf_grid:
            self.fetch_psf_grid(recalc_psf_grid=recalc_psf_grid, grid_inds_fn=grid_inds_fn, 
                                grid_inds_kwargs=grid_inds_kwargs, transmission_map_fn=transmission_map_fn,
                                transmission_map_kwargs=transmission_map_kwargs, grid_fn=grid_fn, **grid_kwargs)
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


    def initialize_webbpsf_instance(self, file, options):
        inst = webbpsf.setup_sim_to_match_file(file)
        inst.options.update(options)
        self.inst_webbpsf = inst

    
    def initialize_webbpsf_ext_instance(self, options):
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
    
    
    def fetch_psf_grid(self, recalc_psf_grid=False, grid_inds_fn=get_jwst_psf_grid_inds,
                        grid_inds_kwargs={}, transmission_map_fn=get_jwst_coron_transmission_map,
                        transmission_map_kwargs={}, grid_fn=generate_lyot_psf_grid,
                        **grid_kwargs):
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
            if self._psf_shift is None:
                self.calc_psf_shift()
            out = grid_fn(self.inst_webbpsf, source_spectrum=self.source_spectrum,
                          shift=self._psf_shift, osamp=self.osamp, fov_pixels=self.fov_pixels,
                          show_progress=self.show_progress, **grid_kwargs)
            
            self.psfs, self.psf_offsets_polar, self.psf_offsets = out
            hdul = fits.HDUList(hdus=[fits.PrimaryHDU(self.psfs), 
                                      fits.ImageHDU(self.psf_offsets_polar),
                                      fits.ImageHDU(self.psf_offsets)])
            hdul.writeto(self.psf_file, overwrite=True)
            
        if self.blursigma != 0:
            self.psfs = gaussian_filter_sequence(self.psfs, self.blursigma*self.osamp)
            
        if grid_inds_fn is not None:
            psf_inds_osamp = []
            for c_coron in self._c_coron_sci:
                psf_inds = grid_inds_fn(c_coron, self.psf_offsets_polar, osamp=self.osamp, shape=(self._ny, self._nx), pxscale=self.pxscale, **grid_inds_kwargs)
                psf_inds_osamp.append(psf_inds)
            self._psf_inds_osamp = np.asarray(psf_inds_osamp)
        else:
            self._psf_inds_osamp = None

        if transmission_map_fn is not None:
            coron_tmaps_osamp = []
            for c_coron in self._c_coron_sci:
                coron_tmap = transmission_map_fn(self.inst_webbpsfext, c_coron, osamp=self.osamp, shape=(self._ny, self._nx), **transmission_map_kwargs)
                coron_tmaps_osamp.append(coron_tmap)
            self._coron_tmaps_osamp = np.asarray(coron_tmaps_osamp)
        else:
            self._coron_tmaps_osamp = np.ones((len(self._c_coron_sci), self._ny*self.osamp, self._nx*self.osamp))

        self._grid_fetched = True
        self.set_crop(self.cropped_shape)


    def set_crop(self, cropped_shape=None):
        self.cropped_shape = (np.asarray(cropped_shape) if cropped_shape is not None else cropped_shape)
        if self._grid_fetched:
            if cropped_shape is not None:
                cr_ny, cr_nx = self.cropped_shape
                x0, y0 = self._c_star
                x1, y1 = max(0, int(np.round(x0-(cr_nx-1.)/2.))), max(0, int(np.round(y0-(cr_ny-1.)/2.)))
                cr_cent = np.array([x0-x1, y0-y1])
                self.c_star = cr_cent
                self.coron_tmaps_osamp, _, self.crop_indices_osamp = crop_data(self._coron_tmaps_osamp,
                                                                c_to_c_osamp(self._c_star, self.osamp),
                                                                self.cropped_shape*self.osamp,
                                                                return_indices=True, copy=False)
                y1o, y2o, x1o, x2o = self.crop_indices_osamp
                self.psf_inds_osamp = self._psf_inds_osamp[..., y1o:y2o, x1o:x2o]
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

    
class SpaceReduction:
    def __init__(self, file_to_load=None, spacerdi=None, output_ext=None, im=None, rolls=None, err=None,
                 err_rolls=None, c_star_out=None, concat=None, derotated=True):
        
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
                    
                h1.update(NAXIS1=self.nx, NAXIS2=self.ny, CRPIX1=c_star_out[0]+1, CRPIX2=c_star_out[1]+1)
                h1['PXSCALE'] = ((self.pxscale << u.arcsec/u.pixel).value, 'average pixel scale in arcsec/pixel')

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
            
            self.filename = '{}JWST_{}_{}_{}_{}_{}_{}.fits'.format(spacerdi.database.output_dir,
                                                                      h0['INSTRUME'],
                                                                      h0['DETECTOR'],
                                                                      h0['filter'],
                                                                      h0['coronmsk'],
                                                                      h0['SUBARRAY'],
                                                                      output_ext)
            
            self.image_headers = image_headers
            self.c_coron = np.array(c_coron_out)
            
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

                else:
                    raise ValueError("""
                    To load a saved SpaceReduction object, you must provide a) a file_to_load path, 
                    b) a spacerdi object, output_ext, and concat, or c) a spacerdi object with a
                                     concatenation already loaded and output_ext.      
                    """)
                file_to_load = '{}JWST_{}_{}_{}_{}_{}_{}.fits'.format(spacerdi.database.output_dir,
                                                                         h0['INSTRUME'],
                                                                         h0['DETECTOR'],
                                                                         h0['filter'],
                                                                         h0['coronmsk'],
                                                                         h0['SUBARRAY'],
                                                                         output_ext)

            self.filename = file_to_load
            hdul = fits.open(file_to_load)
            self.primary_header = hdul[0].header
            self.im = (hdul['SCI'].data if 'SCI' in hdul else None)
            self.err = (hdul['ERR'].data if 'ERR' in hdul else None)
            
            self.rolls = []
            self.err_rolls = []
            self.image_headers = []
            
            # We can usually assume either 1 or 2 rolls... but just in case:
            end_of_rolls = False
            i = 1
            while not end_of_rolls:
                if f'SCI_ROLL{i}' in hdul:
                    self.image_headers.append(hdul[f'SCI_ROLL{i}'].header)
                    if hdul[f'SCI_ROLL{i}'].data is not None:
                        self.rolls.append(hdul[f'SCI_ROLL{i}'].data)
                    if f'ERR_ROLL{i}' in hdul:
                        self.err_rolls.append(hdul[f'ERR_ROLL{i}'].data)
                    i += 1
                else:
                    end_of_rolls = True

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


    def save(self, overwrite=False):
        hdul = self.to_hdulist()
                
        try:
            hdul.writeto(self.filename, overwrite=overwrite)
        except OSError:
            raise OSError("""
                  The file you are attempting to write already exists! Use overwrite=True to overwrite
                  the existing file, or assign a different file name by changing the filename attribute.
                  """)