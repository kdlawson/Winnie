import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import astropy.units as u
from matplotlib import animation


def mpl_centered_extent(shape, cent=None, pxscale=None):
    """
    Calculate the extent of a matplotlib plot centered around a given pixel position.

    Parameters:
    ----------
    shape : tuple
        The shape of the plot as (ny, nx), where ny is the number of rows and nx is the number of columns.
    cent : array-like, optional
        The pixel center position as [x, y]. If not provided, the geometric center of the plot is assumed.
    pxscale : float or astropy quantity, optional
        The scale of the pixels. If provided, the extent will be returned in the same units as the pxscale's numerator.
        If not provided, the extent will be returned in units of pixels.

    Returns:
    -------
    extent : array-like
        The extent of the plot as [xmin, xmax, ymin, ymax]. The units of the extent depend on the value of pxscale.

    Notes:
    ------
    - If cent is not provided, the geometric center of the plot is assumed to be at [(nx-1)/2, (ny-1)/2].
    - If pxscale is provided as an astropy quantity, the extent will be scaled accordingly.
    - If pxscale is provided as a float, the extent will be scaled by multiplying with pxscale.
    """
    ny, nx = shape
    if cent is None:
        cent = (np.array([nx, ny]) - 1) / 2.

    extent = np.array([0 - cent[0] - 0.5, nx - cent[0] - 0.5, 0 - cent[1] - 0.5, ny - cent[1] - 0.5])
    if pxscale is not None:
        if isinstance(pxscale, u.quantity.Quantity):
            extent *= pxscale.value
        else:
            extent *= pxscale
    return extent


def percentile_clim(im, percentile):
    """
    Compute the color stretch limits for an image based on percentiles.

    Parameters:
    ----------
    im : array-like
        The input image.
    percentile : float or list of float
        The percentile(s) to use for computing the color stretch limits. If a single value is provided, a symmetric
        color stretch is generated spanning plus and minus the P-percentile of the absolute value of im. If two values
        are provided, they are used as the lower and upper limit percentiles.

    Returns:
    -------
    clim : array
        The lower and upper limits of the color stretch.
    """
    vals = np.unique(im)
    if np.isscalar(percentile) or len(percentile) == 1:
        clim = np.array([-1,1])*np.nanpercentile(np.abs(vals), percentile)
    else:
        clim = np.nanpercentile(vals, percentile)
    return clim


def quick_implot(im, clim=None, clim_perc=[1.0, 99.0], cmap=None,
                 show_ticks=False, lims=None, ylims=None,
                 norm=mpl.colors.Normalize, norm_kwargs={},
                 figsize=None, panelsize=[5,5], fig_and_ax=None, extent=None,
                 show=True, tight_layout=True, alpha=1.0,
                 cbar=False, cbar_orientation='vertical',
                 cbar_kwargs={}, cbar_label=None, cbar_label_kwargs={},
                 interpolation=None, sharex=True, sharey=True,
                 save_name=None, save_kwargs={}):
    """
    Plot an image or set of images with customizable options.

    Parameters:
    ----------
    im : array-like
        The input image(s) to plot. If im is a 2D array, a single panel will be created. If im is a 3D array, a row of
        panels will be created. If im is a 4D array, a grid of panels will be created. E.g., 
        im=[[im1, im2], [im3, im4], [im5, im6]] will create a plot with 3 rows and 2 columns.
    clim : str or tuple, optional
        The color stretch limits. If a string is provided, it should contain a comma-separated pair of values.
        If a tuple is provided, it should contain the lower and upper limits of the color stretch.
    clim_perc : float or list of float, optional
        The percentile(s) to use for computing the color stretch limits. If a single value is provided, a symmetric
        color stretch is generated spanning plus and minus the P-percentile of the absolute value of im. If two values
        are provided, they are used as the lower and upper limit percentiles.
    cmap : str or colormap, optional
        The colormap to use for the image.
    show_ticks : bool, optional
        Whether to show ticks on the plot.
    lims : tuple, optional
        The x-axis (and y-axis if ylims is not provided) limits of the plot.
    ylims : tuple, optional
        The y-axis limits of the plot.
    norm : matplotlib.colors.Normalize or subclass, optional
        The normalization class to use for the color mapping.
    norm_kwargs : dict, optional
        Additional keyword arguments to pass to the normalization class.
    figsize : tuple, optional
        The size of the figure in inches. If not provided, the size will be determined based on the number of panels and
        the panelsize argument.
    panelsize : list, optional
        The size of each panel in the figure. 
    fig_and_ax : tuple, optional
        A tuple containing a matplotlib Figure and Axes object to use for the plot.
    extent : array-like, optional
        The extent of the plot as [xmin, xmax, ymin, ymax].
    show : bool, optional
        Whether to show the plot or return the relevant matplotlib objects.
    tight_layout : bool, optional
        Whether to use tight layout for the plot.
    alpha : float, optional
        The transparency of the image.
    cbar : bool, optional
        Whether to show a colorbar.
    cbar_orientation : str, optional
        The orientation of the colorbar ('vertical' or 'horizontal').
    cbar_kwargs : dict, optional
        Additional keyword arguments to pass to the colorbar.
    cbar_label : str, optional
        The label for the colorbar.
    cbar_label_kwargs : dict, optional
        Additional keyword arguments to pass to the colorbar label.
    interpolation : str, optional
        The interpolation method to use with imshow.
    sharex : bool, optional
        Whether to share the x-axis among subplots.
    sharey : bool, optional
        Whether to share the y-axis among subplots.
    save_name : str, optional
        The filename to save the plot. Plot will be saved only if this argument is provided.
    save_kwargs : dict, optional
        Additional keyword arguments to pass to the save function.

    Returns:
    -------
    fig : matplotlib Figure
        The created Figure object.
    ax : matplotlib Axes or array of Axes
        The created Axes object(s).
    cbar : matplotlib Colorbar, optional
        The created Colorbar object.

    Notes:
    ------
    - If clim is a string, it should contain a comma-separated pair of values. The values can be interpretable as floats,
      in which case they serve as the corresponding entry in the utilized clim. Alternatively, they can contain a '%'
      symbol, in which case they are used as a percentile bound. For example, clim='0, 99.9%' will yield an image with a
      color stretch spanning [0, np.nanpercentile(im, 99.9)]. If clim contains a '*', the options can be multiplied.
    - If clim is not provided, the color stretch limits will be computed based on the clim_perc parameter.
    - If clim_perc is a single value, the lower and upper limits of the color stretch will be symmetric.
    - If clim_perc is a list of two values, they will be used as the lower and upper limit percentiles.
    """
    if isinstance(clim, str):
        s_clim = [i.strip() for i in clim.split(',')]
        clim = []
        for s in s_clim:
            if s.isdigit():
                clim.append(float(s))
            elif '%' in s:
                if '*' in s:
                    svals = []
                    for si in s.split('*'):
                        if '%' in si:
                            svals.append(np.nanpercentile(im, float(si.replace('%',''))))
                        else:
                            svals.append(float(si))
                    clim.append(np.prod(svals))
                else:
                    clim.append(np.nanpercentile(im, float(s.replace('%',''))))
            else:
                raise ValueError(
                    """
                    If clim is a string, it should contain a comma separating
                    two entries. These entries should be one of:
                    a) interpretable as a float, in which case they serve as the 
                    corresponding entry in the utilized clim, b) they should contain a
                    % symbol, in which case they are used as a percentile bound;
                    e.g., clim='0, 99.9%' will yield an image with a color
                    stretch spanning [0, np.nanpercentile(im, 99.9)], or c) they
                    should contain a '*' symbol, separating either of the 
                    aforementioned options, in which case they will be multiplied.
                    """)
            
    elif clim is None:
        clim = percentile_clim(im, clim_perc)
        
    if ylims is None:
        ylims = lims
        
    normalization = norm(vmin=clim[0], vmax=clim[1], **norm_kwargs)

    if np.ndim(im) in [2,3,4]:
        im_4d = np.expand_dims(im, np.arange(4-np.ndim(im)).tolist()) # Expand dimensions to 4D if not already to easily extract nrows and ncols
        nrows, ncols = im_4d.shape[0:2]
    else:
        raise ValueError("Argument 'im' must be a 2, 3, or 4 dimensional array")
    n_ims = nrows * ncols

    if fig_and_ax is None:
        if figsize is None:
            figsize = np.array([ncols,nrows])*np.asarray(panelsize)
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    else:
        fig, ax = fig_and_ax

    if n_ims == 1:
        ax, im = [ax], [im.squeeze()]
    else:
        im = np.asarray(im).reshape((nrows*ncols, *np.shape(im)[-2:]))
        ax = np.asarray(ax).flatten()

    for ax_i, im_i in zip(ax, im):
        implot = ax_i.imshow(im_i, origin='lower', cmap=cmap, norm=normalization, extent=extent, alpha=alpha, interpolation=interpolation)
        if not show_ticks:
            ax_i.set(xticks=[], yticks=[])
        ax_i.set(xlim=lims, ylim=ylims)
    if tight_layout:
        fig.tight_layout()
    if cbar:
        cbar = fig.colorbar(implot, ax=ax, orientation=cbar_orientation, **cbar_kwargs)
        cbar.set_label(cbar_label, **cbar_label_kwargs)
    if save_name is not None:
        plt.savefig(save_name, bbox_inches='tight', **save_kwargs)
    if show:
        plt.show()
        return None
    if n_ims == 1:
        ax = ax[0]
    if cbar:
        return fig, ax, cbar
    return fig, ax


def animate_quick_implot(im_cube, dur=3, titles=None, border_color='k', fig_ax_fns=None, title_pad=15, 
                          title_fontsize=None, outfile=None, save_dpi=250, save_kwargs={}, **quick_implot_kwargs):
    if titles is None: titles = np.arange(len(im_cube)).astype(str)
    else: titles = np.asarray(titles).astype(str)
    fig,ax = quick_implot(im_cube[0], show=False, **quick_implot_kwargs)
    children = ax.get_children()
    im_ind = np.where([type(child) == mpl.image.AxesImage for child in children])[0][0]
    implot = children[im_ind]
    ax.set_title(titles[0], pad=10)
    [ax.spines[key].set_edgecolor(border_color) for key in ax.spines]
    
    nframes = len(im_cube)
    
    if fig_ax_fns is not None:
        for fn in fig_ax_fns:
            fig, ax = fn(fig, ax)
        
    if outfile is not None:
        ndigits = int(np.ceil(np.log10(nframes)))
        save_frames_fmt = '.'.join(outfile.split('.')[:-1])+'_{}.png'
        
    else:
        save_frames_fmt = None
        
    def animate(ind, save=False):
        implot.set_data(im_cube[ind])
        ax.set_title(titles[ind], pad=title_pad, fontsize=title_fontsize)
        if save:
            plt.savefig(save_frames_fmt.format(str(ind).rjust(ndigits, '0')), bbox_inches='tight', dpi=save_dpi)
        return implot
    
    fig.tight_layout()
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    anim = animation.FuncAnimation(fig, animate, frames=nframes, interval=dur*1000 / len(im_cube))
    if save_frames_fmt is not None:
        import glob
        import os
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError:
            raise ModuleNotFoundError("To save an animation using this function, you must first install the optional imageio dependency.")
        frames = list(anim.frame_seq)
        for frame in frames:
            animate(frame, save=True)
        s_frames = save_frames_fmt.format('*')
        if 'format' not in save_kwargs:
            save_kwargs['format']='GIF-PIL'
        imageio.mimwrite(outfile, [imageio.imread(f) for f in np.sort(glob.glob(s_frames))], **save_kwargs)
        os.system('rm '+s_frames)
    plt.close()
    return anim