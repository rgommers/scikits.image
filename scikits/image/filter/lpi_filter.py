"""
:author: Stefan van der Walt, 2008
:license: modified BSD
"""

__all__ = ['forward', 'inverse', 'wiener', 'LPIFilter2D']
__docformat__ = 'restructuredtext en'

import numpy as np
from scipy.fftpack import fftshift, ifftshift

eps = np.finfo(float).eps

def _min_limit(x, val=eps):
    """Set array values smaller than val to val."""
    mask = np.abs(x) < eps
    x[mask] = np.sign(x[mask]) * eps

def _centre(x, oshape):
    """Return an array of oshape from the centre of x.

    Parameters
    ----------
    x : ndarray
        The data array, 2-dimensional.
    oshape : tuple
        Shape tuple, with two elements. This defines the shape of the returned
        array.

    Returns
    -------
    out : ndarray
        The output array, of shape `oshape`, with values from the center of
        `x`.

    Examples
    --------
    >>> _centre(np.arange(20).reshape(5, 4), (2, 2))
    array([[10, 11],
           [14, 15]])
    """
    start = (np.array(x.shape) - np.array(oshape)) / 2. + 1
    out = x[[slice(s, s + n) for s, n in zip(start, oshape)]]
    return out

def _pad(data, shape):
    """Pad the data to the given shape with zeros.

    Parameters
    ----------
    data : ndarray
        Input data, has to be 2-dimensional.
    shape : tuple
        Shape tuple, containing two elements, which are the desired sizes of
        each dimension after padding.

    Returns
    -------
    out : ndarray
        Output array, the input padded to `shape`.
    """
    out = np.zeros(shape)
    out[[slice(0, n) for n in data.shape]] = data
    return out



class LPIFilter2D(object):
    """Linear Position-Invariant Filter (2-dimensional)

    Parameters
    ----------
    impulse_response : callable or ndarray
        If a callable, has to have signature ``f(r, c, **filter_params)`` and
        be a function that yields the impulse response.  `r` and `c` are
        1-dimensional vectors that represent row and column positions, in other
        words coordinates are (r[0],c[0]),(r[0],c[1]) etc.  `**filter_params`
        are passed through.  In other words, the example would be called like
        this::

            r = [0,0,0,1,1,1,2,2,2]
            c = [0,1,2,0,1,2,0,1,2]
            impulse_response(r, c, **filter_params)

        If an ndarray, `impulse_response` has to be 2-dimensional and contain
        the filter kernel.
    kernel_shape : tuple, optional
        Shape tuple for the kernel. Determines the size of the kernel extracted
        from `impulse_response` if that is a callable. If `impulse_response` is
        an ndarray, `kernel_shape` is ignored.
    filter_params : dict, optional
        The extra parameters passed to `impulse_response` if that is a
        callable.

    Attributes
    ----------
    impulse_response : callable or ndarray
        Holds the `impulse_response` parameter. If modified, the user has to
        keep in mind that repeated calling with the same data shape can result
        in the cache containing the old impulse response result.
    kernel_shape : ndarray or None
        Holds the `kernel_shape` parameter. Can be modified.
    filter_params : dict
        The parameters passed to `impulse_response` if it is a callable.

    Examples
    --------
    To define a spherical Gaussian filter, we do:

    >>> def filt_func(r, c):
            return np.exp(-(r**2 + c**2)/1.)

    >>> filter = LPIFilter2D(filt_func)
    """
    def __init__(self, impulse_response, kernel_shape=None, **filter_params):
        if impulse_response is None:
            raise ValueError("Impulse response must be a callable.")

        self.impulse_response = impulse_response
        self.kernel_shape = kernel_shape
        self.filter_params = filter_params
        self._cache = None

    def _prepare(self, data):
        """Calculate filter and data FFT in preparation for filtering.

        Parameters
        ----------
        data : ndarray
            The input array, of shape ``(M, N)``.

        Returns
        -------
        F : ndarray
            The filter in Fourier space, i.e. the Fourier-transformed impulse
            response evaluated at the data coordinates.
        G : ndarray
            The Fourier-transformed input data, of shape ``(2*M-1, 2*N-1)``.

        Notes
        -----
        `F` is cached, so for input data of the same shape repeated filtering
        is fast.
        """
        dshape = np.array(data.shape)
        dshape += (dshape % 2 == 0) # all filter dimensions must be uneven
        oshape = np.array(data.shape) * 2 - 1

        if self._cache is None or np.any(self._cache.shape != oshape):
            coords = np.mgrid[[slice(0, float(n)) for n in dshape]]
            # this steps over two sets of coordinates,
            # not over the coordinates individually
            for k,coord in enumerate(coords):
                coord -= (dshape[k] - 1)/2.
            coords = coords.reshape(2, -1).T # coordinate pairs (r,c)

            f = self.impulse_response(coords[:,0],coords[:,1],
                                      **self.filter_params).reshape(dshape)

            f = _pad(f,oshape)
            F = np.dual.fftn(f)
            self._cache = F
        else:
            F = self._cache

        data = _pad(data, oshape)
        G = np.dual.fftn(data)

        return F, G

    def __call__(self, data):
        """Apply the filter to the given data.

        Parameters
        ----------
        data : ndarray
            The input array, of shape ``(M, N)``.

        Returns
        -------
        out : ndarray
            The filtered input.
        """
        F, G = self._prepare(data)
        out = np.dual.ifftn(F * G)
        out = np.abs(_centre(out, data.shape))
        return out


def forward(data, impulse_response=None, filter_params={},
            predefined_filter=None):
    """Apply the given filter to data.

    Parameters
    ----------
    data : (M,N) ndarray
        Input data.
    impulse_response : callable ``f(r, c, **filter_params)``
        Impulse response of the filter.  See LPIFilter2D.__init__.
    filter_params : dict
        Additional keyword parameters to the impulse_response function.

    Other Parameters
    ----------------
    predefined_filter : LPIFilter2D
        If you need to apply the same filter multiple times over
        different images, construct the LPIFilter2D and specify
        it here.

    Examples
    --------

    Gaussian filter:

    >>> def filt_func(r, c):
            return np.exp(-np.hypot(r, c)/1)

    >>> forward(data, filt_func)

    """
    if predefined_filter is None:
        predefined_filter = LPIFilter2D(impulse_response, **filter_params)
    return predefined_filter(data)


def inverse(data, impulse_response=None, filter_params={}, max_gain=2,
            predefined_filter=None):
    """Apply the filter in reverse to the given data.

    Parameters
    ----------
    data : (M,N) ndarray
        Input data.
    impulse_response : callable `f(r, c, **filter_params)`
        Impulse response of the filter.  See LPIFilter2D.__init__.
    filter_params : dict
        Additional keyword parameters to the impulse_response function.
    max_gain : float
        Limit the filter gain.  Often, the filter contains
        zeros, which would cause the inverse filter to have
        infinite gain.  High gain causes amplification of
        artefacts, so a conservative limit is recommended.

    Other Parameters
    ----------------
    predefined_filter : LPIFilter2D
        If you need to apply the same filter multiple times over
        different images, construct the LPIFilter2D and specify
        it here.

    """
    if predefined_filter is None:
        filt = LPIFilter2D(impulse_response, **filter_params)
    else:
        filt = predefined_filter

    F, G = filt._prepare(data)
    _min_limit(F)

    F = 1/F
    mask = np.abs(F) > max_gain
    F[mask] = np.sign(F[mask]) * max_gain

    return _centre(np.abs(ifftshift(np.dual.ifftn(G * F))), data.shape)

def wiener(data, impulse_response=None, filter_params={}, K=0.25,
           predefined_filter=None):
    """Minimum Mean Square Error (Wiener) inverse filter.

    Parameters
    ----------
    data : (M,N) ndarray
        Input data.
    K : float or (M,N) ndarray
        Ratio between power spectrum of noise and undegraded
        image.
    impulse_response : callable `f(r, c, **filter_params)`
        Impulse response of the filter.  See LPIFilter2D.__init__.
    filter_params : dict
        Additional keyword parameters to the impulse_response function.

    Other Parameters
    ----------------
    predefined_filter : LPIFilter2D
        If you need to apply the same filter multiple times over
        different images, construct the LPIFilter2D and specify
        it here.

    """
    if predefined_filter is None:
        filt = LPIFilter2D(impulse_response, **filter_params)
    else:
        filt = predefined_filter

    F, G = filt._prepare(data)
    _min_limit(F)

    H_mag_sqr = np.abs(F)**2
    F = 1/F * H_mag_sqr / (H_mag_sqr + K)

    return _centre(np.abs(ifftshift(np.dual.ifftn(G * F))), data.shape)

def constrained_least_squares(data, lam, impulse_response=None,
                              filter_params={}):
    raise NotImplementedError

