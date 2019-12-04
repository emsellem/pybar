#!/usr/bin/python
"""
This module deals with bar deprojection and calculation of the corresponding in-plane velocity maps
It assumes that all motions are within the disc plane, 
following the paper by Maciejewski, Emsellem and Krajnovic 2012.
"""

# Importing the most important modules
# First numpy and mathematical functions
import numpy as np
from numpy import deg2rad, rad2deg, cos, sin, arctan, tan, pi

# Then scipy and the interpolation functions
import scipy
from scipy.interpolate import griddata as gdata
from scipy.spatial import distance

# -------------------------------------
__version__ = '0.0.2  (20/08/2017)'
# Version 0.0.2: EE - Updating
# __version__ = '0.0.1  (02/05/2017)'
# Version 0.0.1: EE - Creation
# -------------------------------------

# -----------------------------------------------------------
# First, a set of useful function to check consistency
# -----------------------------------------------------------
## Check if 1D and all sizes are consistent
def _check_allconsistency_sizes(list_arrays) :
    """Check if all arrays in list have the same size
    and are 1D. If the arrays are nD, there are all flattened.

    Parameters:
    -----------
    list_arrays: list of numpy arrays

    Return
    ------
    Boolean and input list of (flattened!) arrays
    """
    Testok = True
    ## Check formats and sizes
    if not _check_ifarrays(list_arrays) :
        print("ERROR: not all input data are arrays")
        Testok = False
    for i in range(len(list_arrays)) :
        if np.ndim(list_arrays[i]) > 1 :
            list_arrays[i] = list_arrays[i].ravel()
    if not _check_consistency_sizes(list_arrays) :
        print("ERROR: not all input data have the same length")
        Testok = False
    if not _check_ifnD(list_arrays) :
        print("ERROR: not all input data are 1D arrays")
        Testok = False

    return Testok, list_arrays

## Check if all sizes are consistent
def _check_consistency_sizes(list_arrays) :
    """Check if all arrays in list have the same size

    Parameters:
    -----------
    list_arrays: list of numpy arrays

    Return
    ------
    Boolean
    """
    if len(list_arrays) == 0 :
        return True

    return all(myarray.size == list_arrays[0].size for myarray in list_arrays)

## Check if all are arrays
def _check_ifarrays(list_arrays) :
    """Check if all items in the list are numpy arrays

    Parameters:
    -----------
    list_arrays: list of numpy arrays

    Return
    ------
    Boolean
    """
    if len(list_arrays) == 0 :
        return True

    return all(isinstance(myarray, (np.ndarray)) for myarray in list_arrays)

## Check if all are 1D arrays
def _check_ifnD(list_arrays, ndim=1) :
    """Check if all are of a certain dimension

    Parameters:
    -----------
    list_arrays: list of numpy arrays
    ndim: dimension which is expected (integer, default is 1)

    Return
    ------
    Boolean
    """
    if len(list_arrays) == 0 :
        return True

    return all(np.ndim(myarray) == ndim for myarray in list_arrays)

## Setting up the rotation matrix
def set_rotmatrix(angle=0.0) :
    """Rotation matrix given a specified angle

    Parameters:
    -----------
    angle: angle in radian. Default is 0
    """
    cosa, sina = cos(angle), sin(angle)
    return np.matrix([[cosa, sina],[-sina, cosa]])


# --------------------------------------------------
# Functions to provide reference matrices
# --------------------------------------------------
## Setting up the stretching matrix
def set_stretchmatrix(coefX=1.0, coefY=1.0) :
    """Streching matrix

    Parameters:
    -----------
    coefX, coefY : coefficients (float) for the matrix
          [coefX   0
             0   coefY]
    """
    return np.array([[coefX, 0],[0, coefY]])

def set_reverseXmatrix() :
    """Reverse X axis
    using set_strechmatrix(-1.0, 1.0)
    Return
    ------
    Stretching matrix
    """
    return set_stretchmatrix(-1.0, 1.0)

def set_reverseYmatrix() :
    """Reverse Y axis 
    using set_strechmatrix(1.0, -1.0)

    Return
    ------
    Stretching matrix
    """
    return set_stretchmatrix(1.0, -1.0)

# --------------------------------------------------
# Functions to help the sampling
# --------------------------------------------------
def guess_step(Xin, Yin, index_range=[0,100], verbose=False) :
    """Guess the step from a 1 or 2D grid
    Using the distance between points for the range of points given by
    index_range

    Parameters:
    -----------
    Xin, Yin: input (float) arrays
    index_range : tupple or array of 2 integers providing the min and max indices = [min, max]
            default is [0,100]
    verbose: default is False

    Return
    ------
    step : guessed step (float)
    """
    ## Stacking the first 100 points of the grid and determining the distance
    stackXY = np.vstack((Xin[index_range[0]:index_range[1]], Yin[index_range[0]:index_range[1]])).T
    diffXY = distance.cdist(stackXY, stackXY)

    step = np.min(diffXY[diffXY > 0])
    if verbose: print("New step will be %s"%(step))

    return step

def get_extent(Xin, Yin) :
    """Return the extent using the min and max of the X and Y arrays

    Return
    ------
    [xmin, xmax, ymin, ymax]
    """

    return [Xin.min(), Xin.max(), Yin.min(), Yin.max()]

# --------------------------------------------------
# Resampling the data and visualisation
# --------------------------------------------------
def resample_data(Xin, Yin, Zin, newextent=None, newstep=None, fill_value=np.nan, method='linear', verbose=False) :
    """Resample input data from an irregular grid
    First it derives the limits, then guess the step it should use (if not provided)
    and finally resample using griddata (scipy version).

    The function spits out the extent, and the new grid and interpolated values
    """

    ## First test consistency
    test, [Xin, Yin, Zin] = _check_allconsistency_sizes([Xin, Yin, Zin]) 
    if not test : 
        if verbose:
            print("Warning: error in resample_data, not all array size are the same")
        return None, None, None, 0

    ## Get the step and extent
    if newstep is None :
        newstep = guess_step(Xin, Yin, verbose=verbose)
    if newextent is None :
        newextent = get_extent(Xin, Yin)
    [Xmin, Xmax, Ymin, Ymax] = newextent

    dX, dY = Xmax - Xmin, Ymax - Ymin
    nX, nY = np.int(dX / newstep + 1), np.int(dY / newstep + 1)
    Xnewgrid, Ynewgrid = np.meshgrid(np.linspace(Xmin, Xmax, nX), np.linspace(Ymin, Ymax, nY))
    newZ = gdata(np.vstack((Xin, Yin)).T, Zin, np.vstack((Xnewgrid.ravel(), Ynewgrid.ravel())).T, 
            fill_value=fill_value, method=method)
    return newextent, Xnewgrid, Ynewgrid, newZ.reshape(Xnewgrid.shape)

def visualise_data(Xin, Yin, Zin, newextent=None, fill_value=np.nan, method='linear', verbose=False, newstep=None, **kwargs) :
    """Visualise a data set via 3 input arrays, Xin, Yin and Zin. The shapes of these arrays should be the
    same.

    Input
    =====
    Xin, Yin, Zin: 3 real arrays
    newextent : requested extent [xmin, xmax, ymin, ymax] for the visualisation. Default is None
    fill_value : Default is numpy.nan
    method : 'linear' as Default for the interpolation, when needed.
    """
    import matplotlib
    from matplotlib import pyplot as pl
    extent, newX, newY, newZ = resample_data(Xin, Yin, Zin, newextent=newextent, newstep=newstep, fill_value=fill_value, method=method,
            verbose=verbose)
    pl.clf()
    pl.imshow(newZ, extent=extent, **kwargs)
    return extent, newX, newY, newZ

# --------------------------------------------------
# Main class
# --------------------------------------------------
class mybar(object) :
    """Class defining the bar structure which will contain
    all the velocity map information
    """
    def __init__(self, Flux=None, Velocity=None, Xin=None, Yin=None, PAnodes=0., alphaNorth=0., PAbar=0., inclin=60.0, NE_direct=True, Origin=[0.,0.], fill_value=np.nan, method='linear') :
        """Initialisation of the class
        Flux, Velocity, XCoord, YCoord should be 1D arrays
        Angles : all in degrees

        Parameters
        ----------
        Flux : ndarray of floats. Fluxes
        Velocity : ndarray of floats. Velocities
        Xin : ndarray of floats. X axis coordinates
        Yin : ndarray of floats. Y axis coordinates

        method : {'linear', 'nearest', 'cubic'}, optional
            Method of interpolation. One of
        
            ``nearest``
              return the value at the data point closest to
              the point of interpolation.  See `NearestNDInterpolator` for
              more details.
        
            ``linear``
              tesselate the input point set to n-dimensional
              simplices, and interpolate linearly on each simplex.  See
              `LinearNDInterpolator` for more details.
        
            ``cubic`` (1-D)
              return the value determined from a cubic
              spline.
        
            ``cubic`` (2-D)
              return the value determined from a
              piecewise cubic, continuously differentiable (C1), and
              approximately curvature-minimizing polynomial surface. See
              `CloughTocher2DInterpolator` for more details.
        fill_value : float, optional
            Value used to fill in for requested points outside of the
            convex hull of the input points.  If not provided, then the
            default is ``nan``. This option has no effect for the
            'nearest' method.
        """

        test, [Flux, Velocity, Xin, Yin] =  _check_allconsistency_sizes([Flux, Velocity, Xin, Yin])
        if not test : return

        ## Getting the initial arrays - all 1D
        self.Flux = Flux
        self.Vel = Velocity
        self.Xin = Xin
        self.Yin = Yin

        ## Getting the initial angles and orientation for the field
        ## Position angles are angles (in degrees) going from the North
        ## counter-clockwise (Eastward)
        self.NE_direct = NE_direct # True when East is counter-clockwise from North
        self.alphaNorth = alphaNorth # angle defining the North (from the top of the frame)
        if inclin == 90. :
            print("ERROR: deprojecting from exactly edge-on (inclin=90deg) is not possible")
            return
        self.inclin = inclin # Inclination of the galaxy
        self.PAnodes = PAnodes # Position angle of the line of nodes
        self.PAbar = PAbar # Position angle of the photometric bar

        self.fill_value = fill_value
        self.method = method

        self.align_NorthEast()
        self.align_lineofnodes()
        self.align_bar()
        self.align_deproj_bar()
        self.get_Vrt(fill_value=fill_value, method=method)

    ## Setting up NE direct or not
    @property
    def NE_direct(self) :
        return self._NE_direct

    @NE_direct.setter
    def NE_direct(self, NE_direct) :
        self._NE_direct = NE_direct
        self._mat_direct = np.where(NE_direct, set_stretchmatrix(), set_reverseXmatrix())

    ## Setting up inclination
    @property
    def inclin(self) :
        return self._inclin

    @inclin.setter
    def inclin(self, inclin) :
        self._inclin = inclin
        self._inclin_rad = deg2rad(inclin)
        self._mat_inc = set_stretchmatrix(coefY=1./cos(self._inclin_rad))

    ## Setting up PA nodes - towards the major-axis
    @property
    def PAnodes(self) :
        return self._PAnodes

    @PAnodes.setter
    def PAnodes(self, PAnodes) :
        self._PAnodes = PAnodes
        self._PAnodes_rad = deg2rad(PAnodes)
        self._mat_lon = set_rotmatrix(self._PAnodes_rad + pi / 2.)

    ## Setting up PA bar - towards the major-axis
    @property
    def PAbar(self) :
        return self._PAbar

    @PAbar.setter
    def PAbar(self, PAbar) :
        self._PAbar = PAbar
        self._PAbar_rad = deg2rad(PAbar)
        self.PAbarlon = PAbar - self.PAnodes
        self._PAbarlon_rad = deg2rad(self.PAbarlon)
        self._PAbarlon_dep_rad = arctan(tan(self._PAbarlon_rad) / cos(self._inclin_rad))
        self._PAbarlon_dep = rad2deg(self._PAbarlon_dep_rad)
        self._mat_bar = set_rotmatrix(self._PAbar_rad + pi / 2.)
        self._mat_bardep = set_rotmatrix(self._PAbarlon_dep_rad)

    ## Setting up North-East to the top
    @property
    def alphaNorth(self) :
        return self._alphaNorth

    @alphaNorth.setter
    def alphaNorth(self, alphaNorth) :
        """Initialise the parameters in the mybar structure for alphaNorth angles
        in degrees and radian, as well as the associated transformation matrix

        Input
        -----
        alphaNorth: angle in degrees for the PA of the North direction
        """
        self._alphaNorth = alphaNorth
        self._alphaNorth_rad = deg2rad(alphaNorth)
        self._mat_NE = self._mat_direct * set_rotmatrix(self._alphaNorth_rad)

    def rotate(self, Xin=None, Yin=None, matrix=np.identity(2)) :
        """Rotation of coordinates using an entry matrix

        Input
        -----
        Xin, Yin: input grid (arrays)
        matrix : transformation matrix (matrix)

        Returns
        -------
        The rotated array
        """
        if Xin is None:
            shape = self.Xin.shape
        else :
            shape = Xin.shape
        newX, newY = np.asarray(matrix * np.vstack((np.where(Xin is None, self.Xin, Xin), np.where(Yin is None, self.Yin, Yin))), dtype=float)
        return newX.reshape(shape), newY.reshape(shape)

    def align_NorthEast(self, force=False) :
        """Get North to the top and East on the left
        """
        self.X_NE, self.Y_NE = self.rotate(matrix=self._mat_NE)

    def align_lineofnodes(self) :
        """Set the Line of Nodes (defined by its Position Angle, angle from the North
        going counter-clockwise) as the positive X axis
        """
        self.X_lon, self.Y_lon = self.rotate(matrix=self._mat_lon * self._mat_NE)

    def align_bar(self) :
        """Set the bar (defined by its Position Angle, angle from the North
        going counter-clockwise) as the positive X axis
        """
        self.X_bar, self.Y_bar = self.rotate(matrix=self._mat_bar * self._mat_NE)

    def align_deproj_bar(self) :
        """Set the bar (defined by its Position Angle, angle from the North
        going counter-clockwise) as the positive X axis after deprojection
        """
        self._mat_deproj_bar = self._mat_bardep * self._mat_inc * self._mat_lon * self._mat_NE
        self.X_bardep, self.Y_bardep = self.rotate(matrix=self._mat_deproj_bar)
        ## Mirroring the coordinates
        self.X_mirror, self.Y_mirror = self.rotate(matrix=np.linalg.inv(self._mat_deproj_bar),
                Xin=self.X_bardep, Yin=-self.Y_bardep)

    def mirror_grid(self, Xin, Yin) :
        """Find the mirrored grid (above/below major-axis)

        Input
        -----
        Xin, Yin: input grid

        Returns
        -------
        Xin -Yin
        """
        return Xin, -Yin

    def deproject_Velocities(self) :
        """Deproject Velocity values by dividing by the sin(inclination)
        """

        self.Vdep = self.Vel / sin(self._inclin_rad)

    def get_Vrt(self, fill_value=np.nan, method='linear') :
        """Compute the in-plane deprojected velocities

        Input
        -----
        fill_value : filling value when no data (default is None)
        method : method to re-sample (default is 'linear'). Options are {'linear', 'nearest', 'cubic'}.
       """
        self.deproject_Velocities()
        ## Mirroring the Velocities 
        self.V_mirror = gdata(np.vstack((self.Xin, self.Yin)).T, self.Vdep, np.vstack((self.X_mirror, self.Y_mirror)).T, 
                fill_value=fill_value, method=method)
        self.gamma_rad = np.arctan2(self.Y_bardep, self.X_bardep)
        self.Vr = (self.Vdep * cos(self._PAbarlon_dep_rad - self.gamma_rad)
                - self.V_mirror * cos(self._PAbarlon_dep_rad + self.gamma_rad)) / sin(2.* self._PAbarlon_dep_rad)
        self.Vt = (self.Vdep * sin(self._PAbarlon_dep_rad - self.gamma_rad) 
                + self.V_mirror * sin(self._PAbarlon_dep_rad + self.gamma_rad)) / sin(2.* self._PAbarlon_dep_rad)
        self.Vx = self.Vr * cos(self.gamma_rad) - self.Vt * sin(self.gamma_rad)
        self.Vy = self.Vr * sin(self.gamma_rad) + self.Vt * cos(self.gamma_rad)

    def tremaine_weinberg(self, slit_width=1.0):
        """ Get standard Tremaine Weinberg method applied on the bar

        Using X_lon, Y_lon, Flux and Velocity
        """

        fV = self.Flux * -self.Vel
        fx = self.Flux * self.X_lon

        maxY = np.maximum(np.abs(np.max(self.Y_lon)), np.abs(np.min(self.Y_lon)))
        self.nslits = np.int(maxY / slit_width - 1 / 2.)
        self.y_slits = np.linspace(-self.nslits * slit_width, self.nslits * slit_width, 
                                    2*self.nslits+1)
        sw2 = slit_width / 2.

        # Initialise arrays
        self.Omsini_tw = np.zeros(2*self.nslits+1, dtype=np.float) 
        self.dfV_tw = np.zeros_like(self.Omsini_tw)
        self.dfx_tw = np.zeros_like(self.Omsini_tw)
        self.df_tw = np.zeros_like(self.Omsini_tw)

        # For a number of slits going from 0 to max
        for i, y in enumerate(self.y_slits):
            edges = [y - sw2, y + sw2]
            selY = (self.Y_lon > edges[0]) & (self.Y_lon < edges[1])
            self.df_tw[i] = np.nansum(self.Flux[selY])
            if self.df_tw[i] != 0:
                self.dfV_tw[i] = np.nansum(fV[selY]) / self.df_tw[i]
                self.dfx_tw[i] = np.nansum(fx[selY]) / self.df_tw[i]
            else:
                self.dfV_tw[i] = 0.
                self.dfx_tw[i] = 0.
            if self.dfx_tw[i] != 0:
                self.Omsini_tw[i] = self.dfV_tw[i] / self.dfx_tw[i]
            else:
                self.Omsini_tw[i] = 0.

    def get_PatternSpeed(self, step_factor=1.0, fill_value=None, method=None, fullgrid=False) :
        """Derive the Pattern Speed on the 2D plane

        Input
        -----
        step_factor : multiplicative factor to divide the present step guessed from the input deprojected grid
                      (default is 1.0)
        fill_value : filling value when no data (default is None)
        method : method to re-sample (default is 'linear'). Options are {'linear', 'nearest', 'cubic'}.
        fullgrid: when computing the gradient, keep the input deprojected grid as reference, or shift the grid
        """
        if fill_value == None : fill_value = self.fill_value
        if method == None : method = self.method

        self._FVx = self.Flux * self.Vx
        self._FVy = self.Flux * self.Vy
        step = guess_step(self.X_bardep, self.Y_bardep)
        self.step_Omega = step
        dX = step / step_factor

        if fullgrid :
            dY = step / step_factor
            FVx_Xplus = gdata(np.vstack((self.X_bardep, self.Y_bardep)).T, self._FVx, np.vstack((self.X_bardep + dX, self.Y_bardep)).T, 
                    fill_value=fill_value, method=method)
            FVx_Xminus = gdata(np.vstack((self.X_bardep, self.Y_bardep)).T, self._FVx, np.vstack((self.X_bardep - dX, self.Y_bardep)).T, 
                    fill_value=fill_value, method=method)
            FVy_Yplus = gdata(np.vstack((self.X_bardep, self.Y_bardep)).T, self._FVy, np.vstack((self.X_bardep, self.Y_bardep + dY)).T, 
                    fill_value=fill_value, method=method)
            FVy_Yminus = gdata(np.vstack((self.X_bardep, self.Y_bardep)).T, self._FVy, np.vstack((self.X_bardep, self.Y_bardep - dY)).T, 
                    fill_value=fill_value, method=method)
            F_Xplus = gdata(np.vstack((self.X_bardep, self.Y_bardep)).T, self.Flux, np.vstack((self.X_bardep + dX, self.Y_bardep)).T, 
                    fill_value=fill_value, method=method)
            F_Xminus = gdata(np.vstack((self.X_bardep, self.Y_bardep)).T, self.Flux, np.vstack((self.X_bardep - dX, self.Y_bardep)).T, 
                    fill_value=fill_value, method=method)
            F_Yplus = gdata(np.vstack((self.X_bardep, self.Y_bardep)).T, self.Flux, np.vstack((self.X_bardep, self.Y_bardep + dY)).T, 
                    fill_value=fill_value, method=method)
            F_Yminus = gdata(np.vstack((self.X_bardep, self.Y_bardep)).T, self.Flux, np.vstack((self.X_bardep, self.Y_bardep - dY)).T, 
                    fill_value=fill_value, method=method)

            Up = (FVx_Xplus - FVx_Xminus) / dX + (FVy_Yplus - FVy_Yminus) / dY
            Down = self.X_bardep * (F_Yplus - F_Yminus) / dY - self.Y_bardep * (F_Xplus - F_Xminus) / dX
            self.Omegap = Up / Down
        else :
            ex, nX, nY, resamp_FVx = resample_data(self.X_bardep, self.Y_bardep, self._FVx, newstep=dX,
                    fill_value=fill_value, method=method)
            ex, nX, nY, resamp_FVy = resample_data(self.X_bardep, self.Y_bardep, self._FVy, newstep=dX,
                    fill_value=fill_value, method=method)
            ex, nX, nY, resamp_F = resample_data(self.X_bardep, self.Y_bardep, self.Flux, newstep=dX,
                    fill_value=fill_value, method=method)

            Up = (resamp_FVx[1:-1,2:] - resamp_FVx[1:-1,:-2]) / dX + (resamp_FVy[2:,1:-1] - resamp_FVy[:-2,1:-1]) / dX
            Down = nX[1:-1,1:-1] * (resamp_F[2:,1:-1] - resamp_F[:-2,1:-1]) / dX - nY[1:-1,1:-1] * (resamp_F[1:-1,2:] - resamp_F[1:-1,:-2]) / dX
            self.Omegap = Up / Down
            self.nX_Omega, self.nY_Omega = nX[1:-1, 1:-1], nY[1:-1, 1:-1]
