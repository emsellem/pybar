# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This is an Astropy affiliated package to create and manipulate MGE models.
This package allows the use of Multi Gaussian Expansion models (Monnet et al.
1992, Emsellem et al. 1994). It can read and write MGE input ascii files and
computes a number of basic parameters for the corresponding models.  It includes
the derivation of velocity moments via the Jeans Equations, and the generation
of positions and velocities for N body models.

WARNING: this module is evolving quickly (and may still contains some obvious bugs).
You are welcome to send comments to Eric Emsellem (eric.emsellem@eso.org).

The package provides functions to :

* Deproject bar velocity fields
* Compute the in-plane velocities (Vx, Vy)

Submodules:
===========
    pybar: 
        Main module
"""

import pybar
from pybar import *
