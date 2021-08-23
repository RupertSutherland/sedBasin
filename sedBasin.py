#!/usr/bin/env python
# python3
"""
    docstring
"""
import numpy as np
from scipy.optimize import newton

__author__ = "Rupert Sutherland"
__credits__ = [
    "Rupert Sutherland",
]
__license__ = "GPL"
__version__ = "1.1"
__maintainer__ = "Rupert Sutherland"
__email__ = "rupert.sutherland@vuw.ac.nz"
__status__ = "Prototype"
# -----------------------------------------------------------------------------


def decompact(depthTopNow, thicknessNow, depthTopThen, surfacePorosity,
              compactionLength):
    '''
    Decompact a sediment layer from what is observed (depthTopNow,thicknessNow) 
    to some time in the past (then), given compaction parameters for the 
    sediment layer i.e. (surfacePorosity,compactionLength).

    RETURNS: Decompacted thickness of the layer at depthTopThen

    Assume grain height remains unchanged and
    porosity follows compaction equation.  
    Compute grain height for given (depthTopNow,thicknessNow).
    Solve compaction equation for equal grain heights and
    new depthTopThen to find void height then, and hence thicknessThen.
    '''
    # ------------------------------------------------------------------
    '''
    Solve...
    DTop        = Depth to top of layer
    DBase       = Depth to base of layer
    T           = DBase - DTop  = Thickness of layer
    phi         = Surface porosity
    L           = Compaction Length    
    grainHeight = T - voidHeight
    voidHeight  = phi * L * (exp(-DTop/L)-exp(-DBase/L))
        
    grainHeightNow = grainHeightThen
       
    TNow - phi*L*(exp(-DTopNow/L)-exp(-DBaseNow/L)) = TThen - phi*L*(exp(-DTopThen/L)-exp(-DBaseThen/L))
    
    # re-arrange
    
    0 = TThen + phi*L*exp(-DBaseThen/L) + phi*L*(exp(-DTopNow/L) - exp(-DBaseNow/L) - exp(-DTopThen/L)) - TNow
    0 = TThen + phi*L*exp(-DBaseThen/L) + C
    
    # Write: DBaseThen = TThen + DTopThen
    
    0 = TThen + phi*L*exp(-DTopThen/L) * exp(-TThen/L) + C 
    0 = TThen + A*exp(B*TThen) + C 
    
    There is now only one unknown, TThen, so solve numerically    
    '''

    # Function for decompaction equation: solve by finding fX=0
    def decompact_fX(x, A, B, C):
        return (x + A * np.exp(B * x) + C)

    # first derivative
    def decompact_dfXdx(x, A, B, C):
        return (1 + A * B * np.exp(B * x))

    Dt1 = float(depthTopNow)
    Dt0 = float(depthTopThen)
    T1 = float(thicknessNow)
    L = float(compactionLength)
    phi = float(surfacePorosity)
    
    parametersMakeSense = (depthTopNow >= 0 and
                           thicknessNow > 0 and 
                           depthTopThen >= 0 and 
                           L > 0. and 
                           phi > 0. and 
                           phi < 1.)

    if  parametersMakeSense:
        # input parameters make sense
        Db1 = T1 + Dt1
        A = phi * L * np.exp(-Dt0 / L)
        B = -1.0 / L
        C = L * phi * (np.exp(-Dt1 / L) - np.exp(-Dt0 / L) -
                       np.exp(-Db1 / L)) - T1
        param = (A, B, C)

        # numerical solution by Newton-Raphson approach 
        # i.e. Parabolic Halley method
        thickness_then = newton(func=decompact_fX,
                                x0=T1,
                                fprime=decompact_dfXdx,
                                args=param,
                                tol=0.0001 * T1,
                                maxiter=50)

        return thickness_then

    else:
        # input parameters don't make sense
        return 0.0


def depth_from_twt(twt_seabed, twt_reflector, water_sound_velocity,
                   poly_coeff):
    '''
    Assume polynomial has general form:
       depth = A*T*T + B*T + C + twt_seabed/water_sound_velocity

    polynomial order is determined by the number of parameters supplied.
    poly_coeff is tuple or list of (A,B,C) parameters.
    '''
    T = twt_reflector - twt_seabed
    water_thickness = twt_seabed / water_sound_velocity
    polynomial_object = np.poly1d(poly_coeff)
    sediment_thickness = np.polyval(polynomial_object, T)

    return water_thickness + sediment_thickness


def depthBsf_from_twtBsf(twtBsf, poly_coeff):
    '''
    Assume polynomial has general form:
       depth = A*T*T + B*T + C 

    polynomial order is determined by the number of parameters supplied.
    poly_coeff is tuple or list of (A,B,C) parameters.
    '''
    polynomial_object = np.poly1d(poly_coeff)
    return np.polyval(polynomial_object, twtBsf)


def grainHeight(depthTop, thickness, surfacePorosity, compactionLength):
    '''
    Find the equivalent total height of rock grains in a sediment unit.
    Assume that compaction is irreversible.
    depthTop is the maximum depth experienced. This may be greater than 
    now observed if erosion subsequently occurred above it.

    RETURNS: the equivalent thickness of rock for the unit
    i.e. the thickness that would ultimately be achieved by full compaction. 
    '''

    return thickness - \
        voidHeight(depthTop, thickness, surfacePorosity, compactionLength)


def isostatic_rebound_sediment_thickness_under_water(
        max_depth_experienced_at_top_of_unit, thickness_now,
        original_porosity_at_surface, compaction_length, density_mantle,
        density_grain, density_water):
    '''
    Returns the isostatic rebound that would occur if you replace the sediment
    grain-rock column with an equivalent mass of water and mantle. 
    The height of new extra mantle is the uplift response.
    '''
    L = compaction_length
    phi = original_porosity_at_surface
    Hg = grainHeight(max_depth_experienced_at_top_of_unit, thickness_now, phi,
                     L)

    return Hg * (density_grain - density_water) / (density_mantle -
                                                   density_water)


def isostatic_rebound_sediment_thickness_on_land(
        max_depth_experienced_at_top_of_unit, thickness_now,
        original_porosity_at_surface, compaction_length, density_mantle,
        density_grain, density_water):
    '''
    Returns the isostatic rebound that would occur if you replace the sediment
    grain-rock column with an equivalent mass of air and mantle.
    Rebound on land is more than under water, because mass of air is negligible
    and much less than water. 
    Assume sediment is saturated with water to ground surface.
    '''
    column_mass = massSedimentColumnWet(max_depth_experienced_at_top_of_unit,
                                        thickness_now,
                                        original_porosity_at_surface,
                                        compaction_length, density_mantle,
                                        density_grain, density_water)

    return column_mass / density_mantle


def massSedimentColumnWet(max_depth_experienced_at_top_of_unit, thickness_now,
                          original_porosity_at_surface, compaction_length,
                          density_grain, density_water):
    '''
    Returns the total mass of the sediment column, including 
    grains and pore water mass.
    '''
    L = compaction_length
    phi = original_porosity_at_surface
    Hg = grainHeight(max_depth_experienced_at_top_of_unit, thickness_now, phi,
                     L)
    Hw = thickness_now - Hg

    return density_grain * Hg + density_water * Hw


def porosity(depth, surfacePorosity, compactionLength):
    '''
    Compaction porosity as a function of depth

    RETURNS:
        porosity = surfacePorosity * exp(depth/compactionLength)
    '''

    return float(surfacePorosity) * np.exp(
        -float(depth) / float(compactionLength))


def voidHeight(depthTop, thickness, surfacePorosity, compactionLength):
    '''
    Integrate the compaction equation to get total height of void space
    in a unit at depth. Assume that compaction is irreversible.
    Depth to top of unit is the maximum depth experienced.
    This may be greater than now observed if there was erosion above it.

    RETURNS: equivalent thickness of void space for the unit. 
    '''
    dTop = np.array(depthTop, dtype=float)
    dBase = np.array(depthTop + thickness, dtype=float)
    phi = float(surfacePorosity)
    L = float(compactionLength)

    return phi * L * (np.exp(-dTop / L) - np.exp(-dBase / L))
