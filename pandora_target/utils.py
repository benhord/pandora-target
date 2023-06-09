"""Useful functions used elsewhere in the generation of the target list"""

# Third-party
import numpy as np

# Defining some constants
REarthSI = 6.3781e6 # Earth radius (m)
RJupSI = 7.149e7 # Jupiter radius (m)
RSunSI = 6.957e8 # Solar radius (m)
MEarthSI = 5.9723e24 # Earth mass (kg)
MJupSI = 1.8986e27 # Jupiter mass (kg)
MSunSI = 1.9889e30 # Solar mass (kg)
muSI = 1.67262192e-27 # Proton mass (kg)
GSI = 6.674e-11 # Gravitational constant (SI units)
kSI = 1.380649e-23 # Boltzman constant (J/K)
AUSI = 1.496e11 # Astronomical Unit (m)

def computeTSM(RpValRE, MpValME, RsRS, TeqK, Jmag):
    """
    Calculates the TSM values for a set of targets.

    Parameters
    ----------
    RpValRE : arr (floats)
        Planetary radii in units of Earth radius.
    MpValME : arr (floats)
        Mass of planets in units of Earth masses.
    RsRS : arr (floats)
        Radius of host stars in units of Solar radius.
    TeqK : arr (floats)
        Equilbrium temperature of planets in Kelvin.
    Jmag : arr (floats)
        J magnitudes of host stars.

    Returns
    -------
    TSM : arr (floats)
        TSM values of the input planets.
    """
    
    nAll = len(RpValRE)
    
    # Indices for different radii of each scale factor
    ixsA = np.arange(nAll)[np.isfinite(RpValRE)]
    ixsB = np.arange(nAll)[np.isfinite(RpValRE)==False]
    ixs1 = ixsA[(RpValRE[ixsA]<1.5)]
    ixs2 = ixsA[(RpValRE[ixsA]>=1.5) * (RpValRE[ixsA]<2.75)]
    ixs3 = ixsA[(RpValRE[ixsA]>=2.75) * (RpValRE[ixsA]<4.0)]
    ixs4 = ixsA[(RpValRE[ixsA]>=4.0) * (RpValRE[ixsA]<10.)]
    ixs5 = ixsA[(RpValRE[ixsA]>=10)]
    
    # Scale factors provided in Table 1 of Kempton et al (2018)
    c1 = 0.190
    c2 = 1.26
    c3 = 1.28
    c4 = 1.15
    c5 = 1.15 # had previously set this to 1.0
    
    # TSM before applying scale factor
    Rp3 = RpValRE**3.
    MpRs2 = MpValME * (RsRS**2.)
    y = (Rp3 * TeqK / MpRs2) * (10**(-Jmag / 5.))
    TSM = np.zeros(nAll)

    # Applying scale factor to TSM
    TSM[ixs1] = c1 * y[ixs1]
    TSM[ixs2] = c2 * y[ixs2]
    TSM[ixs3] = c3 * y[ixs3]
    TSM[ixs4] = c4 * y[ixs4]
    TSM[ixs5] = c5 * y[ixs5]
    TSM[ixsB] = np.nan
    return TSM

def computeTransSignal(RpRs, RpValRE, TeqK, MpValME, H):
    """
    Compute the expected transmission spectroscopic transit signal from a
    given set of planets.

    Parameters
    ----------
    RpRs : arr (floats)
        Ratio between the radii of the planets and the radii of their stars.
    RpValRE : arr (floats)
        Planetary radii in units of Earth radii.
    TeqK : arr (floats)
        Equilibrium temperatures of planets in Kelvin.
    MpValME : arr (floats)
        Mass value of planets in Earth Masses.
    H : float
        Number of scale heights to calculate the expected signal.

    Returns
    -------
    transit_signals : arr (floats)
        Expected transmission signal sizes of the input planets.
    """

    # Defining some constants
    RpValSI = RpValRE * np.longdouble(REarthSI)#np.longdouble(6.3781e6)
    MpValSI = MpValME * np.longdouble(MEarthSI)#np.longdouble(5.9722e24)
    
    # Filtering planets by radius
    nAll = len(RpRs)
    ixsA = np.arange(nAll)[np.isfinite(RpValRE)]
    ixsB = np.arange(nAll)[np.isfinite(RpValRE)==False]
    ixs1 = ixsA[(RpValRE[ixsA]<=1.5)]
    ixs2 = ixsA[(RpValRE[ixsA]>1.5)]
    
    # Assigning mean molecular weight of atmospheres
    mus = np.zeros(nAll)
    mus[ixs1] = 18 * muSI
    mus[ixs2] = 2.3 * muSI

    # Transit signal calculation according to Kempton, et al. 2018
    transit_signals = (H * 2 * (RpRs**2.) *
                       (RpValSI * np.longdouble(kSI) * TeqK) /
                       (mus * np.longdouble(GSI) * MpValSI))
    return transit_signals

def calcTeqK(TstarK, aRs):
    """
    Calculate the equilibrium temperature of a given planet.

    Parameters
    ----------
    TstarK : float
        Effective temperature of the host star's surface.
    aRs : float
        Ratio between the semi-major axis of the planet and the stellar radius

    Returns
    -------
    TeqK : float
        Equilibrium temperature of the planet.
    
    """
    
    TeqK = (TstarK / np.sqrt(aRs)) * (0.25**0.25)
    return TeqK


def massRadiusExoArchive(RpRE_in):
    """
    Evaluates the mean of the Chen & Kipping ( 2017 ) distribution up until
    a radius of 15 Jupiter radii, at which point it sets the mass to one
    Jupiter mass. This matches the procedure of the NASA Exoplanet Archive.

    Parameters
    ----------
    RpRE_in : arr (floats)
        Array of planetary radii in units of Earth radii.

    Returns
    -------
    MpME_out : arr (floats)
        Array of the same length as RpRE_in containing planetary masses in
        units of Earth masses.
    """

    # Power law indices:
    S1 = 0.2790
    S2 = 0.589
    #S3 = -0.044 # value quoted in Chen & Kipping (2017)
    S3 = 0.01 # mild tweak done purely for convenience
    S4 = 0.881
    # Other transition points from Table 1
    T12ME = np.log10( 2.04 )
    T23ME = np.log10( 0.414*( MJupSI/MEarthSI ) )
    T34ME = np.log10( 0.080*( MSunSI/MEarthSI ) )
    # Terran power law constant from Table 1:
    C1curl = np.log10( 1.008 )
    # Iteratively derive other power law constants:
    C2curl = C1curl + ( S1-S2 )*T12ME
    C3curl = C2curl + ( S2-S3 )*T23ME
    C4curl = C3curl + ( S3-S4 )*T34ME

    log10MpME = np.linspace( -3, 5, 1000 )

    log10M12 = np.log10( 2.04 )
    log10M23 = np.log10( 0.414*( MJupSI/MEarthSI ) )
    log10M34 = np.log10( 0.080*( MSunSI/MEarthSI ) )
    ixs1 = ( log10MpME<=log10M12 )
    ixs2 = ( log10MpME>log10M12 )*( log10MpME<=log10M23 )
    ixs3 = ( log10MpME>log10M23 )*( log10MpME<=log10M34 )
    ixs4 = ( log10MpME>log10M34 )

    log10RpRE = np.ones_like( log10MpME )
    log10RpRE[ixs1] = C1curl + ( log10MpME[ixs1]*S1 )
    log10RpRE[ixs2] = C2curl + ( log10MpME[ixs2]*S2 )
    log10RpRE[ixs3] = C3curl + ( log10MpME[ixs3]*S3 )
    log10RpRE[ixs4] = C4curl + ( log10MpME[ixs4]*S4 )

    log10MpME_out = np.interp( np.log10( RpRE_in ), log10RpRE, log10MpME )
    MpME_out = 10**log10MpME_out

    res = [idx for idx, val in enumerate(RpRE_in) if val >= 15.]
    MpME_out[res] = MJupSI/MEarthSI
    #print(res)
    #print(RpRE_in.iloc[res])
    #print(MpME_out[res])
    #print( RpRE_in, MpME_out )
        
    return MpME_out


def calcSMAx(period, MsValMS, MpValME):
    """
    Function to calculate the semi-major axis of a planet.

    Parameters
    ----------
    period : float
        Orbital period of the planet in days.
    MsValMS : float
        Mass of the host star in solar masses.
    MpValME : float
        Mass of the planet in Earth Masses.

    Returns
    -------
    smax : float
        Semi-major axis of the planet in AU.
    """
    
    # Unit conversions
    MsValSI = MsValMS * MSunSI
    MpValSI = MpValME * MEarthSI
    
    # Implementing Newton's version of Kepler's Third Law
    num = (period * 24 * 3600)**2 * GSI * (MsValSI + MpValSI)
    denom = 4 * np.pi**2
    
    smax = (num / denom)**(1 / 3) / AUSI

    return smax

def calcRatDoR(smaxAU, RsValRS):
    """
    Computes the ratio between a planet's semi-major axis and the radius of
    the host star.

    Parameters
    ----------
    smaxAU : float
        Semi-major axis of the planet in AU.
    RsValRS : float
        Radius of the host star in Solar radii.

    Returns
    -------
    ratdor : float
        Ratio between the semi-major axis and stellar radius
    """

    # Convert units to SI
    smaxSI = smaxAU * AUSI
    RsValSI = RsValRS * RSunSI

    # Compute ratio between semi-major axis and stellar radius
    ratdor = smaxSI / RsValSI

    return ratdor

def calcRatRoR(RpValRE, RsValRS):
    """
    Computes the ratio between a planet's semi-major axis and the radius of
    the host star.

    Parameters
    ----------
    RpValRE : float
        Radius of planet in Earth radii.
    RsValRS : float
        Radius of the host star in Solar radii.

    Returns
    -------
    ratror : float
        Ratio between the planetary radius and stellar radius
    """

    # Convert units to SI
    RpValSI = RpValRE * REarthSI
    RsValSI = RsValRS * RSunSI

    # Compute ratio between semi-major axis and stellar radius
    ratror = RpValSI / RsValSI

    return ratror
