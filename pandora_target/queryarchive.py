"""Functions required to query the Exoplanet Archive and clean the results"""

# Standard library
import os
from datetime import datetime
import warnings

# Third-party
import requests
import pandas as pd
import numpy as np

from .utils import computeTSM, computeTransSignal, calcTeqK, massRadiusExoArchive, calcSMAx, calcRatDoR, calcRatRoR

pd.options.mode.chained_assignment = None  # default='warn'

def fetch_confirmed_pls(forceDownload=False, composite=True):
    """
    Fetches parameters for confirmed planets from NASA Exoplanet Archive.

    Parameters
    ----------
    forceDownload : bool
        Flag to force a new download of confirmed planets from the NASA
        Exoplanet Archive.
    composite : bool
        Flag to determine whether the composite table is queried from the
        NASA Exoplanet Archive. This table combines parameters from multiple
        references into one parameter set.

    Returns
    -------
    planets_df : DataFrame
        Confirmed planets and their parameters from the Archive.
    """
    
    date = str(datetime.date(datetime.now()))
    path = 'PS_'+date+'.csv'
    confirmedFpath = path
    
    if not forceDownload:
        if os.path.exists(f'{os.getcwd()}/{path}'):
            planets_df = pd.read_csv(path, header=0)

            if ('default_flag' not in planets_df.columns) != composite:
                warnings.warn('Loaded version of planet data does not ' +\
                              'match input composite flag! Try rerunning ' +\
                              'with forceDownload=True.', Warning,
                              stacklevel=2)
                
            return planets_df

    core_elements = [
        'pl_name','discoverymethod','disc_year','disc_facility','tran_flag',\
        'pl_orbper','pl_orbpererr1','pl_orbpererr2','pl_orbperlim',\
        'pl_orbsmax','pl_orbsmaxerr1','pl_orbsmaxerr2','pl_orbsmaxlim',
        'pl_rade','pl_radeerr1','pl_radeerr2','pl_radelim','pl_bmassprov',\
        'pl_orbeccen','pl_orbeccenerr1','pl_orbeccenerr2','pl_orbeccenlim',\
        'pl_insol','pl_insolerr1','pl_insolerr2','pl_insollim','pl_orbincl',\
        'pl_orbinclerr1','pl_orbinclerr2','pl_orbincllim','pl_imppar',\
        'pl_impparerr1','pl_impparerr2','pl_impparlim','pl_trandur',\
        'pl_trandurerr1','pl_trandurerr2','pl_trandurlim','pl_ratdor',\
        'pl_ratdorerr1','pl_ratdorerr2','pl_ratdorlim','pl_ratror',\
        'pl_ratrorerr1','pl_ratrorerr2','pl_ratrorlim','pl_eqt',\
        'pl_eqterr1','pl_eqterr2','st_teff','st_tefferr1','st_tefferr2',\
        'st_tefflim','st_rad','st_raderr1','st_raderr2','st_radlim',\
        'st_mass','st_masserr1','st_masserr2','st_masslim','st_logg',\
        'st_loggerr1','st_loggerr2','st_logglim','rastr','ra','decstr',\
        'dec','sy_dist','sy_disterr1','sy_disterr2','sy_vmag','sy_vmagerr1',\
        'sy_vmagerr2','sy_jmag','sy_jmagerr1','sy_jmagerr2','sy_hmag',\
        'sy_hmagerr1','sy_hmagerr2','sy_kmag','sy_kmagerr1','sy_kmagerr2',\
        'st_rotp','st_rotperr1','st_rotperr2',
    ]
        
    if not composite:
        default_query = ("https://exoplanetarchive.ipac.caltech.edu/TAP/" +\
                         "sync?query=select+*+from+ps+where+tran_flag+=+1"+\
                         "&format=csv")
        
        extra_elements = [
            'default_flag','pl_masse','pl_masseerr1','pl_masseerr2',\
            'pl_masselim','pl_refname','st_refname',
        ]
    else:
        default_query = ("https://exoplanetarchive.ipac.caltech.edu/TAP/" +\
                         "sync?query=select+*+from+pscomppars+where+tran_"+\
                         "flag+=+1&format=csv")
        
        extra_elements = [
            'pl_orbper_reflink', 'pl_orbsmax_reflink','pl_rade_reflink',\
            'pl_bmasse','pl_bmasseerr1','pl_bmasseerr2','pl_bmasselim',\
            'pl_bmasse_reflink','pl_insol_reflink','pl_eqt_reflink',\
            'pl_imppar_reflink','pl_trandur_reflink','pl_ratdor_reflink',\
            'pl_ratror_reflink',
        ]
    
    core_elements.extend(extra_elements)
    add_few_elements = ','.join(core_elements)
    
    all_planets =  requests.get(default_query.split('*')[0] +
                                add_few_elements +
                                default_query.split('*')[1])
    
    all_planets = all_planets.text.split('\n')
    
    planets_df = pd.DataFrame(columns=all_planets[0].split(','), 
                              data = [i.split(',') for i in all_planets[1:-1]])
    planets_df = planets_df.replace(to_replace='', value=np.nan)

    
    for i in planets_df:
        planets_df[i] = planets_df[i].str.replace('"', "")
        
    planets_df = planets_df.apply(pd.to_numeric, errors='ignore')
    if not composite:
        planets_df = planets_df[planets_df['default_flag'] == 1]
    planets_df.head()
    planets_df = planets_df.reset_index().drop(columns='index')
    planets_df.to_csv(confirmedFpath, index=False)
    
    return planets_df

def process_confirmed_pls(planets_df, H=1):
    """
    Processes the queried set of planets to fill empty values, calculate TSM,
    and calculate expected transit spectroscopic signal.

    Parameters
    ----------
    planets_df : pandas DataFrame
        DataFrame containing planet parameters from the NASA Exoplanet
        Archive to be processed.
    H : float
        Number of scale heights to be used in the calculation of the expected
        transit spectroscopic signal.
    
    Returns
    -------
    planets_df : pandas DataFrame
        DataFrame containing processed planets with all necessary parameters
        included.
    
    """
    
    # Fill empty entries or entries that are NaN with calculated values (maybe move to a separate fill_nans function?)

    composite = 'default_flag' not in planets_df.columns

    if composite:
        mass_key = 'pl_bmasse'
        mass_lim = 'pl_bmasselim'
    else:
        mass_key = 'pl_masse'
        mass_lim = 'pl_masselim'
    
    # Set any parameters that are limits to NaNs to be calculated
    ix = planets_df[planets_df[mass_lim] == 1].index.tolist()
    planets_df[mass_key].iloc[ix] = np.nan
    
    # Calculate mass for targets without a reported mass and lower uncertainty
    ix = planets_df[planets_df[mass_key].isnull()].index.tolist()
    planets_df[mass_key].iloc[ix] = massRadiusExoArchive(
        planets_df['pl_rade'].iloc[ix]
    )

    planets_df['pl_bmassprov'].iloc[ix] = 'Calculated'
    
    # Calculate semi-major axis
    ix = planets_df[planets_df['pl_orbsmax'].isnull()].index.tolist()
    planets_df['pl_orbsmax'].iloc[ix] = calcSMAx(
        planets_df['pl_orbper'].iloc[ix],
        planets_df['st_mass'].iloc[ix],
        planets_df[mass_key].iloc[ix],
    )

    if not composite:
        planets_df['pl_orbsmax_reflink'] = planets_df.loc[:, 'pl_refname']
    
    planets_df['pl_orbsmax_reflink'].iloc[ix] = 'Calculated'
    
    # Calculate ratio of semi-major axis to stellar radius
    ix = planets_df[planets_df['pl_ratdor'].isnull()].index.tolist()
    planets_df['pl_ratdor'].iloc[ix] = calcRatDoR(
        planets_df['pl_orbsmax'].iloc[ix],
        planets_df['st_rad'].iloc[ix],
    )

    if not composite:
        planets_df['pl_ratdor_reflink'] = planets_df.loc[:, 'pl_refname']
    
    planets_df['pl_ratdor_reflink'].iloc[ix] = 'Calculated'
    
    # Calculate confidence of mass measurement and remove planets below thresh
    ### Not ready yet. Determine how to handle mass NaNs first ###
    #planets_df['mass_sigma'] = (planets_df['pl_masse'] /
    #                            planets_df['pl_masseerr2'])
    #planets_df = planets_df[planets_df['mass_sigma'] >= mass_sig_thresh]

    # check rprs and recalculate if necessary

    # Calculate equilibrium temperature
    ix = planets_df[planets_df['pl_eqt'].isnull()].index.tolist()
    planets_df['pl_eqt'].iloc[ix] = calcTeqK(
        planets_df['st_teff'].iloc[ix],
        planets_df['pl_ratdor'].iloc[ix],
    )

    if not composite:
        planets_df['pl_eqt_reflink'] = planets_df.loc[:, 'pl_refname']
    
    planets_df['pl_eqt_reflink'].iloc[ix] = 'Calculated'

    # Calculate ratio of planet radius to stellar radius
    ix = planets_df[planets_df['pl_ratror'].isnull()].index.tolist()
    planets_df['pl_ratror'].iloc[ix] = calcRatRoR(
        planets_df['pl_rade'].iloc[ix],
        planets_df['st_rad'].iloc[ix],
    )

    if not composite:
        planets_df['pl_ratror_reflink'] = planets_df.loc[:, 'pl_refname']
    
    planets_df['pl_ratror_reflink'].iloc[ix] = 'Calculated'
    
    # Calculate TSM
    planets_df['TSM'] = computeTSM(
        planets_df['pl_rade'],
        planets_df[mass_key],
        planets_df['st_rad'],
        planets_df['pl_eqt'],
        planets_df['sy_jmag'],
    )

    # Save current version of DataFrame for reference later

    # Calculate expected transit signal size
    planets_df['transSignal'] = computeTransSignal(
        planets_df['pl_ratror'],
        planets_df['pl_rade'],
        planets_df['pl_eqt'],
        planets_df[mass_key],
        H,
    )

    if planets_df['transSignal'].isnull().sum() > 0:
        ix = planets_df[planets_df['transSignal'].isnull()].index.tolist()

        message = (','.join(planets_df['pl_name'].iloc[ix]) + ' have ' +\
                   'an issue with one or more of their input parameters ' +\
                   'and will not be included in the final target list!')
        warnings.warn(message, Warning, stacklevel=2)

    return planets_df

def apply_pl_constraints(
        planets_df,
        jmag_max = 11.5,
        jmag_min = 7,
        hmag_max = 11,
        period_max = 36,
        teff_max = 5000,
        tsm_min = 1,
        tran_signal_min = 0,
        mass_sig_min = 5,
):
    """
    Function to apply parameter constraints to the planetary dataset.

    Parameters
    ----------
    jmag_thresh : float
        Host stars with Jmags larger than this value will be removed.
    hmag_thresh : float
        Host stars iwth Hmags larger than this value will be removed.
    period_thresh : float
        Planets with orbital periods larger than this value will be removed.
    teff_thresh : float
        Host stars with effective temperatures larger than this value will be
        removed.
    tsm_thresh : float
        Planets with TSM values below this value will be removed.
    tran_signal_thresh : float
        Planets with an expected transit signal below this value will be
        removed. This will soon be deprecated in favor of observability
        based on the performance of Pandora from the pandora-sat package.
    mass_sig_thresh : float
        Threshold for mass measurements below which planets will be removed.
        This feature is not yet implemented.

    Returns
    -------
    planets_df : pandas DataFrame
        DataFrame containing planetary parameters conforming to the
        thresholds set.
    """
    
    # Apply threshold cuts
    planets_df = planets_df[
        (planets_df.sy_jmag < jmag_max) &
        (planets_df.sy_jmag > jmag_min) &
        (planets_df.sy_hmag < hmag_max) &
        (planets_df.pl_orbper < period_max) &
        (planets_df.st_teff < teff_max) &
        (planets_df.TSM > tsm_min) &
        (planets_df.transSignal > tran_signal_min)]

    planets_df = planets_df.reset_index().drop(columns='index')

    planets_df = (planets_df.sort_values(
        'TSM', ascending=False).reset_index().drop(columns='index'))
    
    return planets_df
