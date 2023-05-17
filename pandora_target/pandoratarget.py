"""Generates a target list for Pandora"""

# Third-party
import astropy.units as u
import numpy as np
import pandas as pd

from .queryarchive import fetch_confirmed_pls, process_confirmed_pls, apply_pl_constraints
from .utils import computeTransSignal

# @dataclass
class TargetList(object):
    """
    Generates and holds the information on the Pandora target list.

    Args:
        TBD : TBD
    """

    def __init__(
            self,
            jmag_max: u.Quantity = 11.5,
            jmag_min: u.Quantity = 7,
            hmag_max: u.Quantity = 11,
            period_max: u.Quantity = 36,
            starRot: u.Quantity = None,
            teff_max: u.Quantity = 5000,
            H: u.Quantity = 5,
            massSigma_min: u.Quantity = 0, #change to 5 for release
            tsm_min: u.Quantity = 1,
            tranSig_min: u.Quantity = 0,
            toplength: u.Quantity = 25,
            forceDownload: bool = False,
            composite : bool = True,
            input_file = None,
    ):
        self.jmag_max = jmag_max
        self.jmag_min = jmag_min
        self.hmag_max = hmag_max
        self.period_max = period_max
        self.starRot = starRot
        self.teff_max = teff_max
        self.tsm_min = tsm_min
        self.tranSig_min = tranSig_min
        self.H = H
        self.massSigma_min = massSigma_min
        self.toplength = toplength

        self.run_status = True
        self.removed_targs = []
        self.added_targs = []

        if input_file is None:
            self.all_pls = fetch_confirmed_pls(
                forceDownload=forceDownload,
                composite=composite
            )
        else:
            self.all_pls = pd.read_csv(input_file, header=0, comment='#')

        self.composite = 'default_flag' not in self.all_pls.columns

        self.all_pls = process_confirmed_pls(self.all_pls, H = self.H)
        self.all_pls['manual_add'] = np.zeros(len(self.all_pls['pl_name']))
        
        self.top_pls = apply_pl_constraints(
            self.all_pls,
            jmag_max = self.jmag_max,
            jmag_min = self.jmag_min,
            hmag_max = self.hmag_max,
            period_max = self.period_max,
            teff_max = self.teff_max,
            tsm_min = self.tsm_min,
            tran_signal_min = self.tranSig_min,
            mass_sig_min = self.massSigma_min,
        )
        
        #print(self.top_pls)

        # generate shortlist of top 25 targets
        self.shortlist = self.top_pls.head(self.toplength)

    def show_top(self, number=25):
        """
        Method to show the top N Pandora targets.

        Parameters
        ----------

        Returns
        -------
        
        """

        if number == 'all':
            print(self.top_pls)
        else:
            print(self.shortlist)

        #if number > len(shortlist):
        #   function to add next targets to shortlist
        #else:
        #   only display top number on shortlist, including manual adds

    def show_params(self, keys=None):
        """
        Method to print the current set of constraints and their values.

        Parameters
        ----------
        keys : list or None
            List of keys to show values for. If None, a default set will be
            printed.
        """

        if keys is None:
            keys = ['jmag_max', 'jmag_min', 'hmag_max', 'period_max', 'starRot',
                    'teff_max', 'tsm_min', 'tranSig_min', 'H', 'massSigma_min',
                    'run_status', 'composite']
        
        params = dict((key, self.__dict__[key]) for key in keys
                      if key in self.__dict__)

        print(params)
        
    def rerun(self):
        """
        Method to rerun target list using current specified parameters.
        """

        if self.run_status:
            print('Current target list is already up to date!')

        else:
            # Re-apply parameter constraints
            self.top_pls = apply_pl_constraints(
                self.all_pls,
                jmag_max = self.jmag_max,
                jmag_min = self.jmag_min,
                hmag_max = self.hmag_max,
                period_max = self.period_max,
                teff_max = self.teff_max,
                tsm_min = self.tsm_min,
                tran_signal_min = self.tranSig_min,
                mass_sig_min = self.massSigma_min,
            )

            # Recalculate expected spectroscopic signal size in case H changed
            if self.composite:
                mass_key = 'pl_bmasse'
            else:
                mass_key = 'pl_masse'

            #change this to calculate signal for all_pls to better add planets?
            self.top_pls['transSignal'] = computeTransSignal(
                self.all_pls['pl_ratror'],
                self.all_pls['pl_rade'],
                self.all_pls['pl_eqt'],
                self.all_pls[mass_key],
                self.H,
            )

            self.shortlist = self.top_pls.head(self.toplength)
            
            # add in targets on the added_targs list to shortlist
            
            # Denote that the current parameters have been run with run_status
            self.__dict__.update(run_status=True)
        
        
    def update(self, rerun=False, **params):
        """
        Method to update parameters for the target list.

        Parameters
        ----------
        **params
            Arguments corresponding to parameters and the values that you'd
            like to update them to.
        run : bool
            Flag to determine whether the data set will be remade according
            to the updated parameters.
        """

        # Update specified parameters with specified values
        self.__dict__.update(params)

        # Denote that the current parameters have been run with run_status
        self.__dict__.update(run_status=False)

        # Rerun target list with updated parameters if run=True
        if rerun:
            self.rerun()

    def add(self, planet):
        """
        Add a planet manually to the shortlist of top planets.

        Parameters
        ----------

        Returns
        -------
        """

        #add functionality to parse list of planets or dicts
        #functionality might need to change if a radius distribution is
        #   trying to be kept. Maybe identify which radius bin the desired
        #   planet is in and remove planet specifically from that bin?

        #can probably change this to treat everything the same and just do one
        #   big check to find the indices
        if type(planet) is str:
            planet = [planet]
            
        if type(planet) is list:
            try:
                # Search for planet and find its row in all_pls
                ix = (self.all_pls[
                    self.all_pls['pl_name'].isin(planet)].index)
                    
                row = self.all_pls.iloc[ix]
                name = planet
                
            except:
                print(str(planet[i]) + ' not found! Try another common' +\
                      ' name or double check the spelling.')
                #fix error message and above to skip bad names
                
        elif type(planet) is dict:
            row = planet
            name = row['pl_name']

        # Remove bottom planet based on TSM - find radius here?
        self.shortlist = self.shortlist.drop(
            self.shortlist[
                self.shortlist['manual_add'] == 0
            ].nsmallest(len(planet), 'TSM').index
        )

        # Add in new planet to shortlist
        #self.shortlist = self.shortlist.append(row, ignore_index=True)
        self.shortlist = pd.concat([self.shortlist, row], ignore_index=True)
        
        # Set manual_add flag to 1 for added target
        self.shortlist.loc[
            self.shortlist.tail(len(planet)).index,
            'manual_add',
        ] = 1

        # Reset order of planets based on TSM
        self.shortlist = (self.shortlist.sort_values(
            'TSM', ascending=False).reset_index().drop(columns='index'))
        
        # Check if added planet is on removed list and take it off that list
        self.removed_targs = [i for i in self.removed_targs if i not in planet]

    def remove(self, planet):
        """
        Removes the specified planet(s) from the shortlist.

        Parameters
        ----------
        planet : str or list
            Planet(s) that you want to remove from the shortlist.

        Returns
        -------
        """

        if type(planet) is str:
            planet = [planet]
            
        # Search for planet(s) and delete row(s) in shortlist
        ix = (self.shortlist[self.shortlist['pl_name'].isin(planet)].index)
        self.shortlist.drop(ix, axis=0, inplace=True)

        # Add removed planet to removed_targs and remove from added_targs
        self.removed_targs.extend(planet)
        self.added_targs = [i for i in self.added_targs if i not in planet]
        
        # Add in planet(s) with next highest TSM
        #from top_pls add highest not in shortlist and not in removed_targs
        ix = (self.top_pls[
            (~self.top_pls['pl_name'].isin(self.shortlist['pl_name']))
            & (~self.top_pls['pl_name'].isin(self.removed_targs))
        ].nlargest(len(planet), 'TSM').index)
        self.shortlist = pd.concat([self.shortlist, self.top_pls.iloc[ix]],
                                   ignore_index=True)
        
        #throw warning for planet entries not in self.shortlist.pl_name

        # Reset order of planets based on TSM
        self.shortlist = (self.shortlist.sort_values(
            'TSM', ascending=False).reset_index().drop(columns='index'))

    def to_csv(self, name='pandora_targets.csv'):
        """
        Save the current shortlist as a CSV file.

        Parameters
        ----------
        
        """
        
        self.shortlist.to_csv(name, index=False)

    def extend(self, number=1):
        """
        Extends the length of the shortlist by a specified number of targets.

        Parameters
        ----------
        
        """

        ix = (self.top_pls[
            (~self.top_pls['pl_name'].isin(self.shortlist['pl_name']))
            & (~self.top_pls['pl_name'].isin(self.removed_targs)) &
            (~self.top_pls['pl_name'].isin(self.added_targs))
        ].nlargest(number, 'TSM').index)
        
        self.shortlist = pd.concat([self.shortlist, self.top_pls.iloc[ix]],
                                   ignore_index=True)

        self.toplength += number

        # Reset order of planets based on TSM
        self.shortlist = (self.shortlist.sort_values(
            'TSM', ascending=False).reset_index().drop(columns='index'))

        #depending on radius, teff, etc. distribution scheme, identify which
        #   group needs to be pulled from and slice top_pls like that to pull
        #   from in this method
        #iteratively identify targets that are best to observe until one is
        #   found that's not on the removed_targs or added_targs lists or
        #   already present in the shortlist (while loop)
        #update self.toplength += to number

        #OR

        #just make this wrap self.show_top

        #OR

        #just get rid of show_top since the user can just print shortlist

        #OR

        #keep both and change show_top() to be able to show slices of top_pls
        
        

    #method to save current shortlist as csv

    #method to manually remove target

    #method to clear removed targets? or just have user add them back in

    #method to schedule using pandora-schedule

    #add option to pull shortlist targets from different radius bins
