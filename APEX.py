import cobaya as cb
import pyccl as ccl
import sacc as sc 
import numpy as np
from matplotlib import pyplot as plt
from cobaya.likelihood import Likelihood
from mpi4py import MPI
from cobaya.log import LoggedError
import pandas as pd
from astropy.io import fits

#extension imports
from Plots import Plots
from InputHandler import estimate_error

class MCMCWorkspace:
    """
    A class to handle MCMC runs with a given model and likelihood function. Aims to encapsulate the MCMC configuration and execution, giving a more streamlined interface for setting up and running MCMC simulations. 
    """


    def __init__(self, sacc_file=None, model=None, likelihood_function=None, full_info=None, params=None):
        ''' Initializes the MCMC workspace with a SACC file, model, and likelihood function.

        Parameters:
        sacc_file: str, path to the SACC file containing the data
        model: Model object, the model to be used in the MCMC run
        likelihood_function: str, name of the likelihood function to be used - should be a methof of the model object
        full_info: dict, full information about the MCMC run, including parameters, priors, references, proposals, and grouped parameters - allows for a setup akin to cobaya's native MCMC configuration.
        params: list of str, names of the parameters to be used in the MCMC run - if full_info is provided, this will be ignored.
        
        Note: If full_info is provided, the MCMC configuration will be set based on that information, otherwise it will be set based on the provided parameters, priors, references, and proposals. MCMC_config can be called later to update the configuration.
        '''



        self.sacc_file = sacc_file

        try:
            self.likelihood_function = getattr(model, likelihood_function)

            self.data = sc.Sacc.load_fits(sacc_file)
        except TypeError:
            print("Initializing without external model or likelihood function.")

        self.info = {}

        self.full_info = full_info

        self.params = params

        if full_info is not None:
            self.MCMC_config(None)

    def set_param_priors(self, params, priors):
        """ Set the priors for the parameters.

        Parameters:
        params: list of str, names of the parameters
        priors: list of tuples, each tuple containing the min and max values for the corresponding parameter
        """

        self.params_with_priors = {}
        for param in params:
            self.params_with_priors[param] = priors[params.index(param)]

    def set_param_references(self, params, references):
        """ Set the references for the parameters.

        Parameters:
        params: list of str, names of the parameters
        references: list of floats, each float is the reference value for the corresponding parameter
        """

        self.params_with_references = {}
        for param in params:
            self.params_with_references[param] = references[params.index(param)]

    def set_param_proposals(self, params, proposals):
        """ Set the proposals for the parameters.

        Parameters:
        params: list of str, names of the parameters
        proposals: list of floats, each float is the proposal value for the corresponding parameter
        """

        self.params_with_proposals = {}
        for param in params:
            self.params_with_proposals[param] = proposals[params.index(param)]

    def set_grouped_params(self, grouped_dict):
        """
        Set the grouped parameters for the MCMC run. Assigns a lamda function to each group of parameters, as cobaya expects.

        Parameters:
        grouped_dict: Dictionary where keys are parameter names and values are lists of parameters that should be grouped together.
        """
        self.grouped_params = []
        for key, value in grouped_dict.items():
               arg_string = ", ".join(grouped_dict[key])
               lambda_string = f"lambda {arg_string}: [{arg_string}]"

               func = eval(lambda_string)
               func.__module__ = __name__
               
               self.grouped_params.append((key, {"value": func, "derived": False}))
            


    def MCMC_config(self, params, sampler_info={'mcmc': {"max_tries": 10000, "proposal_scale": 1.5}}):

        """ Configure the MCMC run with the given parameters and sampler information.
        
        Parameters:
        params: list of str, names of the parameters to be used in the MCMC run
        sampler_info: dict, information about the MCMC sampler, such as max_tries and proposal_scale
        """

        if self.full_info is not None:
            self.info = self.full_info
        
        else:
            if self.params_with_references is None:
                self.params_with_references = {param: None for param in params}
            
            if self.params_with_proposals is None:
                self.params_with_proposals = {param: None for param in params}
            
            if self.params_with_priors is None:
                print("Please set the priors for the parameters before running MCMC.")
                return


            self.info['likelihood'] = {f'{self.likelihood_function}': self.likelihood_function}

            param_list = [
                (f'{param}', 
                {
                'prior': {"min": self.params_with_priors[param][0], "max": self.params_with_priors[param][1]},
                'ref': self.params_with_references[param],
                'proposal': self.params_with_proposals[param]
                })
                for param in params
            ]



            param_list = param_list + self.grouped_params if hasattr(self, 'grouped_params') else param_list
            #print("Parameters with priors:", param_list)
            
            self.info['params'] = dict(param_list)


            self.info['sampler'] = sampler_info


    def print_config(self):
        """ Print the MCMC configuration, including the sacc file, likelihood function, parameters with priors, references, proposals, and grouped parameters. """

        print("MCMC Configuration:")
        print(f"Sacc file: {self.sacc_file}")
        print(f"Likelihood function: {self.likelihood_function}")
        print("Parameters with priors:")
        for param, prior in self.params_with_priors.items():
            print(f"{param}: {prior}")
        print("Parameters with references:")
        for param, ref in self.params_with_references.items():
            print(f"{param}: {ref}")
        print("Parameters with proposals:")
        for param, proposal in self.params_with_proposals.items():
            print(f"{param}: {proposal}")
        if hasattr(self, 'grouped_params'):
            print("Grouped parameters:")
            for group in self.grouped_params:
                print(group)

        print(self.info)

    def serial_run(self):
        """ Run the MCMC simulation in serial mode. 
        
        Returns:
        updated_info: dict, updated information after running the MCMC simulation
        sampler: cobaya.sampler.Sampler, the sampler object containing the results of the MCMC run
        """

        updated_info, self.sampler = cb.run(self.info)
        return updated_info, self.sampler
    
    def mpi_run(self):
        """ Run the MCMC simulation in MPI mode. This method is designed to be run in a distributed environment using MPI. 
        It initializes the MPI communicator, runs the MCMC simulation, and gathers the results across all processes. 
        
        
        Returns: 
        upd_info: dict, updated information after running the MCMC simulation
        sampler: cobaya.sampler.Sampler, the sampler object containing the results of the MCMC run
        """
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()

        success = False
        try:
            upd_info, self.sampler = cb.run(self.info)
            success = True

        except LoggedError as err:
            pass

        success = all(comm.allgather(success))

        #save chain information to txt file

       
        if not success and self.rank == 0:
            print("MCMC run failed. Check the logs for details.")

        return upd_info, self.sampler

    def minimize_run(self):
        """ Run the MCMC simulation in a minimized mode, to find best fit parameters without full MCMC sampling.
        
        Returns:
        sampler: cobaya.sampler.Sampler, the sampler object containing the results of the minimized MCMC run
        """
        upd_info, sampler = cb.run(self.info, minimize=True)

        return sampler

    def save_chain(self, name_root='', mpi=False):
        """
        Save the chain to a file. The output file will resemble a native cobaya chain file.
        
        Parameters:
        name_root: str, root name for the file
        mpi: bool, whether to save the file with a rank suffix for MPI runs
        """
        if mpi:
            name = f'{name_root}{self.rank}.txt'
        else:
            name = f'{name_root}.txt'

        if self.sampler is not None:
            chain = self.sampler.samples().data

            df = pd.DataFrame(chain)
            formatted_chain = df.to_string(index=False, header=True)
            
            with open(name, 'w') as f:
                f.write(formatted_chain)

    def corner_plot(self, params_to_plot):
        """ Create a corner plot for the given parameters.

        Parameters:
        params_to_plot: list of str, names of the parameters to be plotted
        """

        import getdist.plots as gdplt
        gdsamples = self.sampler.samples(combined=True, to_getdist=True, skip_samples=0.3)

        gdplot = gdplt.get_subplot_plotter(width_inch=5)
        gdplot.triangle_plot(gdsamples,params_to_plot, filled=True)
        
        #save the plot
        gdplot.export(f'corner_plot{params_to_plot}.png')

    def getdist_load(self, name_root=''):
        """ Load the chain data using getdist and create a corner plot.

        Parameters:
        name_root: str, root name for the file

        note: not tested, doubt it works. WILL UPDATE LATER.
        """

        import getdist
        import getdist.plots as gdplt

        chain_data = getdist.MCSamples(name_root)
        
        gdplot = gdplt.get_subplot_plotter(width_inch=5)
        gdplot.triangle_plot(chain_data, filled=True)

        #save the plot
        gdplot.export(f'{name_root}_corner_plot.png')

    from InputHandler import estimate_error



class GalaxyDensityTracers:
    """ A class that represents a galaxy density tracer. It contains all the relevant information about the tracer."""

    def __init__(self, name, index, z, nz, sacc_file):
        """ Initializes the GalaxyDensityTracers object with the given parameters.

        Parameters:
        name: str, name of the tracer
        index: int, index of the tracer
        z: numpy array, redshift values for the tracer
        nz: numpy array, number of data points at each redshift for the tracer
        sacc_file: str, path to the SACC file containing the data for the tracer
        """

        self.z, self.nz = z, nz
        self.name = f'{name}{index}' 
        self.s = sc.Sacc.load_fits(sacc_file)

    def get_c_ells(self, s):
        """
        Get the C_ell for this tracer.
        
        Parameters:
        s: sacc.Sacc object
        
        Returns:
        c_ell: numpy array, C_ell for this tracer
        """
        _, c_ell = s.get_ell_cl(None, self.name, self.name, return_cov=False)
        return c_ell

    def get_ells(self, s):
        """
        Get the ell values for this tracer.
        
        Parameters:
        s: sacc.Sacc object
        
        Returns:
        ell: numpy array, ell values for this tracer
        """
        ell, _ = s.get_ell_cl(None, self.name, self.name, return_cov=False)
        return ell
    
    def get_cut_data(self, sacc_workspace, ell_min=100, ell_max=1000, tracer_2=None):
        """
        Get the C_ell for this tracer, cut to a specific range.
        
        Parameters:
        sacc_workspace: sacc.Sacc object
        ell_min: int, minimum ell value
        ell_max: int, maximum ell value
        tracer_2: str, name of the second tracer (optional)
        
        Returns:
        ell[mask]: numpy array, ell values for this tracer in the specified range
        cl[mask]: numpy array, C_ell for this tracer in the specified range
        mask: numpy array, boolean mask indicating which ell values are in the specified range
        """
        s = sacc_workspace.data

        tracer = self.name

        #print(f"Getting C_ell for tracer {self.name} and {tracer_2} in range {ell_min}-{ell_max}")

        reverse_order = sacc_workspace.reverse_order

        if self.name in sacc_workspace.aliases:
            tracer = sacc_workspace.aliases[self.name]
        if tracer_2 in sacc_workspace.aliases:
            tracer_2 = sacc_workspace.aliases[tracer_2]

        #print(f"Getting C_ell for tracer {tracer} and {tracer_2} in range {ell_min}-{ell_max}")

        if tracer_2 is None:
            #print(f"Getting C_ell for tracer {tracer} in range {ell_min}-{ell_max}")
            ell, cl = s.get_ell_cl(None, tracer, tracer, return_cov=False)

        else:

            if reverse_order:
                tracer, tracer_2 = tracer_2, tracer

            #print(f"Getting C_ell for tracer {tracer} and {tracer_2} in range {ell_min}-{ell_max}")
            ell, cl = s.get_ell_cl(None, tracer, tracer_2, return_cov=False)

        
        mask = (ell >= ell_min) & (ell <= ell_max)
        return ell[mask], cl[mask], mask

    def get_z(self):
        """ Get the redshift values for this tracer. 
        
        Returns: 
        z: numpy array, redshift values for this tracer
        """
        return self.z

    def get_nz(self):
        """ Get the nz values for this tracer. 
        
        Returns:
        nz: numpy array, nz values for this tracer
        """
        return self.nz

class GalaxyDensityTracerWorkspace:
    """ A class that handles the workspace for galaxy density tracers. It loads the SACC file and defines the tracers based on the provided parameters.
    Works in conjunction with the GalaxyDensityTracers class to provide a more structured way to handle galaxy density tracers in cosmological analyses."""
    def __init__(self, sacc_file, tracer_name_root, cosmology, single_index=None, max_index=None):
        """ Initializes the GalaxyDensityTracerWorkspace with the given parameters.

        Parameters:
        sacc_file: str, path to the SACC file containing the data
        tracer_name_root: str, root name for the tracers
        cosmology: pyccl.core.Cosmology object, the cosmology to be used in the analysis
        single_index: int, index of the tracer to be used (optional) - if provided, only this tracer will be defined
        max_index: int, maximum index of the tracers to be defined (optional) - if provided, tracers from 0 to max_index will be defined
        """

        self.sacc_file = sacc_file
        self.tracer_name_root = tracer_name_root
        self.max_index = max_index
        self.cosmology = cosmology
        self.single_index = single_index
        
        self.data = sc.Sacc.load_fits(sacc_file)

    def get_kernel(self, i):
        """ Get the redshift and nz values for a specific tracer index, without a need to instantiate the GalaxyDensityTracers class.

        Parameters:
        i: int, index of the tracer

        Returns:
        z: numpy array, redshift values for the tracer
        nz: numpy array, nz values for the tracer
        """

        z = self.data.get_tracer(f'{self.tracer_name_root}{i}').z
        nz = self.data.get_tracer(f'{self.tracer_name_root}{i}').nz

        return z, nz

    def generate_tracer_objects(self):
        """ Generate the GalaxyDensityTracers objects for all indices up to max_index. This method is called when defining the tracer dictionary.
        
        Returns:
        tracers_obj: list of GalaxyDensityTracers objects, each corresponding to a tracer index from 0 to max_index
        """
        self.tracers_obj = []

        if self.single_index is None:
            for i in range(self.max_index+1):
                z, nz = self.get_kernel(i)
                self.tracers_obj.append(GalaxyDensityTracers(self.tracer_name_root, i, z, nz, self.sacc_file))
        elif self.single_index is not None:
            z, nz = self.get_kernel(self.single_index)
            self.tracers_obj.append(GalaxyDensityTracers(self.tracer_name_root, self.single_index, z, nz, self.sacc_file))
        
        return self.tracers_obj

    def define_tracer_dict(self):
        """ Define a dictionary of tracers, where the keys are the tracer names and the values are the corresponding GalaxyDensityTracers objects. 
        
        Returns:
        tracers_dict: dict, keys are tracer names and values are GalaxyDensityTracers objects
        """

        self.generate_tracer_objects()

        self.tracers_dict = {
            t.name: ccl.NumberCountsTracer(
                self.cosmology,
                has_rsd=False,
                dndz=(t.z, t.nz),
                bias=(t.z, (1 * np.ones_like(t.z)))  
            ) for t in self.tracers_obj
        }
        
        return self.tracers_dict
    
    
    def return_measured_auto_c_ells(self):
        """
        Return the measured auto C_ells for all tracers.
        
        Returns:
        measured_c_ells: list of numpy arrays, each containing the measured C_ells for a tracer
        """
        self.tracers_dict = self.define_tracer_dict()

        measured_c_ells = []
        for tracer in self.tracers_dict:
            ell, cl = self.data.get_ell_cl(None, tracer, tracer, return_cov=False)
            measured_c_ells.append(cl)
        
        return measured_c_ells

    
class CIBIntensityTracers:
    """ A class that represents a CIB intensity tracer. It contains all the relevant information about the tracer, including the redshift, effective source flux, and the kernel for the tracer."""

    def __init__(self, tracer_name_root, index, cosmo, snu_z, z_arr, z_min=0., z_max=6.):
        """ Initializes the CIBIntensityTracers object with the given parameters.

        Parameters:
        tracer_name_root: str, root name for the tracer
        index: int, index of the tracer
        cosmo: pyccl.core.Cosmology object, the cosmology to be used in the analysis
        snu_z: numpy array, effective source flux for the tracer in units of Jy/Lsun
        z_arr: numpy array, redshift values to compute the kernel
        z_min: float, minimum redshift down to which the kernel is defined (default: 0.0)
        z_max: float, maximum redshift up to which the kernel is defined (default: 6.0)
        """
        
        self.cosmo = cosmo
        self.snu_z = snu_z
        self.z_arr = z_arr
        self.z_min = z_min
        self.z_max = z_max

        self.tracer = self.CIBnuTracer()


        self.name = f'{tracer_name_root}{index}'


    def CIBnuTracer(self):
        from scipy.interpolate import interp1d
        """Specific :class:`Tracer` associated with the cosmic infrared
        background intensity at a specific frequency nu. 
        The radial kernel for this tracer is

        .. math::
        W(\\chi) = \\frac{\\chi^{2} S_\\nu^{eff}}{K}.

        See Eq. 12 of https://arxiv.org/pdf/2206.15394. This should be
        combined with a 3D power spectrum of the star formation rate density
        in units of :math:`M_\\odot\\,{\\rm Mpc}^{-3}\\,{\\rm yr}^{-1}` to
        yield a 2D power spectrum for the CIB in units of :math:`{\\rm Jy}`.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): Cosmology object.
            snu_z (array): effective source flux for one frequency in units of
                :math:`{\\rm Jy}\\,{\\rm L_{Sun}}^{-1}\\.
            z_arr (array): redshift values to compute chi_z
            zmin (float): minimum redshift down to which we define the
                kernel.
            zmax (float): maximum redshift up to which we define the
                kernel. zmax = 6 by default (reionization)

        Thanks to Dr David Alonso for the original implementation.
        """
        tracer = ccl.Tracer()
        chi_max = ccl.comoving_radial_distance(self.cosmo, 1./(1+self.z_max))
        chi_min = ccl.comoving_radial_distance(self.cosmo, 1./(1+self.z_min))
        chi_z = ccl.comoving_radial_distance(self.cosmo, 1./(1+self.z_arr))
        # in Jy/Lsun
        # see https://github.com/abhimaniyar/halomodel_cib_tsz_cibxtsz/blob/1f6cc5c4fdbb8f1e0d04aa8301d38d1a43100cfa/input_var.py#L82C44-L82C56
        snu_inter = interp1d(chi_z, self.snu_z, kind='linear',
                            bounds_error=False, fill_value="extrapolate")
        chi_arr = np.linspace(chi_min, chi_max, len(self.snu_z))
        snu_arr = snu_inter(chi_arr)
        K = 1.0e-10  # Kennicutt constant in units of M_sun/yr/L_sun
        w_arr = chi_arr**2*snu_arr/K
        tracer.add_tracer(self.cosmo, kernel=(chi_arr, w_arr))

        return tracer
    
    def get_beam_window(self):
        pass #to be added

    def get_cut_data(self, sacc_workspace, ell_min=100, ell_max=1000, tracer_2=None, chi_max=None):
        """
        Get the C_ell for this tracer, cut to a specific range.
        
        Parameters:
        s: sacc.Sacc object
        ell_min: int, minimum ell value
        ell_max: int, maximum ell value
        tracer_2: str, name of the second tracer (optional)
        chi_max: float, maximum comoving distance (optional) - if provided, the C_ell will be cut to this distance
        
        Returns:
        c_ell: numpy array, C_ell for this tracer in the specified range
        """
        #print(f"Getting C_ell for tracer {self.name} and {tracer_2} in range {ell_min}-{ell_max}")

        s = sacc_workspace.data

        tracer = self.name

        if self.name in sacc_workspace.aliases:
            tracer = sacc_workspace.aliases[self.name]
        if tracer_2 in sacc_workspace.aliases:
            tracer_2 = sacc_workspace.aliases[tracer_2]

        #print(f"Getting C_ell for tracer {tracer} and {tracer_2} in range {ell_min}-{ell_max}")
        
        if tracer_2 is None:
            #print(f"Getting C_ell for tracer {tracer} in range {ell_min}-{ell_max}")
            ell, cl = s.get_ell_cl(None, tracer, tracer, return_cov=False)
        else:
    
            #print(f"Getting C_ell for tracer {tracer} and {tracer_2} in range {ell_min}-{ell_max}")
            ell, cl = s.get_ell_cl(None, tracer, tracer_2, return_cov=False)

        
        mask = (ell >= ell_min) & (ell <= ell_max)
        return ell[mask], cl[mask], mask

class CIBIntensityTracerWorkspace:
    """ A class that handles the workspace for CIB intensity tracers. It loads the flux fits file and defines the tracers based on the provided parameters."""

    def __init__(self, flux_fits_file, cosmology, tracer_name_root, single_index=None, max_index=None):
        """ Initializes the CIBIntensityTracerWorkspace with the given parameters.

        Parameters:
        flux_fits_file: str, path to the flux fits file containing the data
        cosmology: pyccl.core.Cosmology object, the cosmology to be used in the analysis
        tracer_name_root: str, root name for the tracers
        single_index: int, index of the tracer to be used (optional) - if provided, only this tracer will be defined
        max_index: int, maximum index of the tracers to be defined (optional) - if provided, tracers from 0 to max_index will be defined
        """

        self.data = fits.open(flux_fits_file) #Can i use sacc here?

        self.tracer_name_root = tracer_name_root

        if single_index is not None:
            self.zs_CIB = self.data[1].data
            self.snus_CIB = {f'{tracer_name_root}{single_index}':self.data[0].data[single_index+3]/1e6}
        elif max_index is not None:
            self.zs_CIB = self.data[1].data
            self.snus_CIB = {f'{tracer_name_root}{index}':self.data[0].data[index+3]/1e6 for index in range(max_index+1)}

        self.cosmology = cosmology
        self.single_index = single_index

    def define_tracer_objects(self):
        """ Define the CIB intensity tracer objects based on the provided parameters. Puts them into a list of CIBIntensityTracers objects, which can be used to access the tracers directly.
        
        Returns:
        tracers_obj: list of CIBIntensityTracers objects, each corresponding to a tracer index from 0 to max_index or the single_index if provided.
        """

        self.tracers_obj = []

        if self.single_index is None:
            for index in range(len(self.snus_CIB)):
                self.tracers_obj.append(
                    CIBIntensityTracers(
                        self.tracer_name_root, index,
                        self.cosmology, 
                        self.snus_CIB[f'{self.tracer_name_root}{index}'], 
                        self.zs_CIB, 
                        z_min=0., z_max=6.
                    )
                )

        elif self.single_index is not None:
            self.tracers_obj.append(
                CIBIntensityTracers(
                    self.tracer_name_root, self.single_index,
                    self.cosmology, 
                    self.snus_CIB[f'{self.tracer_name_root}{self.single_index}'], 
                    self.zs_CIB, 
                    z_min=0., z_max=6.
                )
            )

        return self.tracers_obj
        
    def define_tracer_dict(self):
        """ Define a dictionary of tracers, where the keys are the tracer names and the values are the corresponding CIBIntensityTracers objects.
        
        Returns:
        tracers_dict: dict, keys are tracer names and values are CIBIntensityTracers objects
        """

        self.define_tracer_objects()

        self.tracers_dict = {
            t.name: t.tracer for t in self.tracers_obj
        }
        return self.tracers_dict
    
class SaccWorkspace:
    """ A class that handles the SACC file being used in the analysis."""
    def __init__(self, sacc_file=None, tracer_combinations=None, reverse_order=False):
        """ Initializes the SaccWorkspace with the given parameters.

        Parameters:
        sacc_file: str, path to the SACC file containing the data
        tracer_combinations: list of tuples, each tuple containing a pair of tracer names for which the C_ell will be extracted
        reverse_order: bool, whether to reverse the order of the tracers in the tracer combinations (default: False) - this is useful for cases where the SACC file has tracers in a different order than expected.
        """


        self.sacc_file = sacc_file

        self.reverse_order = reverse_order


        if type(sacc_file) is str:
            self.data = sc.Sacc.load_fits(sacc_file)

        self.tracer_combinations = tracer_combinations

        self.aliases = {}

    def get_tracer(self, tracer_name):
        """ Get a specific tracer from the SACC file.
        
        Parameters:
        tracer_name: str, name of the tracer

        Returns:
        tracer: sacc.Tracer, the tracer object corresponding to the tracer_name
        """
        return self.data.get_tracer(tracer_name)

    def get_ell_cl(self, tracer1, tracer2):
        """ Get the C_ell for a specific tracer combination.

        Parameters:
        tracer1: str, name of the first tracer
        tracer2: str, name of the second tracer

        Returns:
        ell_cl: numpy array, C_ell for the tracer combination
        """

        if self.aliases.get(f'{tracer1}') is not None:
            tracer1 = self.aliases[f'{tracer1}']
        if self.aliases.get(f'{tracer2}') is not None:
            tracer2 = self.aliases[f'{tracer2}']

        return self.data.get_ell_cl(None, tracer1, tracer2, return_cov=False)
    
    def define_alias(self, tracer_name, alias):
        """
        Define an alias for a tracer.
        
        Parameters:
        tracer_name: str, name of the tracer
        alias: str, alias for the tracer
        """
        self.aliases[alias] = tracer_name

        print(self.aliases)

    def select_from_sacc(self, data_type, tracer_combos=None):
        '''
        FROM DR TOM CORNISH, University of Oxford, 2025.

        Given a Sacc object and a set of tracer combinations, will return a new
        Sacc object containing only the information for those tracer combinations.

        Parameters
        ----------
        s: sacc.sacc.Sacc or str
            The Sacc object containing the information for many tracers. If a
            string, must be the path to a Sacc file.

        tracer_combos: list[tuple]
            List of tuples, with each tuple containing a pair of tracer names.

        data_type: str
            Data type for which the information is to be extracted. E.g. 'cl_00' or
            'galaxy_density_cl'. Use print(sacc.standard_types) to see list of
            possible values.

        Returns
        -------
        s_new: sacc.sacc.Sacc
            Sacc object containing only the desired information.
        '''
        import sacc

        s = self.sacc_file

        if tracer_combos is None:
            tracer_combos = self.tracer_combinations
        
        # Check if input is a string
        if type(s) is str:
            s = sacc.Sacc.load_fits(s)

        # Get the unique tracer names
        tc_unique = np.unique(tracer_combos)
        # Set up a new Sacc object and add tracers
        s_new = sacc.Sacc()
        for tc in tc_unique:

            if self.aliases.get(f'{tc}') is not None:
                tc = self.aliases[f'{tc}']

            s_new.add_tracer_object(s.get_tracer(tc))
        # Now add ell and C_ell info for each desired combination
        inds_all = []
        for tc in tracer_combos:

            if self.aliases.get(f'{tc[0]}') is not None:
                tc = (self.aliases[f'{tc[0]}'], tc[1])
            if self.aliases.get(tc[1]) is not None:
                tc = (tc[0], self.aliases[f'{tc[1]}'])
            
            ells, cells, inds = s.get_ell_cl(data_type, *tc, return_ind=True)
            # Get the window functions
            wins = s.get_bandpower_windows(inds)
            # Add the ell_cl info
            s_new.add_ell_cl(data_type, *tc, ells, cells, window=wins)
            # Add the indices to the list
            inds_all.extend(list(inds))
        # Add the covariance info
        if s.covariance is not None:
            s_new.covariance = sacc.covariance.FullCovariance(
                s.covariance.covmat[inds_all][:, inds_all]
                )
        else:
            s_new.covariance = None

        return s_new
    
    def get_covariance_matrix(self, data_type, tracer_combos=None):
        """ Get the covariance matrix for a specific data type from the SACC file.

        Parameters:
        data_type: str, data type for which the covariance matrix is to be extracted
        tracer_combos: list of tuples, each tuple containing a pair of tracer names for which the covariance matrix is to be extracted

        Returns:
        cov_matrix: numpy array, covariance matrix for the specified data type
        """

        cov_matrix = self.select_from_sacc(data_type, tracer_combos=tracer_combos).covariance.covmat

        return cov_matrix

    def cut_covariance_matrix(self, datatype, masks, tracer_combos=None):
        """ Cut the covariance matrix to a specific range based on a mask and width.

        Parameters:
        datatype: str, data type for which the covariance matrix is to be obtained and then cut
        mask: numpy array, boolean mask indicating which elements to keep
        tracer_combos: list of tuples, each tuple containing a pair of tracer names for which the covariance matrix is to be extracted (optional) - if not provided, will use the tracer combinations defined in the SaccWorkspace object.

        Returns:
        new_cov_matrix: numpy array, cut covariance matrix
        """
        masks = np.array(masks).flatten()
        nkeep = int(masks.sum())
        covmask = np.outer(masks, masks)

        if tracer_combos is None:
            tracer_combos = self.tracer_combinations

        cov_matrix = self.get_covariance_matrix(datatype, tracer_combos=tracer_combos)
        
        new_cov_matrix = cov_matrix[covmask].reshape((nkeep, nkeep))
        return new_cov_matrix

    def get_c_ells(self, sacc_file=None, tracer_combinations=None):
        """
        Get the C_ell for a specific tracer combination.
        
        Parameters:
        sacc_file: str, path to the SACC file containing the data (optional) - if not provided, will use the SACC file defined in the SaccWorkspace object.
        tracer_combinations: list of tuples, each tuple containing a pair of tracer names for which the C_ell will be extracted (optional) - if not provided, will use the tracer combinations defined in the SaccWorkspace object.
        
        Returns:
        array:
            ells: list of numpy arrays, each containing the ell values for a tracer combination
            measured_c_ells: list of numpy arrays, each containing the measured C_ells for a tracer combination
        """

        # Check if self.data is set, else use sacc_file function parameter
        if self.data is None and sacc_file is not None:
            self.data = sc.Sacc.load_fits(sacc_file)
        if tracer_combinations is None:
            tracer_combinations = self.tracer_combinations
        
        
        ells = []
        measured_c_ells = []

        for tracer in tracer_combinations:
            if type(tracer) is not tuple:
                raise ValueError("Tracer combinations must be tuples of tracer names.")
            
          
            if self.aliases.get(tracer[0]) is not None:
                tracer1 = self.aliases[tracer[0]]
            else:
                tracer1 = tracer[0]

            if self.aliases.get(tracer[1]) is not None:
                tracer2 = self.aliases[tracer[1]]
            else:
                tracer2 = tracer[1]

            ell, cl = self.data.get_ell_cl(None, tracer1, tracer2, return_cov=False) #add check for order flip here
           
            ells.append(ell)
            measured_c_ells.append(cl)

        return [ells, measured_c_ells]
    
    def get_errors(self, tracer_combos=None):
        """ Get the errors for the C_ell for a specific tracer combination.

        Parameters:
        tracer_combos: list of tuples, each tuple containing a pair of tracer names

        Returns:
        errors: list of numpy arrays, each containing the errors for a tracer combination
        """
        if self.tracer_combinations is None:
            self.tracer_combinations = tracer_combos
        
        if self.data is None:
            raise ValueError("SACC data not loaded. Please load the SACC file first.")

        errors = []

            

        for tracer in self.tracer_combinations:

            if self.aliases.get(tracer[0]) is not None:
                tracer1 = self.aliases[tracer[0]]
            else:
                tracer1 = tracer[0]

            if self.aliases.get(tracer[1]) is not None:
                tracer2 = self.aliases[tracer[1]]
            else:
                tracer2 = tracer[1]

    


            if type(tracer) is not tuple:
                raise ValueError("Tracer combinations must be tuples of tracer names.")
            ell, cl, cov_matrix = self.data.get_ell_cl(None, tracer1, tracer2, return_cov=True)
        
            errors.append(np.sqrt(np.diag(cov_matrix)))

        return errors
    
    def get_beam_window(self, tracer_combo):
        if self.reverse_order: #expects CIB to be the second tracer in the combination
            tracer = tracer_combo[0]
            
        else:
            tracer = tracer_combo[1]

        if self.aliases.get(tracer) is not None: #checks if an alias has been defined for the tracer
            tracer = self.aliases[tracer]

        return self.data.tracers[tracer].ell, self.data.tracers[tracer].beam
    
    def get_cut_beam_window(self, tracer_combo, cut_ells):

        ells, beam = self.get_beam_window(tracer_combo) 

        mask = np.isin(ells, np.round(cut_ells))

        beam_cut = beam[mask]

        return beam_cut
    

class MaleubreModel():
    """ Likelihood model for angular power spectra, as described in the paper by Maleubre et al. (TBC).
    
    Expects the galaxy density tracers to be defined as the leadind tracer in each combination in the sacc file, and the U tracer to be defined as the second tracer in each combination. If this is not the case, one can 
    flag the reverse_order parameter to True when initializing the SaccWorkspace object, which will reverse the order of the tracers in the tracer combinations. Then pass tracer combinations in in the order of the SACC file, i.e. (U, G) instead of (G, U)."""

    def __init__(self, tracer_combos, cosmology, Tracer1Workspace, Tracer2Workspace=None, sacc_workspace=None, logged_N=False, min_ell=100, max_ell=1000, k_max=None, beam_window=None):
        """ Initializes the MaleubreModel with the given parameters.
        
        Parameters:
        tracer_combos: list of tuples, each tuple containing a pair of tracer names for which the C_ell will be extracted
        cosmology: pyccl.core.Cosmology object, the cosmology to be used in the analysis
        Tracer1Workspace: GalaxyDensityTracerWorkspace object, workspace containing the first tracer information
        Tracer2Workspace: GalaxyDensityTracerWorkspace object, workspace containing the second tracer information (optional) - if not provided, only the first tracer will be used and the cross-correlations will not be computed
        sacc_workspace: SaccWorkspace object, workspace containing the SACC file information (will eventuall be optional) 
        logged_N: bool, defines if the N_gg and N_gnu passed in to the log_likelihood_function are logged or not (default: False) - if True, the parameters will be transformed from log space before being used in the likelihood function
        min_ell: int, minimum ell value for the C_ell (default: 100)
        max_ell: int, maximum ell value for the C_ell (default: 1000)
        k_max: float, maximum k value for the power spectrum (optional) - if defined, this will override any value for max_ell
        """
        
        self.Tracer1Workspace = Tracer1Workspace
        self.Tracer2Workspace = Tracer2Workspace
        self.tracer_combos = tracer_combos
        
        self.cosmology = cosmology

        self.logged_N = logged_N

        self.sacc_workspace = sacc_workspace if sacc_workspace is not None else None

        self.data = sacc_workspace.data if sacc_workspace is not None else None

        self.beam_window = beam_window

        self.pre_calculated = False

        self.min_ell = min_ell
        self.max_ell = max_ell

        self.icovariance = None

        self.P_mms = {}
        self.P_mm_ksquares = {}

        k_arr= np.geomspace(1E-4, 100, 256)
        a_arr = 1. / (1. + np.linspace(0, 6, 16)[::-1])
        pk_mm = ccl.linear_matter_power(self.cosmology, k_arr, a_arr)
        
        self.pk2d_mm = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=pk_mm, is_logp=False)

        self.pksquare_mm = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=k_arr**2 * pk_mm, is_logp=False)
        
        
        # Initialize the workspace for MCMC
        #self.mcmc_workspace = MCMCWorkspace(sacc_file)


        Tracer1Workspace.define_tracer_dict()
        Tracer2Workspace.define_tracer_dict() if Tracer2Workspace is not None else None

        if Tracer2Workspace is not None:
            self.workspace_dict = {}

            root_name_1 = Tracer1Workspace.tracer_name_root
            root_name_2 = Tracer2Workspace.tracer_name_root

            self.workspace_dict[root_name_1] = Tracer1Workspace
            self.workspace_dict[root_name_2] = Tracer2Workspace
        
        self.k_max = k_max

        tracers1 = self.Tracer1Workspace.define_tracer_dict() 

        # Checks if the second tracer workspace is defined, and if not, uses the first tracer workspace - for cases where only auto-correlations are being computed.
        try:
            tracers2 = self.Tracer2Workspace.define_tracer_dict()
        except AttributeError:
            tracers2 = tracers1

        # Puts the tracers into a single dictionary, so that they can be accessed by their names.
        self.tracers = {**tracers1, **tracers2}


    def kernel_squared_integral(self, tracer, tracerwsp):

        """ Calculate the integral of the square of the kernel for a given tracer. Includes a factor of 1/(chi^2), as seen in the literature.

        Parameters:
        tracer: str, name of the tracer
        tracerwsp: GalaxyDensityTracerWorkspace, workspace containing the tracer information

        Returns:
        integral: float, the integral of the square of the kernel for the tracer
        """
        if self.sacc_workspace is not None:
            if tracer in self.sacc_workspace.aliases:
                tracer = self.sacc_workspace.aliases[tracer]

        z = self.data.get_tracer(tracer).z
        
        #print(z)

        a_values = np.linspace(1/(1+z.max()), 1, 100)

        #ensure all non-zero chi values
        chi_values = ccl.comoving_radial_distance(self.cosmology, a_values)[::-1]
        chi_values = chi_values[chi_values > 0]
        #

        kernel = tracerwsp.tracers_dict[tracer].get_kernel(chi_values)[0]
        kernel_square = kernel ** 2

        chi_factor = 1/(chi_values**2) 

        #print(kernel)

        integral = np.trapezoid(chi_factor*kernel_square, chi_values)

        #is_sorted = np.all(np.diff(chi_values) >= 0)
        #print(f"sorted? {is_sorted}")


        #plot kernel
        #plt.plot(chi_values, kernel_square)

        return integral
    
    def kernel_mixed_integral(self, tracer1, tracer2, tracerwsp1, tracerwsp2):
        """ Calculate the integral of the product of the kernels for two tracers. Includes a factor of 1/(chi^2), as seen in the literature.

        Parameters:
        tracer1: str, name of the first tracer
        tracer2: str, name of the second tracer
        tracerwsp1: GalaxyDensityTracerWorkspace, workspace containing the first tracer information
        tracerwsp2: GalaxyDensityTracerWorkspace, workspace containing the second tracer information

        Returns:
        integral: float, the integral of the product of the kernels for the two tracers
        """
        if self.sacc_workspace is not None:
        
            if self.sacc_workspace.reverse_order:
                tracer1, tracer2 = tracer2, tracer1
                tracerwsp1, tracerwsp2 = tracerwsp2, tracerwsp1
        
        z = self.data.get_tracer(tracer1).z
        a_values = np.linspace(1/(1+z.max()), 1, 100)

        chi_values = ccl.comoving_radial_distance(self.cosmology, a_values)[::-1]
        chi_values = chi_values[chi_values > 0]

        kernel1 = tracerwsp1.tracers_dict[tracer1].get_kernel(chi_values)[0]
        kernel2 = tracerwsp2.tracers_dict[tracer2].get_kernel(chi_values)[0]

        kernel_product = kernel1 * kernel2

        chi_factor = 1/(chi_values**2)

        integral = np.trapezoid(chi_factor*kernel_product, chi_values)

        return integral
    
    def complete_precalculation(self):
        """ Pre-calculate the linear power spectrum for each tracer combination, and the covariance matrix for the SACC file. This is done to speed up the likelihood calculation during MCMC sampling.
        """

        self.all_cut_c_ells = []
        self.masks = []
        self.cut_ells = {}

        self.pre_calculated = True



        # Pre-calculates a dictionary of the linear power spectrum used in the model, for each tracer combination.
        for i in range(len(self.tracer_combos)):
            
            if self.k_max is not None:
                self.max_ell = self.get_ell_max(self.tracer_combos[i])

            if self.tracer_combos[i][0] == self.tracer_combos[i][1]:

                self.workspace = self.workspace_dict[self.tracer_combos[i][0][:-1]]

                mod_val = len(self.workspace.tracers_obj)

                cut_ells, cut_c_ells, mask = self.workspace.tracers_obj[i%mod_val].get_cut_data(self.sacc_workspace, ell_min=self.min_ell, ell_max=self.max_ell)

                self.all_cut_c_ells.append(cut_c_ells)
                self.masks.append(mask)

                self.cut_ells[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}'] = cut_ells

                if self.P_mms.get(f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}') is None:
                    self.P_mms[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}'] = ccl.angular_cl(
                        self.cosmology,
                        self.tracers[self.tracer_combos[i][0]], 
                        self.tracers[self.tracer_combos[i][1]],
                        ell=cut_ells,
                        p_of_k_a=self.pk2d_mm
                    )
                
                if self.P_mm_ksquares.get(f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}') is None:
                    self.P_mm_ksquares[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}'] = ccl.angular_cl(
                        self.cosmology, 
                        self.tracers[self.tracer_combos[i][0]], 
                        self.tracers[self.tracer_combos[i][1]], 
                        ell=cut_ells, 
                        p_of_k_a=self.pksquare_mm)
                
            else:
                self.workspace1 = self.workspace_dict[self.tracer_combos[i][0][:-1]] if self.tracer_combos[i][0] in self.workspace_dict else self.Tracer1Workspace
                self.workspace2 = self.workspace_dict[self.tracer_combos[i][1][:-1]] if self.tracer_combos[i][1] in self.workspace_dict else self.Tracer2Workspace
                
                mod_val = len(self.workspace1.tracers_obj)

                cut_ells, cut_c_ells, mask = self.workspace1.tracers_obj[i%mod_val].get_cut_data(self.sacc_workspace, tracer_2=self.tracer_combos[i][1], ell_min=self.min_ell, ell_max=self.max_ell) # get the cut data for the cross-correlation

                self.cut_ells[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}'] = cut_ells

                self.all_cut_c_ells.append(cut_c_ells)
                self.masks.append(mask)

                if self.P_mms.get(f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}') is None:
                    self.P_mms[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}'] = ccl.angular_cl(
                        self.cosmology,
                        self.tracers[self.tracer_combos[i][0]], 
                        self.tracers[self.tracer_combos[i][1]],
                        ell=cut_ells,
                        p_of_k_a=self.pk2d_mm
                    )
                
                if self.P_mm_ksquares.get(f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}') is None:
                    self.P_mm_ksquares[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}'] = ccl.angular_cl(
                        self.cosmology, 
                        self.tracers[self.tracer_combos[i][0]], 
                        self.tracers[self.tracer_combos[i][1]], 
                        ell=cut_ells, 
                        p_of_k_a=self.pksquare_mm)
            
        covariance = self.sacc_workspace.cut_covariance_matrix('cl_00', self.masks)

        self.icovariance = np.linalg.inv(covariance)

        
    def return_tracers(self):
        tracers1 = self.Tracer1Workspace.define_tracer_dict() 

        try:
            tracers2 = self.Tracer2Workspace.define_tracer_dict()
        except AttributeError:
            tracers2 = tracers1

        tracers = {**tracers1, **tracers2}

        return tracers

    def log_likelihood_function(self, b_gs, N_ggs, A_ggs, N_gnus=None, A_gnus=None, bpsfrs=None):
        """ Calculate the log-likelihood function for the given parameters. Can be used for only auto-correlations, cross-correlations, or both auto and cross-correlations.

        Please note that this has not been optimised and is in nearly all cases obsolete. Only use this if you are paying particular care to the kernel integrals.
        Otherwise, refer to lightweight_log_likelihood_function.

        Parameters:
        b_gs: list of floats, bias parameters for the galaxy tracers
        N_ggs: list of floats, noise parameters for the tracer auto-correlations
        A_ggs: list of floats, amplitude parameters for the tracer auto-correlations
        N_gnus: list of floats, noise parameters for the tracer cross-correlations (optional)
        A_gnus: list of floats, amplitude parameters for the tracer cross-correlations (optional)
        bpsfrs: list of floats, parameters for the bias weighted star formation rate desnsity (optional)

        Returns:
        logL: float, the log-likelihood value for the given parameters
        """

        # Checks if the noise parameters need to be transformed from log space
        if self.logged_N:
            N_ggs = np.power(10, N_ggs)
            if N_gnus is not None:
                N_gnus = np.power(10, N_gnus)

        tracers1 = self.Tracer1Workspace.define_tracer_dict() 

        # Checks if the second tracer workspace is defined, and if not, uses the first tracer workspace - for cases where only auto-correlations are being computed.
        try:
            tracers2 = self.Tracer2Workspace.define_tracer_dict()
        except AttributeError:
            tracers2 = tracers1

        # Puts the tracers into a single dictionary, so that they can be accessed by their names.
        tracers = {**tracers1, **tracers2}

        theory_c_ells = []
        all_cut_c_ells = []
        masks = []

        # Iterates through the tracer combinations and calculates the C_ell for each combination.
        for i in range(len(self.tracer_combos)):
            
            # Checks if a maximum k value is defined, and if so, sets the maximum ell value accordingly.
            if self.k_max is not None:
                self.max_ell = self.get_ell_max(self.tracer_combos[i])
            
            # AUTO-CORRELATIONS - expects only galaxy density - galaxy density tracers in the tracer combinations.
            if self.tracer_combos[i][0] == self.tracer_combos[i][1]:

                self.workspace = self.workspace_dict[self.tracer_combos[i][0][:-1]]

                mod_val = len(self.workspace.tracers_obj)
                
                cut_ells, cut_c_ells, mask = self.workspace.tracers_obj[i%mod_val].get_cut_data(self.sacc_workspace, ell_min=self.min_ell, ell_max=self.max_ell) # get the cut data for the auto-correlation

                theory_c_ells.append(
                b_gs[i%len(b_gs)]**2 * ccl.angular_cl(
                    self.cosmology,
                    tracers[self.tracer_combos[i][0]], 
                    tracers[self.tracer_combos[i][1]],
                    ell=cut_ells,
                    p_of_k_a=self.pk2d_mm) + N_ggs[i%len(N_ggs)]*self.kernel_squared_integral(self.tracer_combos[i][0], self.workspace) + ccl.angular_cl(self.cosmology, tracers[self.tracer_combos[i][0]], tracers[self.tracer_combos[i][0]], ell=cut_ells, p_of_k_a=self.pksquare_mm) * A_ggs[i%len(A_ggs)]
                )

                all_cut_c_ells.append(cut_c_ells)
                masks.append(mask)

            # CROSS-CORRELATIONS - expects galaxy density - CIB intensity tracers in the tracer combinations. Order can be reversed but one must change the SACC file to have the reverse_order flag set to True.
            elif self.tracer_combos[i][0] != self.tracer_combos[i][1]:

                self.workspace1 = self.workspace_dict[self.tracer_combos[i][0][:-1]] if self.tracer_combos[i][0] in self.workspace_dict else self.Tracer1Workspace
                self.workspace2 = self.workspace_dict[self.tracer_combos[i][1][:-1]] if self.tracer_combos[i][1] in self.workspace_dict else self.Tracer2Workspace

                mod_val = len(self.workspace1.tracers_obj)

                cut_ells, cut_c_ells, mask = self.workspace1.tracers_obj[i%mod_val].get_cut_data(self.sacc_workspace, tracer_2=self.tracer_combos[i][1], ell_min=self.min_ell, ell_max=self.max_ell) # get the cut data for the cross-correlation

                theory_c_ells.append(
                b_gs[i%len(b_gs)] * bpsfrs[i%len(bpsfrs)] * ccl.angular_cl( # $b_{g} b_{sfr} C_ell$
                    self.cosmology,
                    tracers[self.tracer_combos[i][0]],
                    tracers[self.tracer_combos[i][1]],
                    ell=cut_ells,
                    p_of_k_a=self.pk2d_mm) 
                    + N_gnus[i%len(N_gnus)]*self.kernel_mixed_integral(f'{self.tracer_combos[i][0]}', f'{self.tracer_combos[i][1]}', self.workspace1, self.workspace2) 
                    + ccl.angular_cl(self.cosmology, tracers[self.tracer_combos[i][0]], tracers[self.tracer_combos[i][1]], ell=cut_ells, p_of_k_a=self.pksquare_mm) * A_gnus[i%len(A_gnus)]
                )

                all_cut_c_ells.append(cut_c_ells)
                masks.append(mask)
           
        # Calculate the log-likelihood value using the standard formula from the literature.

        # Takes the masks for each tracer combination and uses them to mask the full cov matrix. 
        covariance = self.sacc_workspace.cut_covariance_matrix('cl_00', masks)

        icov = np.linalg.inv(covariance)

        diff = np.concatenate(all_cut_c_ells) - np.concatenate(theory_c_ells)
        
        logL = -0.5 *np.dot(diff, np.dot(icov, diff))

        
        return logL
    
    def lightweight_log_likelihood_function(self, b_gs, N_ggs, A_ggs, N_gnus=None, A_gnus=None, bpsfrs=None):
        """ Calculate the log-likelihood function for the given parameters. Can be used for only auto-correlations, cross-correlations, or both auto and cross-correlations.

        Parameters:
        b_gs: list of floats, bias parameters for the galaxy tracers
        N_ggs: list of floats, noise parameters for the tracer auto-correlations
        A_ggs: list of floats, amplitude parameters for the tracer auto-correlations
        N_gnus: list of floats, noise parameters for the tracer cross-correlations (optional)
        A_gnus: list of floats, amplitude parameters for the tracer cross-correlations (optional)
        bpsfrs: list of floats, parameters for the bias weighted star formation rate desnsity (optional)

        Returns:
        logL: float, the log-likelihood value for the given parameters
        """

        # Checks if the noise parameters need to be transformed from log space
        if self.logged_N:
            N_ggs = np.power(10, N_ggs)
            if N_gnus is not None:
                N_gnus = np.power(10, N_gnus)

        theory_c_ells = []
        all_cut_c_ells = []
        masks = []

        # Iterates through the tracer combinations and calculates the C_ell for each combination.
        for i in range(len(self.tracer_combos)):
            
            # Checks if a maximum k value is defined, and if so, sets the maximum ell value accordingly.
            if self.k_max is not None:
                self.max_ell = self.get_ell_max(self.tracer_combos[i])
            
            # AUTO-CORRELATIONS - expects only galaxy density - galaxy density tracers in the tracer combinations.
            if self.tracer_combos[i][0] == self.tracer_combos[i][1]:

                if self.pre_calculated == False:

                    self.workspace = self.workspace_dict[self.tracer_combos[i][0][:-1]]

                    mod_val = len(self.workspace.tracers_obj)
                
                    cut_ells, cut_c_ells, mask = self.workspace.tracers_obj[i%mod_val].get_cut_data(self.sacc_workspace, ell_min=self.min_ell, ell_max=self.max_ell) # get the cut data for the auto-correlation

                    if self.P_mms.get(f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}') is None:
                        self.P_mms[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}'] = ccl.angular_cl(
                            self.cosmology,
                            self.tracers[self.tracer_combos[i][0]], 
                            self.tracers[self.tracer_combos[i][1]],
                            ell=cut_ells,
                            p_of_k_a=self.pk2d_mm
                        )
                    
                    if self.P_mm_ksquares.get(f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}') is None:
                        self.P_mm_ksquares[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}'] = ccl.angular_cl(
                            self.cosmology, 
                            self.tracers[self.tracer_combos[i][0]], 
                            self.tracers[self.tracer_combos[i][1]], 
                            ell=cut_ells, 
                            p_of_k_a=self.pksquare_mm)
                        
                    all_cut_c_ells.append(cut_c_ells)
                    masks.append(mask)
                
                Pm = self.P_mms[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}']
                Pmk2 = self.P_mm_ksquares[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}']

                theory_c_ells.append(
                    b_gs[i%len(b_gs)]**2 * Pm + N_ggs[i%len(N_ggs)] + Pmk2 * A_ggs[i%len(A_ggs)]
                ) 
                

            # CROSS-CORRELATIONS - expects galaxy density - CIB intensity tracers in the tracer combinations. Order can be reversed but one must change the SACC file to have the reverse_order flag set to True.
            elif self.tracer_combos[i][0] != self.tracer_combos[i][1]:

                if self.pre_calculated == False:

                    self.workspace1 = self.workspace_dict[self.tracer_combos[i][0][:-1]] if self.tracer_combos[i][0] in self.workspace_dict else self.Tracer1Workspace
                    self.workspace2 = self.workspace_dict[self.tracer_combos[i][1][:-1]] if self.tracer_combos[i][1] in self.workspace_dict else self.Tracer2Workspace
                    
                    mod_val = len(self.workspace1.tracers_obj)

                    cut_ells, cut_c_ells, mask = self.workspace1.tracers_obj[i%mod_val].get_cut_data(self.sacc_workspace, tracer_2=self.tracer_combos[i][1], ell_min=self.min_ell, ell_max=self.max_ell) # get the cut data for the cross-correlation

                    if self.P_mms.get(f'{self.tracer_combos[i][0], self.tracer_combos[i][1]}') is None:
                        self.P_mms[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}'] = ccl.angular_cl(
                            self.cosmology,
                            self.tracers[self.tracer_combos[i][0]], 
                            self.tracers[self.tracer_combos[i][1]],
                            ell=cut_ells,
                            p_of_k_a=self.pk2d_mm
                        )
                    
                    if self.P_mm_ksquares.get(f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}') is None:
                        self.P_mm_ksquares[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}'] = ccl.angular_cl(
                            self.cosmology, 
                            self.tracers[self.tracer_combos[i][0]], 
                            self.tracers[self.tracer_combos[i][1]], 
                            ell=cut_ells, 
                            p_of_k_a=self.pksquare_mm)
                        
                    all_cut_c_ells.append(cut_c_ells)
                    masks.append(mask)
                else:
                    cut_ells = self.cut_ells[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}']
                    
                Pm = self.P_mms[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}']
                Pmk2 = self.P_mm_ksquares[f'{self.tracer_combos[i][0]}, {self.tracer_combos[i][1]}']

                if self.beam_window:
                    beam = self.sacc_workspace.get_cut_beam_window(self.tracer_combos[i], cut_ells)
                else:
                    beam = 1.0

                theory_c_ells.append(
                (b_gs[i%len(b_gs)] * bpsfrs[i%len(bpsfrs)] * Pm
                    + N_gnus[i%len(N_gnus)]
                    + Pmk2 * A_gnus[i%len(A_gnus)]) * beam
                ) 

                
           
        # Calculate the log-likelihood value using the standard formula from the literature.

        # Takes the masks for each tracer combination and uses them to mask the full cov matrix.
        # If covariance matrix has already been pre-calculated, we default to the stored one. -- assumes same tracer combo order throughout.
        if self.pre_calculated:
            all_cut_c_ells = self.all_cut_c_ells
            masks = self.masks

        if self.icovariance is not None:
            pass
        else:
            covariance = self.sacc_workspace.cut_covariance_matrix('cl_00', masks)

            self.icovariance = np.linalg.inv(covariance)

        diff = np.concatenate(all_cut_c_ells) - np.concatenate(theory_c_ells)
        
        logL = -0.5 *np.dot(diff, np.dot(self.icovariance, diff))

        
        return logL
    

    def get_modelled_data(self, b_gs, N_ggs, A_ggs, N_gnus=None, A_gnus=None, bpsfrs=None, full_ells=False):

        """ Get the modelled data for the given parameters. Can be used for only auto-correlations, cross-correlations, or both auto and cross-correlations.
        
        Parameters:
        b_gs: list of floats, bias parameters for the galaxy tracers
        N_ggs: list of floats, noise parameters for the tracer auto-correlations
        A_ggs: list of floats, amplitude parameters for the tracer auto-correlations
        N_gnus: list of floats, noise parameters for the tracer cross-correlations (optional)
        A_gnus: list of floats, amplitude parameters for the tracer cross-correlations (optional)
        bpsfrs: list of floats, parameters for the bias weighted star formation rate desnsity (optional)
        full_ells: bool, whether to return the full ell values or only the cut ell values (default: False) - if True, will return the full ell values - for extrapolation when plotting, for example.
        
        Returns:
        array:
            cut_ells_arr: list of numpy arrays, each containing the cut ell values for a tracer combination
            theory_c_ells: list of numpy arrays, each containing the modelled C_ells for a tracer combination
            masks: list of numpy arrays, each containing the mask for a tracer combination
        """

        if self.logged_N:
            N_ggs = np.power(10, N_ggs)
            if N_gnus is not None:
                N_gnus = np.power(10, N_gnus)

        tracers1 = self.Tracer1Workspace.define_tracer_dict() 

        try:
            tracers2 = self.Tracer2Workspace.define_tracer_dict()
        except AttributeError:
            tracers2 = tracers1

        tracers = {**tracers1, **tracers2}

        theory_c_ells = []

        cut_ells_arr = []

        masks = []
        #all_cut_c_ells = [] - not needed for now

        for i in range(len(self.tracer_combos)):

            if self.k_max is not None:
                self.max_ell = self.get_ell_max(self.tracer_combos[i])

            tracer_combo = self.tracer_combos[i]

            if full_ells:

                temp_tracer_combo = tracer_combo
                if self.sacc_workspace.aliases.get(temp_tracer_combo[0]) is not None:
                    temp_tracer_combo = (self.sacc_workspace.aliases[temp_tracer_combo[0]], temp_tracer_combo[1])
                if self.sacc_workspace.aliases.get(tracer_combo[1]) is not None:
                    temp_tracer_combo = (temp_tracer_combo[0], self.sacc_workspace.aliases[temp_tracer_combo[1]])

                ells = self.sacc_workspace.get_ell_cl(temp_tracer_combo[0], temp_tracer_combo[1])[0]
        

            if tracer_combo[0] == tracer_combo[1]:

                self.workspace = self.workspace_dict[tracer_combo[0][:-1]]

                mod_val = len(self.workspace.tracers_obj)
                
                cut_ells, cut_c_ells, mask = self.workspace.tracers_obj[i%mod_val].get_cut_data(self.sacc_workspace, ell_min=self.min_ell, ell_max=self.max_ell) # get the cut data for the auto-correlation

                if full_ells == False:
                    ells = cut_ells

                theory_c_ells.append(
                b_gs[i%len(b_gs)]**2 * ccl.angular_cl(
                    self.cosmology,
                    tracers[tracer_combo[0]],
                    tracers[tracer_combo[1]],
                    ell=ells,
                    p_of_k_a=self.pk2d_mm) + N_ggs[i%len(N_ggs)]*self.kernel_squared_integral(tracer_combo[0], self.workspace) + ccl.angular_cl(self.cosmology, tracers[tracer_combo[0]], tracers[tracer_combo[0]], ell=ells, p_of_k_a=self.pksquare_mm) * A_ggs[i%len(A_ggs)]
                )

                cut_ells_arr.append(cut_ells)
                masks.append(mask)
                #all_cut_c_ells.append(cut_c_ells) - not needed for now


            elif tracer_combo[0] != tracer_combo[1]: # notes cross-correlations

                self.workspace1 = self.workspace_dict[tracer_combo[0][:-1]] if tracer_combo[0] in self.workspace_dict else self.Tracer1Workspace
                self.workspace2 = self.workspace_dict[tracer_combo[1][:-1]] if tracer_combo[1] in self.workspace_dict else self.Tracer2Workspace

                mod_val = len(self.workspace1.tracers_obj)

                cut_ells, cut_c_ells, mask = self.workspace1.tracers_obj[i%mod_val].get_cut_data(self.sacc_workspace, tracer_2=tracer_combo[1], ell_min=self.min_ell, ell_max=self.max_ell) # get the cut data for the cross-correlation

            

                if full_ells == False:
                    ells = cut_ells

                if self.beam_window:
                    beam = self.sacc_workspace.get_cut_beam_window(tracer_combo, ells)
                else:
                    beam = 1.0

                theory_c_ells.append(
                (b_gs[i%len(b_gs)] * bpsfrs[i%len(bpsfrs)] * ccl.angular_cl( # $b_{g} b_{sfr} C_ell$
                    self.cosmology,
                    tracers[tracer_combo[0]],
                    tracers[tracer_combo[1]],
                    ell=ells,
                    p_of_k_a=self.pk2d_mm) 
                    + N_gnus[i%len(N_gnus)]*self.kernel_mixed_integral(f'{tracer_combo[0]}', f'{tracer_combo[1]}', self.workspace1, self.workspace2) 
                    + ccl.angular_cl(self.cosmology, tracers[tracer_combo[0]], tracers[tracer_combo[1]], ell=ells, p_of_k_a=self.pksquare_mm) * A_gnus[i%len(A_gnus)]) * beam
                ) 

                cut_ells_arr.append(cut_ells)
                masks.append(mask)
                #all_cut_c_ells.append(cut_c_ells) - not needed for now

        return [cut_ells_arr, theory_c_ells, masks]

    def get_ell_max(self, tracer_combo):

        if self.sacc_workspace.reverse_order:
            tracer = tracer_combo[1]
        else:
            tracer = tracer_combo[0]
        
        z = self.data.get_tracer(tracer).z
        nz = self.data.get_tracer(tracer).nz

        weighted_z = np.average(z, weights=nz)
        
        a_values = np.linspace(1/(1+weighted_z), 1, 100)

        chi_values = ccl.comoving_radial_distance(self.cosmology, a_values)[::-1]
        
        ell_max = self.k_max * max(chi_values) - 0.5

        return np.int(ell_max)
        


def version():
    """ Return the version of the module."""
    return "0.0.1 - Stable Release"







    
    
        

