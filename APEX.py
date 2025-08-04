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

class MCMCWorkspace:
    def __init__(self, sacc_file=None, model=None, likelihood_function=None, full_info=None, params=None):
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
        self.params_with_priors = {}
        for param in params:
            self.params_with_priors[param] = priors[params.index(param)]

    def set_param_references(self, params, references):
        self.params_with_references = {}
        for param in params:
            self.params_with_references[param] = references[params.index(param)]

    def set_param_proposals(self, params, proposals):
        self.params_with_proposals = {}
        for param in params:
            self.params_with_proposals[param] = proposals[params.index(param)]

    def set_grouped_params(self, grouped_dict):
        """
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
        updated_info, self.sampler = cb.run(self.info)
        return updated_info, self.sampler
    
    def mpi_run(self):
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
        upd_info, sampler = cb.run(self.info, minimize=True)

        return sampler

    def save_chain(self, name_root='', mpi=False):
        """
        Save the chain to a file.
        
        Parameters:
        name_root: str, root name for the file
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
        import getdist.plots as gdplt
        gdsamples = self.sampler.samples(combined=True, to_getdist=True, skip_samples=0.3)

        gdplot = gdplt.get_subplot_plotter(width_inch=5)
        gdplot.triangle_plot(gdsamples,params_to_plot, filled=True)
        
        #save the plot
        gdplot.export(f'corner_plot{params_to_plot}.png')

    def getdist_load(self, name_root=''):
        import getdist
        import getdist.plots as gdplt

        chain_data = getdist.MCSamples(name_root)
        
        gdplot = gdplt.get_subplot_plotter(width_inch=5)
        gdplot.triangle_plot(chain_data, filled=True)

        #save the plot
        gdplot.export(f'{name_root}_corner_plot.png')

class GalaxyDensityTracers:
    def __init__(self, name, index, z, nz, sacc_file):
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
        ell, cl = s.get_ell_cl(None, self.name, self.name, return_cov=False)
        return ell
    
    def get_cut_data(self, sacc_workspace, ell_min=100, ell_max=1000, tracer_2=None):
        """
        Get the C_ell for this tracer, cut to a specific range.
        
        Parameters:
        sacc_workspace
        ell_min: int, minimum ell value
        ell_max: int, maximum ell value
        
        Returns:
        c_ell: numpy array, C_ell for this tracer in the specified range
        """
        s = sacc_workspace.data

        if tracer_2 is None:
            ell, cl = s.get_ell_cl(None, self.name, self.name, return_cov=False)
        else:

            try:
                tracer_2 = sacc_workspace.aliases[tracer_2]
                ell, cl = s.get_ell_cl(None, self.name, tracer_2, return_cov=False)

            except KeyError:
                ell, cl = s.get_ell_cl(None, self.name, tracer_2, return_cov=False)

            ell, cl = s.get_ell_cl(None, self.name, tracer_2, return_cov=False)
        
        mask = (ell >= ell_min) & (ell <= ell_max)
        return ell[mask], cl[mask], mask

    def get_z(self):
        return self.z

    def get_nz(self):
        return self.nz

class GalaxyDensityTracerWorkspace:
    def __init__(self, sacc_file, tracer_name_root, max_index, cosmology):
        self.sacc_file = sacc_file
        self.tracer_name_root = tracer_name_root
        self.max_index = max_index
        self.cosmology = cosmology

        
        self.data = sc.Sacc.load_fits(sacc_file)

    def get_kernel(self, i):
        z = self.data.get_tracer(f'{self.tracer_name_root}{i}').z
        nz = self.data.get_tracer(f'{self.tracer_name_root}{i}').nz

        return z, nz

    def generate_tracer_objects(self):
        self.tracers_obj = []
        for i in range(self.max_index+1):
            z, nz = self.get_kernel(i)
            self.tracers_obj.append(GalaxyDensityTracers(self.tracer_name_root, i, z, nz, self.sacc_file))
        
        return self.tracers_obj

    def define_tracer_dict(self):

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

    
    def cut_c_ells(self, tracer):
        tracer.get_cut_data(self.data)
    
class CIBIntensityTracers:

    def __init__(self, tracer_name_root, index, cosmo, snu_z, z_arr, z_min=0., z_max=6.):
        
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

    def get_cut_data(self, s, ell_min=100, ell_max=1000, tracer_2=None):
        """
        Get the C_ell for this tracer, cut to a specific range.
        
        Parameters:
        s: sacc.Sacc object
        ell_min: int, minimum ell value
        ell_max: int, maximum ell value
        
        Returns:
        c_ell: numpy array, C_ell for this tracer in the specified range
        """
        if tracer_2 is None:
            ell, cl = s.get_ell_cl(None, self.name, self.name, return_cov=False)
        else:

            try:
                tracer_2 = s.aliases[tracer_2]
                ell, cl = s.get_ell_cl(None, self.name, tracer_2, return_cov=False)

            except KeyError:
                ell, cl = s.get_ell_cl(None, self.name, tracer_2, return_cov=False)
        
        mask = (ell >= ell_min) & (ell <= ell_max)
        return ell[mask], cl[mask], mask

class CIBIntensityTracerWorkspace:

    def __init__(self, flux_fits_file, cosmology, name_root, sacc_file=None, single_index=None, max_index=None):
        self.data = fits.open(flux_fits_file) #Can i use sacc here?

        self.name_root = name_root

        if single_index is not None:
            self.zs_CIB = self.data[1].data
            self.snus_CIB = {f'{name_root}{single_index}':self.data[0].data[single_index+3]/1e6}
        elif max_index is not None:
            self.zs_CIB = self.data[1].data
            self.snus_CIB = {f'{name_root}{index}':self.data[0].data[index+3]/1e6 for index in range(max_index+1)}

        self.cosmology = cosmology
        self.single_index = single_index

    def define_tracer_objects(self):
        self.tracers_obj = []

        if self.single_index is None:
            for index in range(len(self.snus_CIB)):
                self.tracers_obj.append(
                    CIBIntensityTracers(
                        self.name_root, index,
                        self.cosmology, 
                        self.snus_CIB[f'{self.name_root}{index}'], 
                        self.zs_CIB, 
                        z_min=0., z_max=6.
                    )
                )

        elif self.single_index is not None:
            self.tracers_obj.append(
                CIBIntensityTracers(
                    self.name_root, self.single_index,
                    self.cosmology, 
                    self.snus_CIB[f'{self.name_root}{self.single_index}'], 
                    self.zs_CIB, 
                    z_min=0., z_max=6.
                )
            )
        
    def define_tracer_dict(self):
        self.define_tracer_objects()

        self.tracers_dict = {
            t.name: t.tracer for t in self.tracers_obj
        }
        return self.tracers_dict
    
class SaccWorkspace:
    def __init__(self, sacc_file=None, tracer_combinations=None):
        self.sacc_file = sacc_file

        if type(sacc_file) is str:
            self.data = sc.Sacc.load_fits(sacc_file)

        self.tracer_combinations = tracer_combinations

        self.aliases = {}

    def get_tracer(self, tracer_name):
        return self.data.get_tracer(tracer_name)

    def get_ell_cl(self, tracer1, tracer2):
        return self.data.get_ell_cl(tracer1, tracer2)
    
    def define_alias(self, tracer_name, alias):
        """
        Define an alias for a tracer.
        
        Parameters:
        tracer_name: str, name of the tracer
        alias: str, alias for the tracer
        """
        self.aliases[alias] = tracer_name

        print(self.aliases)

    def select_from_sacc(self, data_type):
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
    
    def get_covariance_matrix(self, data_type):
        cov_matrix = self.select_from_sacc(data_type).covariance.covmat

        return cov_matrix

    def cut_covariance_matrix(self, datatype, mask, width):
        
        cov_matrix = self.get_covariance_matrix(datatype)
        
        cut_ell_mask_full = np.tile(mask, width)

        new_cov_matrix = cov_matrix[cut_ell_mask_full, :][:, cut_ell_mask_full]
        return new_cov_matrix

    def get_c_ells(self, sacc_file=None, tracer_combinations=None):
        """
        Get the C_ell for a specific tracer combination.
        
        Parameters:
        tracer1: str, name of the first tracer
        tracer2: str, name of the second tracer
        
        Returns:
        measured_c_ells: list of numpy arrays, each containing the measured C_ells for a tracer combination
        """

        # Check if self.data is set, else use sacc_file function parameter
        if self.data is None and sacc_file is not None:
            self.data = sc.Sacc.load_fits(sacc_file)
        if self.tracer_combinations is None and tracer_combinations is not None:
            self.tracer_combinations = tracer_combinations
        
        measured_c_ells = []

        for tracer in self.tracer_combinations:
            if type(tracer) is not tuple:
                raise ValueError("Tracer combinations must be tuples of tracer names.")
            ell, cl = self.data.get_ell_cl(None, tracer[0], tracer[1], return_cov=False)
            measured_c_ells.append(cl)

        return measured_c_ells

class MaleubreModel():
    def __init__(self, measured_c_ells, tracer_combos, cosmology, Tracer1Workspace, Tracer2Workspace=None, sacc_workspace=None, logged_N=False):
        self.measured_c_ells = measured_c_ells
        self.Tracer1Workspace = Tracer1Workspace
        self.Tracer2Workspace = Tracer2Workspace
        self.tracer_combos = tracer_combos
        
        self.cosmology = cosmology

        self.logged_N = logged_N

        self.sacc_workspace = sacc_workspace if sacc_workspace is not None else None

        self.data = sacc_workspace.data if sacc_workspace is not None else None

        k_arr= np.geomspace(1E-4, 100, 256)
        a_arr = 1. / (1. + np.linspace(0, 6, 16)[::-1])
        pk_mm = ccl.linear_matter_power(self.cosmology, k_arr, a_arr)
        
        self.pk2d_mm = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=pk_mm, is_logp=False)

        self.pksquare_mm = ccl.pk2d.Pk2D(a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=k_arr**2 * pk_mm, is_logp=False)
        
        
        # Initialize the workspace for MCMC
        #self.mcmc_workspace = MCMCWorkspace(sacc_file)


        Tracer1Workspace.define_tracer_dict()
        Tracer2Workspace.define_tracer_dict() if Tracer2Workspace is not None else None

    def kernel_squared_integral(self, tracer, tracerwsp):

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
        

    def log_likelihood_function(self, b_gs, N_ggs, A_ggs, N_gnus=None, A_gnus=None, bpsfrs=None):

        if self.logged_N:
            N_ggs = np.power(10, N_ggs)
            if N_gnus is not None:
                N_gnus = np.power(10, N_gnus)

        tracers1 = self.Tracer1Workspace.define_tracer_dict() 

        try:
            tracers2 = self.Tracer2Workspace.define_tracer_dict()
        except AttributeError:
            tracers2 = tracers1

        theory_c_ells = []
        all_cut_c_ells = []

        for i in range(len(self.tracer_combos)):
            

            if self.tracer_combos[i][0] == self.tracer_combos[i][1]:

                cut_ells, cut_c_ells, mask = self.Tracer1Workspace.tracers_obj[i%4].get_cut_data(self.sacc_workspace)

                theory_c_ells.append(
                b_gs[i%4]**2 * ccl.angular_cl(
                    self.cosmology,
                    tracers1[self.tracer_combos[i][0]],
                    tracers1[self.tracer_combos[i][1]],
                    ell=cut_ells,
                    p_of_k_a=self.pk2d_mm) + N_ggs[i%4]*self.kernel_squared_integral(self.tracer_combos[i][0], self.Tracer1Workspace) + ccl.angular_cl(self.cosmology, tracers1[self.tracer_combos[i][0]], tracers1[self.tracer_combos[i][0]], ell=cut_ells, p_of_k_a=self.pksquare_mm) * A_ggs[i%4]
                )

                all_cut_c_ells.append(cut_c_ells)


            elif self.tracer_combos[i][0] != self.tracer_combos[i][1]: # notes cross-correlations

                cut_ells, cut_c_ells, mask = self.Tracer1Workspace.tracers_obj[i%4].get_cut_data(self.sacc_workspace, tracer_2=self.tracer_combos[i][1]) # get the cut data for the cross-correlation

                theory_c_ells.append(
                b_gs[i%4] * bpsfrs[i%4] * ccl.angular_cl( # $b_{g} b_{sfr} C_ell$
                    self.cosmology,
                    tracers1[self.tracer_combos[i][0]],
                    tracers2[self.tracer_combos[i][1]],
                    ell=cut_ells,
                    p_of_k_a=self.pk2d_mm) 
                    + N_gnus[i%4]*self.kernel_mixed_integral(f'{self.tracer_combos[i][0]}', f'{self.tracer_combos[i][1]}', self.Tracer1Workspace, self.Tracer2Workspace) 
                    + ccl.angular_cl(self.cosmology, tracers1[self.tracer_combos[i][0]], tracers2[self.tracer_combos[i][1]], ell=cut_ells, p_of_k_a=self.pksquare_mm) * A_gnus[i%4]
                )

                all_cut_c_ells.append(cut_c_ells)

        '''
        plt.figure(figsize=(10, 6))
        plt.plot(cut_ells, all_cut_c_ells[1], label='Measured C_ell')
        plt.plot(cut_ells, theory_c_ells[1], label='Theory C_ell')
        plt.xlabel('Ell')
        plt.ylabel('C_ell')
        plt.title('C_ell Comparison')
        plt.legend()

        plt.xscale('log')
        plt.yscale('log')

        plt.savefig('C_ell_comparison.png')
        '''
        mask_width = len(self.tracer_combos) if self.tracer_combos is not None else 1
        #print(f"Mask width: {mask_width}")
     

        covariance = self.sacc_workspace.cut_covariance_matrix('cl_00', mask, mask_width)

        icov = np.linalg.inv(covariance)

        diff = np.concatenate(all_cut_c_ells) - np.concatenate(theory_c_ells)
        
        logL = -0.5 *np.dot(diff, np.dot(icov, diff))

    
        return logL

        

        










    
    
        

