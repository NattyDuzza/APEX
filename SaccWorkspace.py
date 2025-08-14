import pymaster as nmt
import healpy as hp
import itertools

class SaccWorkspace:
    """ A class that handles the SACC file being used in the analysis, or creates one."""
    def __init__(self, sacc_file=None, tracer_combinations=None):
        self.sacc_file = sacc_file

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
        """ Get the covariance matrix for a specific data type from the SACC file.

        Parameters:
        data_type: str, data type for which the covariance matrix is to be extracted

        Returns:
        cov_matrix: numpy array, covariance matrix for the specified data type
        """

        cov_matrix = self.select_from_sacc(data_type).covariance.covmat

        return cov_matrix

    def cut_covariance_matrix(self, datatype, mask, width):
        """ Cut the covariance matrix to a specific range based on a mask and width.

        Parameters:
        datatype: str, data type for which the covariance matrix is to be obtained and then cut
        mask: numpy array, boolean mask indicating which elements to keep
        width: int, width of the mask to be applied - refers to the number of 'blocks' that need to be masked

        Returns:
        new_cov_matrix: numpy array, cut covariance matrix
        """
        
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
    

    class SaccFileCreation:
    # Sacc file creation - Using NaMaster
        def __init__(self, sacc_workspace):
            self.sacc_workspace = sacc_workspace

        def load_maps_data(self, max_index, file_root, pre_index_name, post_index_name=None):

            return [hp.read_map(file_root + pre_index_name + str(i) + post_index_name) for i in range(4)]

        def dictionarize_maps(self, map_list, map_headers):
            map_dict = {}

            for i in range(len(map_list)):
                map_dict[map_headers[i]] = map_list[i]


        def load_mask_data(self, file_root, mask_name):
            """ Load a mask from a file.

            Parameters:
            file_root: str, root path to the file
            mask_name: str, name of the mask file

            Returns:
            mask: numpy array, loaded mask data
            """
            return hp.read_map(file_root + mask_name)
        
        def define_nmt_fields(self, max_index, file_root, pre_index_name, mask_name, post_index_name=None):
            
            maps = self.load_maps_data(max_index, file_root, pre_index_name, post_index_name)
            mask = self.load_mask_data(file_root, mask_name)

            f_map = [nmt.NmtField(mask, [m]) for m in maps]

            return f_map
        
        def define_map_naming(self):
            pass

        def define_nmt_workspace(self, constituents, combinations, nside=1024, width=40):

            b = nmt.NmtBin.from_nside_linear(nside, width=width)

            workspaces = {}
            for i in range(len(combinations)):
                parts = []
                combo_parts = combinations[i].split('_')
                for j in range(len(combo_parts)):
                    parts.append(constituents[combo_parts[j]])

                workspaces[combinations[i]] = nmt.NmtWorkspace.from_fields(*parts, b)
            
            self.workspaces = workspaces
            return workspaces

        def define_nmt_covariance_workspaces(self, constituents, combinations):
            '''
            
            e.g. constituents = {'HSC': [f_map[0]], 'CIB': [f_map[1]],'}

            e.g. combinations = [HSC_HSC, HSC_CIB, CIB_CIB, HSC_CIB_HSC]
            '''
            self.constituents = constituents
            covariance_workspaces = {}

            for i in range(len(combinations)):
                parts = []
                combo_parts = combinations[i].split('_')
                for j in range(len(combo_parts)):
                    parts.append(constituents[combo_parts[j]][0])

                covariance_workspaces[combinations[i]] = nmt.NmtCovarianceWorkspace()
                covariance_workspaces[combinations[i]].compute_coupling_coefficients(*parts)
            
            self.covariance_workspaces = covariance_workspaces
            return covariance_workspaces


        def build_covariance_matrix(self, map_list, map_headers, map_bin_nums):
            '''
            
            map_bin_nums: list of integers, each integer corresponds to the number of bins in each map
            '''

            

            
            

