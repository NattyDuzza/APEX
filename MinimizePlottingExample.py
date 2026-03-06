import APEX as ap
import pyccl as ccl
import numpy as np

cosmo = ccl.Cosmology(
    Omega_c=0.261,
    Omega_b=0.049,
    h=0.677,
    n_s=0.9665,
    sigma8=0.8102,
    transfer_function="bbks",
    matter_power_spectrum="halofit")

chosen_map = '../CIB-Project/NEW-hsc_x_cib(857).fits'
chosen_tracer = 'hsc_zbin'

gdwsp = ap.GalaxyDensityTracerWorkspace(
    sacc_file=chosen_map,
    tracer_name_root=chosen_tracer,
    max_index=3,
    cosmology=cosmo
)

cibwsp = ap.CIBIntensityTracerWorkspace(
    flux_fits_file="../CIB-Project/filtered_snu_planck.fits",
    cosmology=cosmo,
    tracer_name_root="CIBLenz__",
    single_index=3
)

tracer_combos = [(f'{chosen_tracer}0', f'{chosen_tracer}0'),
                 (f'{chosen_tracer}1', f'{chosen_tracer}1'),
                 (f'{chosen_tracer}2', f'{chosen_tracer}2'),
                 (f'{chosen_tracer}3', f'{chosen_tracer}3'),

                 (f'{chosen_tracer}0', 'CIBLenz__3'),
               
                 (f'{chosen_tracer}1', 'CIBLenz__3'),
                 
                 (f'{chosen_tracer}2', 'CIBLenz__3'),
              
                 (f'{chosen_tracer}3', 'CIBLenz__3')
                 ]

s = ap.SaccWorkspace(chosen_map, tracer_combinations=tracer_combos)

s.define_alias('cib_857GHz', 'CIBLenz__3')

mmodel = ap.MaleubreModel(
    Tracer1Workspace=gdwsp,
    Tracer2Workspace=cibwsp,
    tracer_combos=tracer_combos,
    sacc_workspace=s,
    cosmology=cosmo,
    logged_N=True,
    min_ell=100,
    max_ell=1000
)

mcmc = ap.MCMCWorkspace(
    sacc_file=chosen_map,
    model=mmodel,
    likelihood_function='log_likelihood_function',

)

params = ['b_g0', 'b_g1', 'b_g2', 'b_g3',
          'N_gg0', 'N_gg1', 'N_gg2', 'N_gg3',
          'A_gg0', 'A_gg1', 'A_gg2', 'A_gg3',
          'N_gnu0', 'N_gnu1', 'N_gnu2', 'N_gnu3',
          'A_gnu0', 'A_gnu1', 'A_gnu2', 'A_gnu3',
          'bpsfr0', 'bpsfr1', 'bpsfr2', 'bpsfr3']

mcmc.set_param_priors(
    params=params,
    priors=[
        (0.75, 5), (0.75, 5), (0.75, 5), (0.75, 5), 
        (np.log10(1e-15), np.log10(1)), (np.log10(1e-15), np.log10(1)), (np.log10(1e-15), np.log10(1)), (np.log10(1e-15), np.log10(1)), 
        (-100, 100), (-100, 100), (-100, 100), (-100, 100),
        (np.log10(1e-15), np.log10(1)), (np.log10(1e-15), np.log10(1)), (np.log10(1e-15), np.log10(1)), (np.log10(1e-15), np.log10(1)), 
        (-100, 100), (-100, 100), (-100, 100), (-100, 100),
        (-3, 3), (-3, 3), (-3, 3), (-3, 3),
    ]
)

mcmc.set_param_references(
    params=params,
    references=[
        1.1, 1.1, 1.1, 1.1, 
        np.log10(1.7e-9), np.log10(1.7e-9), np.log10(1.7e-9), np.log10(1.7e-9), 
        7, 7, 7, 7,
        np.log10(1.7e-9), np.log10(1.7e-9), np.log10(1.7e-9), np.log10(1.7e-9), 
        7, 7, 7, 7,
        0.5, 0.5, 0.5, 0.5,
    ]
)

mcmc.set_param_proposals(
    params=params,
    proposals=[
        0.1, 0.1, 0.1, 0.1, 
        np.log10(1e-9), np.log10(1e-9), np.log10(1e-9), np.log10(1e-9),
        1, 1, 1, 1,
        np.log10(1e-9), np.log10(1e-9), np.log10(1e-9), np.log10(1e-9),
        1, 1, 1, 1,
        0.1, 0.1, 0.1, 0.1,
    ]
)

mcmc.set_grouped_params({
    'b_gs': ['b_g0', 'b_g1', 'b_g2', 'b_g3'],
    'N_ggs': ['N_gg0', 'N_gg1', 'N_gg2', 'N_gg3'],
    'A_ggs': ['A_gg0', 'A_gg1', 'A_gg2', 'A_gg3'],
    'N_gnus': ['N_gnu0', 'N_gnu1', 'N_gnu2', 'N_gnu3'],
    'A_gnus': ['A_gnu0', 'A_gnu1', 'A_gnu2', 'A_gnu3'],    
    'bpsfrs': ['bpsfr0', 'bpsfr1', 'bpsfr2', 'bpsfr3']
})

mcmc.MCMC_config(params, sampler_info={'minimize': {'seed':42}})

sampler = mcmc.minimize_run()


hsc_b_gs = [sampler.products()['minimum'][f'b_g{i}'] for i in range(4)]
bpsfrs = [sampler.products()['minimum'][f'bpsfr{i}'] for i in range(4)]

b_g_err = [ap.estimate_error(chain_root='../CIB-Project/apex-outputs/loggedNgg/corrected-HSC-full-data-vector-test', column_name=f'b_g{i}') for i in range(4)]
bpsfr_err = [ap.estimate_error(chain_root='../CIB-Project/apex-outputs/loggedNgg/corrected-HSC-full-data-vector-test', column_name=f'bpsfr{i}') for i in range(4)]

hsc_z = [0.5218218413270964, 0.7372666514723796, 1.0497215014728178, 1.3163402650789335]

jego_z = [0.21, 0.37, 0.50, 0.63, 1.12, 1.87]

jego_b_g = [1.073, 1.382, 1.30, 1.736, 2.075, 2.246]
jego_b_g_err = [0.045, 0.026, 0.017, 0.019, 0.148, 0.185]

jego_bpsfr = [-0.003, 0.034, 0.039, 0.051, 0.119, 0.227]
jego_bpsfr_err = [0.011, 0.008, 0.007, 0.007, 0.014, 0.030]


import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot bias data
ax1.errorbar(jego_z, jego_b_g, yerr=jego_b_g_err, fmt='o', label='Jego et al.', color='blue')
ax1.errorbar(hsc_z, hsc_b_gs, yerr=b_g_err, fmt='o', label='HSC', color='orange')

ax2.errorbar(jego_z, jego_bpsfr, yerr=jego_bpsfr_err, fmt='o', label='Jego et al.', color='blue')
ax2.errorbar(hsc_z, bpsfrs, yerr=bpsfr_err, fmt='o', label='HSC', color='orange')

ax1.set_xlabel('Redshift')
ax1.set_ylabel('Bias')
ax1.set_title('Bias Comparison')
ax1.legend()

ax2.set_xlabel('Redshift')
ax2.set_ylabel('bPSFR')
ax2.set_title('bPSFR Comparison')
ax2.legend()

plt.savefig('bias_bpsfr_comparison.png')