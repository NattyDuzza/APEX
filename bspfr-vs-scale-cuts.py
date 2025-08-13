import APEX as ap
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

k = np.linspace(0.1, 0.5, 10)

cosmo = ccl.Cosmology(
    Omega_c=0.261,
    Omega_b=0.049,
    h=0.677,
    n_s=0.9665,
    sigma8=0.8102,
    transfer_function="bbks",
    matter_power_spectrum="halofit")

gdwsp = ap.GalaxyDensityTracerWorkspace(
    sacc_file="/home/nathand/Documents/AstroCode/CIB-Project/NEW-hsc_x_cib(857).fits",
    tracer_name_root="hsc_zbin",
    max_index=0,
    cosmology=cosmo
)

cibwsp = ap.CIBIntensityTracerWorkspace(
    flux_fits_file="../CIB-Project/filtered_snu_planck.fits",
    cosmology=cosmo,
    tracer_name_root="CIBLenz__",
    single_index=3
)

tracer_combos = [('hsc_zbin0', 'hsc_zbin0'),
                 
                 ('hsc_zbin0', 'CIBLenz__3'),
                ]

s = ap.SaccWorkspace('/home/nathand/Documents/AstroCode/CIB-Project/NEW-hsc_x_cib(857).fits', tracer_combinations=tracer_combos)

s.define_alias('cib_857GHz', 'CIBLenz__3')

bpsfrs = []

for i in k:
    mmodel = ap.MaleubreModel(
        Tracer1Workspace=gdwsp,
        Tracer2Workspace=cibwsp,
        tracer_combos=tracer_combos,
        sacc_workspace=s,
        cosmology=cosmo,
        logged_N=True,
        min_ell=100,
        max_ell=1000,
        k_max = i
    )
    mcmc = ap.MCMCWorkspace(
        sacc_file='/home/nathand/Documents/AstroCode/CIB-Project/NEW-hsc_x_cib(857).fits',
        model=mmodel,
        likelihood_function='log_likelihood_function')

    params = ['b_g0', 'N_gg0', 'A_gg0', 'N_gnu0', 'A_gnu0', 'bpsfr0']

    mcmc.set_param_priors(
        params=params,
        priors=[
            (0.75, 5), 
            (np.log10(1e-15), np.log10(1)), 
            (-100, 100), 
            (np.log10(1e-15), np.log10(1)), 
            (-100, 100), 
            (-3, 3),
        ]
    )

    mcmc.set_param_references(
        params=params,
        references=[
            1.1, 
            np.log10(1.7e-9), 
            7, 
            np.log10(1.7e-9), 
            7, 
            0.5,
        ]
    )

    mcmc.set_param_proposals(
        params=params,
        proposals=[
            0.1, 
            np.log10(1e-9), 
            1, 
            np.log10(1e-9), 
            1, 
            0.1,
        ]
    )

    mcmc.set_grouped_params({
        'b_gs': ['b_g0'],
        'N_ggs': ['N_gg0'],
        'A_ggs': ['A_gg0'],
        'N_gnus': ['N_gnu0'],
        'A_gnus': ['A_gnu0'],    
        'bpsfrs': ['bpsfr0']
    })

    mcmc.MCMC_config(params)

    sampler = mcmc.minimize_run()

    bpsfrs.append(sampler.products()['minimum']['bpsfr0'])

plt.figure(figsize=(10, 6))
plt.plot(k, bpsfrs, marker='o', linestyle='-', color='b')
plt.xlabel('k (Mpc^-1)')
plt.ylabel('bpsfr0')

plt.savefig('bpsfr_vs_k.png')
