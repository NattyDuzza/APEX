import APEX as ap
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

k = np.linspace(0.1, 0.5, 100)

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
    max_index=3,
    cosmology=cosmo
)

cibwsp = ap.CIBIntensityTracerWorkspace(
    flux_fits_file="/home/nathand/Documents/AstroCode/APEX/Developmental Tests/filtered_snu_planck.fits",
    cosmology=cosmo,
    tracer_name_root="CIBLenz__",
    single_index=2
)

chosen_tracer = 'hsc_zbin'

tracer_combos = [(f'{chosen_tracer}0', f'{chosen_tracer}0'),
                 (f'{chosen_tracer}1', f'{chosen_tracer}1'),
                 (f'{chosen_tracer}2', f'{chosen_tracer}2'),
                 (f'{chosen_tracer}3', f'{chosen_tracer}3'),

                 (f'{chosen_tracer}0', 'CIBLenz__2'),
                 (f'{chosen_tracer}1', 'CIBLenz__2'),
                 (f'{chosen_tracer}2', 'CIBLenz__2'),
                 (f'{chosen_tracer}3', 'CIBLenz__2')
                 ]

s = ap.SaccWorkspace('/home/nathand/Documents/AstroCode/CIB-Project/NEW-hsc_x_cib(857).fits', tracer_combinations=tracer_combos)

s.define_alias('cib_857GHz', 'CIBLenz__2')

bpsfrs0 = []
bpsfrs1 = []
bpsfrs2 = []
bpsfrs3 = []


for i in k:
    
    mmodel = ap.MaleubreModel(
        Tracer1Workspace=cibwsp,
        Tracer2Workspace=gdwsp,
        tracer_combos=tracer_combos,
        sacc_workspace=s,
        cosmology=cosmo,
        logged_N=True,
        k_max = i,
        pixel_window=True,
        beam_window=True,
    )

    mmodel.complete_precalculation()

    mcmc = ap.MCMCWorkspace(
        sacc_file='/home/nathand/Documents/AstroCode/CIB-Project/NEW-hsc_x_cib(857).fits',
        model=mmodel,
        likelihood_function='lightweight_log_likelihood_function')
    
    params = ['b_g0', 'b_g1', 'b_g2', 'b_g3',
          'N_gg0', 'N_gg1', 'N_gg2', 'N_gg3',
          'A_gg0', 'A_gg1', 'A_gg2', 'A_gg3',
          'N_gnu0', 'N_gnu1', 'N_gnu2', 'N_gnu3',
          'A_gnu0', 'A_gnu1', 'A_gnu2', 'A_gnu3',
          'bpsfr0', 'bpsfr1', 'bpsfr2', 'bpsfr3']

    mcmc.set_param_priors(
    params=params,
    priors=[
        (0, 4), (0, 4), (0, 4), (0, 4),
        (-12, -4), (-12, -4), (-12, -4), (-12, -4),
        (-100, 100), (-100, 100), (-100, 100), (-100, 100),
        (-15, -8), (-15, -8), (-15, -8), (-15, -8),
        (-100, 100), (-100, 100), (-100, 100), (-100, 100),
        (0, 1), (0, 1), (0, 1), (0, 1),
        ]
    )

    mcmc.set_param_references(
        params=params,
        references=[
            0.96, 1.13, 1.31, 1.66, 
            -7.11, -9.35, -9.89, -8.65, 
            4.84, 8.00, 11.71, 27.02,
            -12.25, -12.75, -12.75, -11.94, 
            0.54, 0.91, 1.10, 0.67,
            0.019, 0.051, 0.089, 0.14,
        ]
    )

    mcmc.set_param_proposals(
        params=params,
        proposals=[
            0.1, 0.1, 0.1, 0.1,
            0.1, 0.1, 0.1, 0.1,
            1, 1, 1, 1,
            0.1, 0.1, 0.1, 0.1,
            1, 1, 1, 1,
            0.003, 0.003, 0.003, 0.003,
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

    bpsfrs0.append(sampler.products()['minimum']['bpsfr0'])
    bpsfrs1.append(sampler.products()['minimum']['bpsfr1'])
    bpsfrs2.append(sampler.products()['minimum']['bpsfr2'])
    bpsfrs3.append(sampler.products()['minimum']['bpsfr3'])



plt.plot(k, bpsfrs0, marker='o', linestyle='-', label='Redshift bin 0')
plt.plot(k, bpsfrs1, marker='o', linestyle='-', label='Redshift bin 1')
plt.plot(k, bpsfrs2, marker='o', linestyle='-', label='Redshift bin 2')
plt.plot(k, bpsfrs3, marker='o', linestyle='-', label='Redshift bin 3')



plt.xlabel('k (Mpc^-1)')
plt.ylabel('bpsfr0')
plt.legend()

plt.savefig('HSC-bpsfr_vs_k.png')
