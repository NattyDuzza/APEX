import APEX as ap
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

k = np.linspace(0.1, 0.5, 20)

cosmo = ccl.Cosmology(
    Omega_c=0.261,
    Omega_b=0.049,
    h=0.677,
    n_s=0.9665,
    sigma8=0.8102,
    transfer_function="bbks",
    matter_power_spectrum="halofit")



cibwsp = ap.CIBIntensityTracerWorkspace(
    flux_fits_file="../CIB-Project/filtered_snu_planck.fits",
    cosmology=cosmo,
    tracer_name_root="CIBLenz__",
    single_index=3
)
bpsfrs = []

plt.figure(figsize=(10, 6))

'''
for j in range(4):
    bpsfrs = []
    for i in k:
        gdwsp = ap.GalaxyDensityTracerWorkspace(
        sacc_file="../CIB-Project/NEW-hsc_x_cib(857).fits",
        tracer_name_root="hsc_zbin",
        max_index=j,
        cosmology=cosmo)

        tracer_combos = [(f'hsc_zbin{j}', f'hsc_zbin{j}'),
                    
                    (f'hsc_zbin{j}', 'CIBLenz__3'),
                    ]

        s = ap.SaccWorkspace('../CIB-Project/NEW-hsc_x_cib(857).fits', tracer_combinations=tracer_combos)

        s.define_alias('cib_857GHz', 'CIBLenz__3')
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
            sacc_file='../CIB-Project/NEW-hsc_x_cib(857).fits',
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

    plt.plot(k, bpsfrs, marker='o', linestyle='-', label=f'Redshift bin {j}')
    '''
chosen_tracer = 'DESI_LRG__'

tracer_combos = [(f'{chosen_tracer}0', f'{chosen_tracer}0'),
                 (f'{chosen_tracer}1', f'{chosen_tracer}1'),
                 (f'{chosen_tracer}2', f'{chosen_tracer}2'),
                 (f'{chosen_tracer}3', f'{chosen_tracer}3'),

                 (f'{chosen_tracer}0', 'CIBLenz__3'),
                 (f'{chosen_tracer}1', 'CIBLenz__3'),
                 (f'{chosen_tracer}2', 'CIBLenz__3'),
                 (f'{chosen_tracer}3', 'CIBLenz__3')
                 ]

gdwsp = ap.GalaxyDensityTracerWorkspace(
        sacc_file="../CIB-Project/cls_desi_lrgs_x_cib.fits",
        tracer_name_root=chosen_tracer,
        max_index=3,
        cosmology=cosmo)

s = ap.SaccWorkspace('../CIB-Project/cls_desi_lrgs_x_cib.fits', tracer_combinations=tracer_combos, reverse_order=True)

s.define_alias('CIBLenz__857', 'CIBLenz__3')

bpsfrs0 = []
bpsfrs1 = []
bpsfrs2 = []
bpsfrs3 = []


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
        k_max = i,
    )
    mcmc = ap.MCMCWorkspace(
        sacc_file='../CIB-Project/cls_desi_lrgs_x_cib.fits',
        model=mmodel,
        likelihood_function='log_likelihood_function')
    
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

plt.savefig('DESI-bpsfr_vs_k.png')
