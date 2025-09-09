import APEX as ap
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

k = np.linspace(0.1, 0.5, 20)
k = np.append(k, 0.15)
k = np.append(k, 0.2)
k = np.append(k, 0.3)

k = np.unique(np.sort(k))

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
    single_index=2
)


plt.figure(figsize=(10, 6))
'''
bpsfrs_single = {}
bg_single={}

for j in range(4):
    bpsfrs_single[j] = []
    bg_single[j] = []

    for i in k:

        print(f"Running for k = {i} at refdshift bin {j}")

        gdwsp = ap.GalaxyDensityTracerWorkspace(
        sacc_file="../CIB-Project/NEW-hsc_x_cib(857).fits",
        tracer_name_root="hsc_zbin",
        single_index=j,
        cosmology=cosmo)

        tracer_combos = [(f'hsc_zbin{j}', f'hsc_zbin{j}'),
                    
                    (f'hsc_zbin{j}', 'CIBLenz__2'),
                    ]

        s = ap.SaccWorkspace('../CIB-Project/NEW-hsc_x_cib(857).fits', tracer_combinations=tracer_combos)

        s.define_alias('cib_857GHz', 'CIBLenz__2')

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

        bpsfrs_single[j].append(sampler.products()['minimum']['bpsfr0'])
        bg_single[j].append(sampler.products()['minimum']['b_g0'])
        '''

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

gdwsp = ap.GalaxyDensityTracerWorkspace(
        sacc_file="../CIB-Project/NEW-hsc_x_cib(857).fits",
        tracer_name_root=chosen_tracer,
        max_index=3,
        cosmology=cosmo)

s = ap.SaccWorkspace('../CIB-Project/NEW-hsc_x_cib(857).fits', tracer_combinations=tracer_combos)

s.define_alias('cib_857GHz', 'CIBLenz__2')

bg0 = []
bg1 = []
bg2 = []
bg3 = []

bpsfr0 = []
bpsfr1 = []
bpsfr2 = []
bpsfr3 = []


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
        beam_window = True
    )

    mcmc = ap.MCMCWorkspace(
        sacc_file='../CIB-Project/NEW-hsc_x_cib(857).fits',
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
            0.8, 0.9, 1.2, 1.5,
            -8, -8, -8, -8,
            7, 7, 7, 7,
            -12, -12, -12, -12,
            7, 7, 7, 7,
            0.059, 0.11, 0.15, 0.18,
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

    bg0.append(sampler.products()['minimum']['b_g0'])
    bg1.append(sampler.products()['minimum']['b_g1'])
    bg2.append(sampler.products()['minimum']['b_g2'])
    bg3.append(sampler.products()['minimum']['b_g3'])

    bpsfr0.append(sampler.products()['minimum']['bpsfr0'])
    bpsfr1.append(sampler.products()['minimum']['bpsfr1'])
    bpsfr2.append(sampler.products()['minimum']['bpsfr2'])
    bpsfr3.append(sampler.products()['minimum']['bpsfr3'])


plt.figure(figsize=(10, 6))
plt.title('Galaxy Bias vs Scale Cuts')

plt.plot(k, bg0, marker='o', linestyle='-', label='Redshift bin 0')
plt.plot(k, bg1, marker='o', linestyle='-', label='Redshift bin 1')
plt.plot(k, bg2, marker='o', linestyle='-', label='Redshift bin 2')
plt.plot(k, bg3, marker='o', linestyle='-', label='Redshift bin 3')

plt.xlabel('k (Mpc^-1)')
plt.ylabel('Galaxy bias (b_g)')
plt.legend()

plt.savefig('HSC-bg_vs_k(with beam).png')


plt.figure(figsize=(10, 6))
plt.title('fixed-BPSFR vs Scale Cuts')

plt.plot(k, bpsfr0, marker='o', linestyle='-', label='Redshift bin 0')
plt.plot(k, bpsfr1, marker='o', linestyle='-', label='Redshift bin 1')
plt.plot(k, bpsfr2, marker='o', linestyle='-', label='Redshift bin 2')
plt.plot(k, bpsfr3, marker='o', linestyle='-', label='Redshift bin 3')

plt.xlabel('k (Mpc^-1)')
plt.ylabel('BPSFR')
plt.legend()

plt.savefig('fixed-HSC-bpsfr_vs_k(with beam).png')

'''
plt.figure(figsize=(10, 6))
plt.title('Diffs')

plt.plot(k, np.abs(np.array(bpsfr0) - bpsfrs_single[0]), marker='o', linestyle='-', label='bpsfr0 diff')
plt.plot(k, np.abs(np.array(bpsfr1) - bpsfrs_single[1]), marker='o', linestyle='-', label='bpsfr1 diff')
plt.plot(k, np.abs(np.array(bpsfr2) - bpsfrs_single[2]), marker='o', linestyle='-', label='bpsfr2 diff')
plt.plot(k, np.abs(np.array(bpsfr3) - bpsfrs_single[3]), marker='o', linestyle='-', label='bpsfr3 diff')

plt.xlabel('k (Mpc^-1)')
plt.ylabel('Difference from single tracer BPSFR')
plt.legend()
plt.savefig('fixed-HSC-bpsfr_diff_vs_k.png')

plt.figure(figsize=(10, 6))
plt.title('Diffs for b_g')
plt.plot(k, np.abs(np.array(bg0) - bg_single[0]), marker='o', linestyle='-', label='b_g0 diff')
plt.plot(k, np.abs(np.array(bg1) - bg_single[1]), marker='o', linestyle='-', label='b_g1 diff')
plt.plot(k, np.abs(np.array(bg2) - bg_single[2]), marker='o', linestyle='-', label='b_g2 diff')
plt.plot(k, np.abs(np.array(bg3) - bg_single[3]), marker='o', linestyle='-', label='b_g3 diff')

plt.xlabel('k (Mpc^-1)')
plt.ylabel('Difference from single tracer b_g')
plt.legend()
plt.savefig('fixed-HSC-bg_diff_vs_k.png')
'''