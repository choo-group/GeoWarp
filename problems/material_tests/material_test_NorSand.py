import sys
sys.path.append('../..')

from simulators.simulator_material_tests import SimulatorTriaxial

import warp as wp
import numpy as np

import meshio


# Loading conditions
# TODO: initial stress
final_strain = 0.1
n_steps = 100
loading_rate = final_strain/n_steps
target_stress_xx = -100.0 # compression negative
target_stress_yy = -100.0
target_stress_zz = -100.0

# Material properties
youngs_modulus = 2.016e4 # kPa
poisson_ratio = 0.3
# plasticity_dict = {'M': 1.05, 'N': 0.4, 'pi': -65.0, 'tilde_lambda': 0.04, 'beta': 0.75, 'v_c0': 1.915, 'v_0': 1.63, 'h': 280.0}
plasticity_dict = {'M': 1.05, 'N': 0.4, 'pi': -65.0, 'tilde_lambda': 0.04, 'beta': 0.75, 'v_c0': 1.915, 'v_0': 1.965, 'h': 280.0}



# Material model
material_name = 'Nor-Sand'#'Drucker-Prager' 

# Solver
n_iter = 30 
tol = 1e-10
solver_name = 'Warp'




sim = SimulatorTriaxial(
                 n_iter=n_iter,
                 youngs_modulus=youngs_modulus,
                 poisson_ratio=poisson_ratio,
                 material_name=material_name,
                 tol=tol,
                 plasticity_dict=plasticity_dict,
                 loading_rate=loading_rate,
                 target_stress_xx=target_stress_xx,
                 target_stress_yy=target_stress_yy,
                 target_stress_zz=target_stress_zz,
                 solver_name=solver_name
                 )



for step in range(1000):
    # print('Load step:', step+1)
    sim.advance_one_step(step, n_steps)

