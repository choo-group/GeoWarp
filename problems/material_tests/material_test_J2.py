import sys
sys.path.append('../..')

from simulators.simulator_material_tests import SimulatorTriaxial

import warp as wp
import numpy as np

import meshio


# Loading conditions
# TODO: initial stress
final_strain = 0.3
n_steps = 300
loading_rate = final_strain/n_steps
target_stress_xx = -10.0 # tension positive
target_stress_yy = -10.0
target_stress_zz = -10.0

# Material properties
youngs_modulus = 1000.0 # kPa
poisson_ratio = 0.3
plasticity_dict = {'kappa': 30.0/np.sqrt(3.0)*np.sqrt(2.0)}

# Material model
material_name = 'J2' #'Hencky elasticity' #'Neo-Hookean'

# Solver
n_iter = 20
tol = 1e-10




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
                 target_stress_zz=target_stress_zz
                 )



for step in range(n_steps):
    print('Load step:', step+1)
    sim.advance_one_step(step, n_steps)

