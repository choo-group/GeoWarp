import sys
sys.path.append('../..')

from simulators.simulator_material_tests import SimulatorTriaxial

import warp as wp
import numpy as np

import meshio


# Loading conditions
# TODO: initial stress
final_strain = 0.05
n_steps = 100
loading_rate = final_strain/n_steps
target_stress_xx = 0.0 #-390 #-425 #-500.0 #-60.0 # compression negative
target_stress_yy = 0.0 #-390 #-425 #-500.0 #-60.0
target_stress_zz = 0.0 #-390 #-425 #-500.0 #-60.0

p0 = -390 #-425
K0_consolidation = 0.45 #0.38

# Material properties
youngs_modulus = 12.82e4 #30e4 #14.4e4 #18.72e4 #2.016e4 # kPa # 
poisson_ratio = 0.424 #0.2
Ir = 250.0 #1000.0
elasticity_dict = {'youngs_modulus_initial': youngs_modulus, 'poisson_ratio_initial': poisson_ratio, 'Ir': Ir, 'kappa': 0.0015, 'G0': 4.5e4}
# plasticity_dict = {'M': 1.05, 'N': 0.4, 'pi': -65.0, 'tilde_lambda': 0.04, 'beta': 0.75, 'v_c0': 1.915, 'v_0': 1.63, 'h': 280.0}
# plasticity_dict = {'M': 1.05, 'N': 0.4, 'pi': -65.0, 'tilde_lambda': 0.04, 'beta': 0.75, 'v_c0': 1.915, 'v_0': 1.965, 'h': 280.0}

# # Validation with Jefferies, Geotechnique, 1993, Fig. 11 D667
# plasticity_dict = {'M': 1.20, 'N': 0.2, 'pi': -65.0-1e-6, 'tilde_lambda': 0.0135, 'beta': 1.0, 'v_c0': 1.751, 'v_0': 1.59, 'h': 280.0}

# # Validation with Jefferies, Geotechnique, 1993, Fig. 11 D662
# plasticity_dict = {'M': 1.26, 'N': 0.2, 'pi': -30.0-1e-6, 'tilde_lambda': 0.0134, 'beta': 1.0, 'v_c0': 1.761, 'v_0': 1.677, 'h': 280.0}


# # Validation with Jefferies, Book, 2015, Fig. 3.16 D662. 
# plasticity_dict = {'M': 1.26, 'N': 0.35, 'pi': -30.0-1e-6, 'tilde_lambda': 0.0134, 'beta': 1.0, 'v_c0': 1.761, 'v_0': 1.677, 'h': 200.0}

# # Validation with Jefferies, Book, 2015, Fig. 3.16 D682. 
# plasticity_dict = {'M': 1.18, 'N': 0.35, 'pi': -250.0-1e-6, 'tilde_lambda': 0.0134, 'beta': 1.0, 'v_c0': 1.732, 'v_0': 1.776, 'h': 45.0}


# Validation with Jefferies, Book, 2015, Fig. 3.16 D667. 
# plasticity_dict = {'M': 1.3, 'N': 0.35, 'pi': -65.0-1e-6, 'tilde_lambda': 0.0134, 'beta': 1.0, 'v_c0': 1.751, 'v_0': 1.59, 'h': 300.0}



# # Validation with Andrade & Ellison, 2008, Brasted dense
# plasticity_dict = {'M': 1.27, 'N': 0.4, 'pi': -534.47, 'tilde_lambda': 0.02, 'beta': 1.0, 'v_c0': 1.8911, 'v_0': 1.57, 'h': 120.0}


# Validation with Andrade & Ellison, 2008, Brasted loose
plasticity_dict = {'M': 1.27, 'N': 0.4, 'pi': -332.30, 'tilde_lambda': 0.02, 'beta': 1.0, 'v_c0': 1.8911, 'v_0': 1.75, 'h': 70.0}



# Material model
material_name = 'Nor-Sand'#'Drucker-Prager' 

# Solver
n_iter = 30 
tol = 1e-10
solver_name = 'Warp'




sim = SimulatorTriaxial(
                 n_iter=n_iter,
                 elasticity_dict=elasticity_dict,
                 material_name=material_name,
                 tol=tol,
                 plasticity_dict=plasticity_dict,
                 p0=p0,
                 K0_consolidation=K0_consolidation,
                 loading_rate=loading_rate,
                 target_stress_xx=target_stress_xx,
                 target_stress_yy=target_stress_yy,
                 target_stress_zz=target_stress_zz,
                 solver_name=solver_name
                 )



for step in range(1000):
    # print('Load step:', step+1)
    sim.advance_one_step(step, n_steps)

