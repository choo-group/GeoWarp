import sys
sys.path.append('..')

import warp as wp
import numpy as np

from material_models.material_utils import return_mapping_J2


@wp.kernel
def initialize_elastic_cto(rows: wp.array(dtype=wp.int32),
                           cols: wp.array(dtype=wp.int32),
                           vals: wp.array(dtype=wp.float64),
                           lame_lambda: wp.float64,
                           lame_mu: wp.float64):

    # elastic stiffness (isotropic) in Voigt notation (refer to: https://en.wikipedia.org/wiki/Hooke%27s_law)
    for i in range(3):
        for j in range(3):
            flattened_id = i*6 + j
            rows[flattened_id] = i
            cols[flattened_id] = j
            vals[flattened_id] = lame_lambda
            if i==j:
                vals[flattened_id] += wp.float64(2.)*lame_mu

    for i in range(3, 6):
        j = i
        flattened_id = i*6 + j
        rows[flattened_id] = i
        cols[flattened_id] = j
        vals[flattened_id] = lame_mu



@wp.kernel
def set_initial_stress(rhs: wp.array(dtype=wp.float64),
                       target_stress_xx: wp.float64,
                       target_stress_yy: wp.float64,
                       target_stress_zz: wp.float64):

    rhs[0] = target_stress_xx
    rhs[1] = target_stress_yy
    rhs[2] = target_stress_zz


@wp.kernel
def initial_loading_at_this_step(new_strain_vector: wp.array(dtype=wp.float64),
                                 loading_rate: wp.float64):
    new_strain_vector[2] += -loading_rate


@wp.kernel
def calculate_stress_residual_J2(new_strain_vector: wp.array(dtype=wp.float64), # Voigt notation
                                 lame_lambda: wp.float64,
                                 lame_mu: wp.float64,
                                 kappa: wp.float64,
                                 target_stress_xx: wp.float64,
                                 target_stress_yy: wp.float64,
                                 rhs: wp.array(dtype=wp.float64),
                                 saved_stress: wp.array(dtype=wp.mat33d)):

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)

    # Here assuming the strain only involves normal components (i.e., no shearing)
    new_strain = wp.matrix(
                 new_strain_vector[0], float64_zero, float64_zero,
                 float64_zero, new_strain_vector[1], float64_zero,
                 float64_zero, float64_zero, new_strain_vector[2],
                 shape=(3,3)
                 )

    e_real = return_mapping_J2(new_strain, lame_lambda, lame_mu, kappa)

    e_trace = wp.trace(e_real)

    stress_principal = lame_lambda*e_trace*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*e_real

    saved_stress[0] = stress_principal

    # Here assuming the stress only involves normal components (i.e., no shearing)
    target_stress = wp.matrix(
                    target_stress_xx, float64_zero, float64_zero,
                    float64_zero, target_stress_yy, float64_zero,
                    float64_zero, float64_zero, stress_principal[2,2],
                    shape=(3,3)
                    )
    stress_residual = target_stress - stress_principal

    # Assemble to rhs
    wp.atomic_add(rhs, 0, stress_residual[0,0])
    wp.atomic_add(rhs, 1, stress_residual[1,1])


@wp.kernel
def assemble_Jacobian_coo_format_material_tests(jacobian_wp: wp.array(dtype=wp.float64),
                                                rows: wp.array(dtype=wp.int32),
                                                cols: wp.array(dtype=wp.int32),
                                                vals: wp.array(dtype=wp.float64),
                                                dof_iter: wp.int32):
    column_index = wp.tid()

    rows[6*dof_iter + column_index] = dof_iter
    cols[6*dof_iter + column_index] = column_index
    vals[6*dof_iter + column_index] = jacobian_wp[column_index]



@wp.kernel
def from_increment_to_strain_vector_initial(strain_increment: wp.array(dtype=wp.float64),
                               new_strain_vector: wp.array(dtype=wp.float64)):
    i = wp.tid()

    new_strain_vector[i] += strain_increment[i]



@wp.kernel
def from_increment_to_solution_triaxial(strain_increment: wp.array(dtype=wp.float64),
                                        new_strain_vector: wp.array(dtype=wp.float64)):
    i = wp.tid()

    if i!=2 and i!=3 and i!=4:
        new_strain_vector[i] -= strain_increment[i] # TODO: check why
