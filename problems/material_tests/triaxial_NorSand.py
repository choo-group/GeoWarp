import warp as wp
import numpy as np

import warp.sparse as wps
import warp.optim.linear

# Author: Yidong Zhao (ydzhao94@gmail.com)
# Triaxial test for Nor-Sand
# The implementation of the Nor-Sand model follows [Borja & Andrade, CMAME, 2006]

wp.init()

# ===================== Material properties =====================
# Brasted sand referring to [Andrade & Ellison, JGGE, 2008]
G0 = 4.5e4 # Shear modulus, kPa
kappa = 0.0015 # Gradient of swelling
tilde_lambda = 0.02 # Gradient of compression
M = 1.27 # CSL slope
N = 0.4 # Curvature of yield surface
beta = 1.0 # Shape factor
vc0 = 1.8911 # Reference specific volume

# Loose
pc = -715. # Initial preconsolidation pressure, kPa
p0 = -390. # Mean normal effective stress after consolidation, kPa
pi = pc * (1. - N)**((1. - N)/N) # Initial image stress, kPa
v0 = 1.75 # Initial specific volume
h = 70. # Hardening coefficient
K0_consolidation = 0.45 # Ratio of the effective horizontal stress to the effective vertical stress

# # Dense
# pc = -1150. # kPa
# p0 = -425. # kPa
# pi = pc * (1. - N)**((1. - N)/N) # kPa
# v0 = 1.57 
# h = 120.0 
# K0_consolidation = 0.38 



K0 = p0 / (-kappa) # Initial bulk modulus (pressure-dependent), kPa
lame_lambda = K0 - 2.0/3.0 * G0 # Lame's first parameter, kPa
lame_mu = G0 # Lame's second parameter, kPa

# ===================== Loading conditions =====================
final_axial_strain = 0.1
n_steps = 500
loading_rate = final_axial_strain/n_steps
slope_K0_pq = (1.-K0_consolidation)/(1./3.*(1.+2.*K0_consolidation)) # Ratio of p to q. This is derived based on K0
target_stress_xx = (3.-slope_K0_pq)/3. * p0 # Radial target stress
target_stress_yy = target_stress_xx # Radial target stress
target_stress_zz = (3.+2*slope_K0_pq)/3. * p0 # Axial target stress

tol = 1e-10

n_iter = 10 # Maximum iteration number for global Newton solver
n_iter_local = 15 # Maximum iteration number for local return mapping

# ===================== Warp arrays =====================
rows = wp.zeros(shape=36, dtype=wp.int32) # Row index for each non-zero. Refer to Coordinate Format (COO) for details: https://lectures.scientific-python.org/advanced/scipy_sparse/coo_array.html
cols = wp.zeros(shape=36, dtype=wp.int32) # Column index for each non-zero
vals = wp.zeros(shape=36, dtype=wp.float64) # Value for each non-zero

cto = wps.bsr_zeros(6, 6, block_type=wp.float64) # Consisten tangent operator
rhs = wp.zeros(shape=6, dtype=wp.float64) # Residual vector
strain_increment = wp.zeros(shape=6, dtype=wp.float64) # Note this includes both elastic and plastic strains
new_trial_strain_array = wp.zeros(shape=6, dtype=wp.float64, requires_grad=True)
new_total_strain_array = wp.zeros(shape=6, dtype=wp.float64)
new_elastic_strain_array = wp.zeros(shape=6, dtype=wp.float64)

stiffness = wps.bsr_zeros(6, 6, block_type=wp.float64) # Stiffness tensor in matrix notation
new_pi_array = wp.zeros(shape=1, dtype=wp.float64) 
old_pi_array = wp.zeros(shape=1, dtype=wp.float64)

# Arrays for saving history values during local iteration.
real_strain_history = wp.zeros(shape=n_iter_local+1, dtype=wp.mat33d, requires_grad=True) # This is important to ensure correct gradient calculation
pi_history = wp.zeros(shape=n_iter_local+1, dtype=wp.float64, requires_grad=True) # TODO: Check whether it's necessary to save the pi history?
delta_lambda_history = wp.zeros(shape=n_iter_local+1, dtype=wp.float64, requires_grad=True) # TODO: Check whether it's necessary to save the delta_lambda history?

# Saved quantities for post-processing
saved_local_residual = wp.zeros(shape=n_iter_local, dtype=wp.vec4d)
saved_stress = wp.zeros(shape=1, dtype=wp.mat33d)


# ===================== Warp kernels =====================
@wp.kernel
def initialize_stiffness(rows: wp.array(dtype=wp.int32),
                         cols: wp.array(dtype=wp.int32),
                         vals: wp.array(dtype=wp.float64),
                         lame_lambda: wp.float64,
                         lame_mu: wp.float64
                         ):
    # Elastic stiffness (isotropic) in Voigt notation (refer to: https://en.wikipedia.org/wiki/Hooke%27s_law)
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
                       target_stress_zz: wp.float64
                       ):
    rhs[0] = target_stress_xx
    rhs[1] = target_stress_yy
    rhs[2] = target_stress_zz

@wp.kernel
def initialize_pi(new_pi_array: wp.array(dtype=wp.float64),
                  pi: wp.float64
                  ):
    new_pi_array[0] = pi

@wp.kernel
def set_old_pi_to_new(new_pi_array: wp.array(dtype=wp.float64),
                      old_pi_array: wp.array(dtype=wp.float64)
                      ):
    old_pi_array[0] = new_pi_array[0]

@wp.kernel
def from_increment_to_strain_initial(strain_increment: wp.array(dtype=wp.float64),
                                     strain: wp.array(dtype=wp.float64)
                                     ):
    i = wp.tid()
    strain[i] += strain_increment[i]

@wp.kernel
def from_increment_to_strain(strain_increment: wp.array(dtype=wp.float64),
                             strain: wp.array(dtype=wp.float64)
                             ):
    i = wp.tid()

    # Only x and y components are unknown in triaxial loading
    if i!=2 and i!=3 and i!=4:
        strain[i] -= strain_increment[i]

@wp.kernel
def get_trial_strain_triaxial(new_trial_strain_array: wp.array(dtype=wp.float64),
                              loading_rate: wp.float64
                              ):
    new_trial_strain_array[2] += -loading_rate

@wp.func
def eta_function_NorSand(M: wp.float64,
                         N: wp.float64,
                         P: wp.float64,
                         pi: wp.float64
                         ):
    eta = wp.float64(0.0)
    if wp.abs(N) < 1e-10:
        eta = M * (wp.float64(1.0) + wp.log(pi/P))
    else:
        eta = (M/N) * (wp.float64(1.0) - (wp.float64(1.0)-N)*wp.pow(P/pi, N/(wp.float64(1.0)-N)))

    return eta


@wp.func
def yield_function_NorSand(P: wp.float64,
                           Q: wp.float64,
                           M: wp.float64,
                           N: wp.float64,
                           pi: wp.float64
                           ):
    eta = eta_function_NorSand(M, N, P, pi)

    return Q + eta * P

@wp.func
def dEtadP_function_NorSand(M: wp.float64,
                            N: wp.float64,
                            P: wp.float64,
                            pi: wp.float64
                            ):
    dEtadP = wp.float64(0.0)
    if wp.abs(N) < 1e-10:
        dEtadP = M * wp.float64(-1.0)/P
    else:
        dEtadP = M * (wp.float64(-1.0)*wp.pow(P/pi, (wp.float64(2.0)*N-wp.float64(1.0))/(wp.float64(1.0)-N))/pi)

    return dEtadP

@wp.func
def dEtadPi_function_NorSand(M: wp.float64,
                             N: wp.float64,
                             P: wp.float64,
                             pi: wp.float64
                             ):
    dEtadPi = wp.float64(0.0)
    if wp.abs(N) < 1e-10:
        dEtadPi = M/pi
    else:
        dEtadPi = M * (wp.float64(-1.0)*wp.pow(P/pi, (wp.float64(2.0)*N-wp.float64(1.0))/(wp.float64(1.0)-N)) * (-P)/wp.pow(pi, wp.float64(2.0)))

    return dEtadPi

@wp.func
def dPistar_dP_function_NorSand(M: wp.float64,
                                N: wp.float64,
                                alpha_bar: wp.float64,
                                psi_i: wp.float64
                                ):
    dPistar_dP = wp.float64(0.0)
    if wp.abs(N) < 1e-10:
        dPistar_dP = wp.exp(alpha_bar*psi_i/M)
    else:
        dPistar_dP = wp.pow(wp.float64(1.0)-alpha_bar*psi_i*N/M, (N-wp.float64(1.0))/N)

    return dPistar_dP

@wp.func
def dPistar_dPi_function_NorSand(M: wp.float64,
                                 N: wp.float64,
                                 alpha_bar: wp.float64,
                                 P: wp.float64,
                                 psi_i: wp.float64,
                                 dPsii_dPi: wp.float64
                                 ):
    dPistar_dPi = wp.float64(0.0)
    if wp.abs(N) < 1e-10:
        dPistar_dPi = P * wp.exp(alpha_bar*psi_i/M) * alpha_bar/M * dPsii_dPi
    else:
        dPistar_dPi = P * (N-wp.float64(1.0))/N * wp.pow(wp.float64(1.0)/(wp.float64(1.0)-alpha_bar*psi_i*N/M), wp.float64(1.0)/N) * (-alpha_bar*N/M) * (dPsii_dPi)

    return dPistar_dPi

@wp.func
def return_mapping_NorSand(kappa: wp.float64,
                           tilde_lambda: wp.float64,
                           M: wp.float64,
                           N: wp.float64,
                           beta: wp.float64,
                           vc0: wp.float64,
                           p0: wp.float64,
                           v0: wp.float64,
                           h: wp.float64,
                           lame_mu: wp.float64,
                           initial_volumetric_strain: wp.float64,
                           old_pi_array: wp.array(dtype=wp.float64),
                           new_pi_array: wp.array(dtype=wp.float64),
                           trial_strain: wp.mat33d,
                           total_strain: wp.mat33d,
                           real_strain_history: wp.array(dtype=wp.mat33d),
                           pi_history: wp.array(dtype=wp.float64),
                           delta_lambda_history: wp.array(dtype=wp.float64),
                           saved_local_residual: wp.array(dtype=wp.vec4d),
                           tol: wp.float64
                           ):

    alpha_bar = wp.float64(-3.5)/beta
    old_pi = old_pi_array[0]
    pi_history[0] = old_pi

    # Trial state
    real_strain = trial_strain
    eps_v = wp.trace(trial_strain)
    eps_s = wp.sqrt(wp.float64(2.0)/wp.float64(9.0) * ((trial_strain[0,0]-trial_strain[1,1])*(trial_strain[0,0]-trial_strain[1,1]) + (trial_strain[1,1]-trial_strain[2,2])*(trial_strain[1,1]-trial_strain[2,2]) + (trial_strain[0,0]-trial_strain[2,2])*(trial_strain[0,0]-trial_strain[2,2])))

    n_e_trial_deviatoric = wp.vec3d()
    if eps_s > tol:
        n_e_trial_deviatoric = wp.vec3d(
                               wp.sqrt(wp.float64(2.0)/wp.float64(3.0)) * (trial_strain[0,0] - eps_v/wp.float64(3.0))/(eps_s),
                               wp.sqrt(wp.float64(2.0)/wp.float64(3.0)) * (trial_strain[1,1] - eps_v/wp.float64(3.0))/(eps_s),
                               wp.sqrt(wp.float64(2.0)/wp.float64(3.0)) * (trial_strain[2,2] - eps_v/wp.float64(3.0))/(eps_s),
                               )
    
    # Pressure-dependent hyperelasticity
    # Refer to section 2.1 of [Borja & Andrade, CMAME, 2006] for details
    omega = -(eps_v - initial_volumetric_strain)/kappa
    P_trial = p0 * wp.exp(omega)
    Q_trial = wp.float64(3.0)*lame_mu * eps_s

    # Update volume
    eps_v_total = wp.trace(total_strain)
    new_J = wp.exp(eps_v_total)
    new_v = new_J * v0

    # Check whether eps_v > 0 (tension case)
    if eps_v > tol:
        real_strain = wp.mat33d()
    else:
        # Check yield
        yield_y = yield_function_NorSand(P_trial, Q_trial, M, N, old_pi)

        if yield_y < tol:
            pass # elasticity
        else:
            # plasticity
            # Local Newton iteration
            delta_lambda = wp.float64(0.)
            delta_lambda_history[0] = delta_lambda
            real_strain_history[0] = trial_strain

            eta = eta_function_NorSand(M, N, P_trial, old_pi)

            # Local iteration (treat pi as an independent variable)
            # Return mapping in strain invariant space
            for local_iter in range(15):
                eps_real_v = wp.trace(real_strain_history[local_iter])
                eps_real_s = wp.sqrt(wp.float64(2.0)/wp.float64(9.0) * ((real_strain_history[local_iter][0,0]-real_strain_history[local_iter][1,1])*(real_strain_history[local_iter][0,0]-real_strain_history[local_iter][1,1]) + (real_strain_history[local_iter][1,1]-real_strain_history[local_iter][2,2])*(real_strain_history[local_iter][1,1]-real_strain_history[local_iter][2,2]) + (real_strain_history[local_iter][0,0]-real_strain_history[local_iter][2,2])*(real_strain_history[local_iter][0,0]-real_strain_history[local_iter][2,2])))
                new_pi = pi_history[local_iter]
                delta_lambda = delta_lambda_history[local_iter]

                dEtadP = dEtadP_function_NorSand(M, N, P_trial, new_pi)
                dEtadPi = dEtadPi_function_NorSand(M, N, P_trial, new_pi)
                eta = eta_function_NorSand(M, N, P_trial, new_pi)

                dFdP = (eta-M)/(wp.float64(1.0)-N) # see Eq. (24) in [Andrade & Borja, IJNME, 2006]
                dFdQ = wp.float64(1.)

                # Residual
                psi_i = new_v - vc0 + tilde_lambda * wp.log(-new_pi)
                pi_star = wp.float64(0.)
                if wp.abs(N) < tol:
                    pi_star = P_trial * wp.exp(alpha_bar * psi_i / M)
                else:
                    pi_star = P_trial * wp.pow(wp.float64(1.0) - alpha_bar*psi_i*N/M, ((N-wp.float64(1.0))/N))

                yield_y = yield_function_NorSand(P_trial, Q_trial, M, N, new_pi)
                hardening = new_pi - old_pi + h*(new_pi-pi_star)*delta_lambda

                residual = wp.vec4d(
                           eps_real_v - eps_v + delta_lambda * beta * dFdP,
                           eps_real_s - eps_s + delta_lambda * dFdQ,
                           yield_y,
                           hardening
                           )
                saved_local_residual[local_iter] = residual

                # Local jacobian
                # Pressure-dependent hyperelasticity
                omega_new = -(eps_real_v-initial_volumetric_strain) / kappa
                d2F_dPdepsv = wp.float64(1.0)/(wp.float64(1.0)-N) * dEtadP * (p0 * wp.exp(omega_new) / wp.float64(-kappa))
                d2F_dPdPi = wp.float64(1.0)/(wp.float64(1.0)-N) * dEtadPi

                dFdepsv = (dEtadP*P_trial + eta) * (p0 * wp.exp(omega_new) / (-kappa))
                dFdepss = wp.float64(3.0) * lame_mu
                dFdPi = P_trial * dEtadPi

                dPistar_dP = dPistar_dP_function_NorSand(M, N, alpha_bar, psi_i)
                dPsii_dPi = tilde_lambda/new_pi
                dPistar_dPi = dPistar_dPi_function_NorSand(M, N, alpha_bar, P_trial, psi_i, dPsii_dPi)
                dHdepsv = -h*delta_lambda*dPistar_dP * (p0 * wp.exp(omega_new) / (-kappa))
                dHdPi = wp.float64(1.0) + h*delta_lambda*(wp.float64(1.0) - dPistar_dPi)

                jacobian = wp.mat44d(
                           wp.float64(1.0)+delta_lambda*beta*d2F_dPdepsv, wp.float64(0.0), beta*dFdP, delta_lambda*beta*d2F_dPdPi,
                           wp.float64(0.0), wp.float64(1.0), dFdQ, wp.float64(0.0),
                           dFdepsv, dFdepss, wp.float64(0.0), dFdPi,
                           dHdepsv, wp.float64(0.0), h*(new_pi-pi_star), dHdPi
                           )

                xdelta = wp.inverse(jacobian) @ residual

                new_eps_real_v = eps_real_v - xdelta[0]
                new_eps_real_s = eps_real_s - xdelta[1]
                delta_lambda_history[local_iter+1] = delta_lambda_history[local_iter] - xdelta[2]
                pi_history[local_iter+1] = pi_history[local_iter] - xdelta[3]

                # Update strain
                # From invariants to principal values
                real_strain_history[local_iter+1] = wp.mat33d(
                                                    new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[0]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0), wp.float64(0.0),
                                                    wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[1]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0),
                                                    wp.float64(0.0), wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[2]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0))
                                                    )

                # Update stress
                eps_v_tmp = wp.trace(real_strain_history[local_iter+1])
                eps_s_tmp = wp.sqrt(wp.float64(2.0)/wp.float64(9.0) * ((real_strain_history[local_iter+1][0,0]-real_strain_history[local_iter+1][1,1])*(real_strain_history[local_iter+1][0,0]-real_strain_history[local_iter+1][1,1]) + (real_strain_history[local_iter+1][1,1]-real_strain_history[local_iter+1][2,2])*(real_strain_history[local_iter+1][1,1]-real_strain_history[local_iter+1][2,2]) + (real_strain_history[local_iter+1][0,0]-real_strain_history[local_iter+1][2,2])*(real_strain_history[local_iter+1][0,0]-real_strain_history[local_iter+1][2,2])))
                # Pressure-dependent hyperelasticity
                omega_new = -(eps_v_tmp-initial_volumetric_strain) / kappa 
                P_trial = p0 * wp.exp(omega_new)
                Q_trial = wp.float64(3.0)*lame_mu * eps_s_tmp

                real_strain = wp.mat33d(
                              real_strain_history[local_iter+1][0,0], wp.float64(0.), wp.float64(0.),
                              wp.float64(0.), real_strain_history[local_iter+1][1,1], wp.float64(0.),
                              wp.float64(0.), wp.float64(0.), real_strain_history[local_iter+1][2,2]
                              )

            new_pi_array[0] = pi_history[15]

    return real_strain




@wp.kernel
def calculate_stress_residual_NorSand(kappa: wp.float64,
                                      tilde_lambda: wp.float64,
                                      M: wp.float64,
                                      N: wp.float64,
                                      beta: wp.float64,
                                      vc0: wp.float64,
                                      p0: wp.float64,
                                      v0: wp.float64,
                                      h: wp.float64,
                                      lame_mu: wp.float64,
                                      initial_volumetric_strain: wp.float64,
                                      old_pi_array: wp.array(dtype=wp.float64),
                                      new_pi_array: wp.array(dtype=wp.float64),
                                      new_trial_strain_array: wp.array(dtype=wp.float64),
                                      new_total_strain_array: wp.array(dtype=wp.float64),
                                      new_elastic_strain_array: wp.array(dtype=wp.float64),
                                      real_strain_history: wp.array(dtype=wp.mat33d),
                                      pi_history: wp.array(dtype=wp.float64),
                                      delta_lambda_history: wp.array(dtype=wp.float64),
                                      target_stress_xx: wp.float64,
                                      target_stress_yy: wp.float64,
                                      target_stress_zz: wp.float64,
                                      rhs: wp.array(dtype=wp.float64),
                                      saved_local_residual: wp.array(dtype=wp.vec4d),
                                      saved_stress: wp.array(dtype=wp.mat33d),
                                      tol: wp.float64
                                      ):

    float64_zero = wp.float64(0.0)

    # Get trial strain matrix
    trial_strain = wp.mat33d(new_trial_strain_array[0], float64_zero, float64_zero,
                             float64_zero, new_trial_strain_array[1], float64_zero,
                             float64_zero, float64_zero, new_trial_strain_array[2])
    total_strain = wp.mat33d(new_total_strain_array[0], float64_zero, float64_zero,
                             float64_zero, new_total_strain_array[1], float64_zero,
                             float64_zero, float64_zero, new_total_strain_array[2])

    new_J = wp.exp(wp.trace(total_strain))

    # e_real = trial_strain
    e_real = return_mapping_NorSand(kappa, tilde_lambda, M, N, beta, vc0, p0, v0, h, lame_mu, initial_volumetric_strain, old_pi_array, new_pi_array, trial_strain, total_strain, real_strain_history, pi_history, delta_lambda_history, saved_local_residual, tol)
    new_elastic_strain_array[0] = e_real[0,0]
    new_elastic_strain_array[1] = e_real[1,1]
    new_elastic_strain_array[2] = e_real[2,2]

    # Pressure-dependent hyperelasticity
    e_trace = wp.trace(e_real)
    omega = -(e_trace-initial_volumetric_strain)/kappa
    stress_principal = wp.mat33d(
                       -p0*kappa*wp.exp(omega)/(-kappa) + wp.float64(2.)/wp.float64(3.)*lame_mu*(wp.float64(3.)*e_real[0,0]-e_trace), wp.float64(0.), wp.float64(0.),
                       wp.float64(0.), -p0*kappa*wp.exp(omega)/(-kappa) + wp.float64(2.)/wp.float64(3.)*lame_mu*(wp.float64(3.)*e_real[1,1]-e_trace), wp.float64(0.),
                       wp.float64(0.), wp.float64(0.), -p0*kappa*wp.exp(omega)/(-kappa) + wp.float64(2.)/wp.float64(3.)*lame_mu*(wp.float64(3.)*e_real[2,2]-e_trace)
                       )
    saved_stress[0] = stress_principal

    # Get residual
    target_stress = wp.mat33d(
                    target_stress_xx, float64_zero, float64_zero,
                    float64_zero, target_stress_yy, float64_zero,
                    float64_zero, float64_zero, target_stress_zz
                    )
    stress_residual = target_stress - stress_principal

    # Assemble to rhs
    wp.atomic_add(rhs, 0, stress_residual[0,0])
    wp.atomic_add(rhs, 1, stress_residual[1,1])

@wp.kernel
def assemble_Jacobian_coo_format(jacobian_wp: wp.array(dtype=wp.float64),
                                 rows: wp.array(dtype=wp.int32),
                                 cols: wp.array(dtype=wp.int32),
                                 vals: wp.array(dtype=wp.float64),
                                 dof_iter: wp.int32
                                 ):
    column_index = wp.tid()

    rows[6*dof_iter + column_index] = dof_iter
    cols[6*dof_iter + column_index] = column_index
    vals[6*dof_iter + column_index] = jacobian_wp[column_index]

@wp.kernel
def set_new_trial_strain_array_to_elastic_strain(new_trial_strain_array: wp.array(dtype=wp.float64),
                                           new_elastic_strain_array: wp.array(dtype=wp.float64)
                                           ):
    new_trial_strain_array[0] = new_elastic_strain_array[0]
    new_trial_strain_array[1] = new_elastic_strain_array[1]
    new_trial_strain_array[2] = new_elastic_strain_array[2]
    

# ===================== Initialization =====================
# Initialize the elasticc cto
wp.launch(kernel=initialize_stiffness,
          dim=1,
          inputs=[rows, cols, vals, lame_lambda, lame_mu])
wps.bsr_set_from_triplets(stiffness, rows, cols, vals, prune_numerical_zeros=False) # If setting prune_numerical_zeros==True (by default), bsr_matrix will contain NaN. See this: https://github.com/NVIDIA/warp/issues/293

# Set initial stress
wp.launch(kernel=set_initial_stress,
          dim=1,
          inputs=[rhs, target_stress_xx, target_stress_yy, target_stress_zz])

# Initialize pi
wp.launch(kernel=initialize_pi,
          dim=1,
          inputs=[new_pi_array, pi])
wp.launch(kernel=set_old_pi_to_new,
          dim=1,
          inputs=[new_pi_array, old_pi_array])

# Solve for initial strain
preconditioner = wp.optim.linear.preconditioner(stiffness, ptype='diag')
solver_state = wp.optim.linear.bicgstab(A=stiffness, b=rhs, x=strain_increment, tol=tol, M=preconditioner)
# From increment to strains
wp.launch(kernel=from_increment_to_strain_initial,
          dim=6,
          inputs=[strain_increment, new_trial_strain_array])
wp.launch(kernel=from_increment_to_strain_initial,
          dim=6,
          inputs=[strain_increment, new_total_strain_array])
new_total_strain_array_np = new_total_strain_array.numpy()
initial_volumetric_strain = new_total_strain_array_np[0] + new_total_strain_array_np[1] + new_total_strain_array_np[2]

# ===================== Simulation =====================
for step in range(n_steps):
    print('Load step:', step+1)
    # Get trial strain
    wp.launch(kernel=get_trial_strain_triaxial,
              dim=1,
              inputs=[new_trial_strain_array, loading_rate])
    wp.launch(kernel=get_trial_strain_triaxial,
              dim=1,
              inputs=[new_total_strain_array, loading_rate])

    for iter_id in range(n_iter):
        # Reset quantities
        strain_increment.zero_()
        rhs.zero_()
        rows.zero_()
        cols.zero_()
        vals.zero_()

        real_strain_history.zero_()
        pi_history.zero_()
        delta_lambda_history.zero_()

        saved_local_residual.zero_()
        saved_stress.zero_()

        tape = wp.Tape()
        with tape:
            wp.launch(kernel=calculate_stress_residual_NorSand,
                      dim=1,
                      inputs=[kappa, tilde_lambda, M, N, beta, vc0, p0, v0, h, lame_mu, initial_volumetric_strain, old_pi_array, new_pi_array, new_trial_strain_array, new_total_strain_array, new_elastic_strain_array, real_strain_history, pi_history, delta_lambda_history, target_stress_xx, target_stress_yy, target_stress_zz, rhs, saved_local_residual, saved_stress, tol])


        # Assemble the global Jacobian matrix using auto-diff
        for dof_iter in range(6):
            select_index = np.zeros(6)
            select_index[dof_iter] = 1.
            e = wp.array(select_index, dtype=wp.float64)

            tape.backward(grads={rhs: e})
            jacobian_wp = tape.gradients[new_trial_strain_array]

            wp.launch(kernel=assemble_Jacobian_coo_format,
                      dim=6,
                      inputs=[jacobian_wp, rows, cols, vals, dof_iter])

            tape.zero()

        tape.reset()


        # Assemble matrix
        wps.bsr_set_from_triplets(cto, rows, cols, vals, prune_numerical_zeros=False) 
        preconditioner = wp.optim.linear.preconditioner(cto, ptype='diag')
        solver_state = wp.optim.linear.bicgstab(A=cto, b=rhs, x=strain_increment, tol=1e-10, M=preconditioner)
                
        # From increment to solution
        wp.launch(kernel=from_increment_to_strain,
                  dim=6,
                  inputs=[strain_increment, new_trial_strain_array])
        wp.launch(kernel=from_increment_to_strain,
                  dim=6,
                  inputs=[strain_increment, new_total_strain_array])

        
        # # Print global residual
        # print('Residual norm:', np.linalg.norm(rhs.numpy()))

        if np.linalg.norm(rhs.numpy())<tol:
            # # Print local residual at the last step
            # with np.printoptions(threshold=np.inf):
            #     for local_iter in range(n_iter_local):
            #         print('\t', local_iter, np.linalg.norm(saved_local_residual.numpy()[local_iter,:]))
            break

    # Set old pi to new
    wp.launch(kernel=set_old_pi_to_new,
              dim=1,
              inputs=[new_pi_array, old_pi_array])

    # Set new_trial_strain_array to elastic strain
    wp.launch(kernel=set_new_trial_strain_array_to_elastic_strain,
              dim=1,
              inputs=[new_trial_strain_array, new_elastic_strain_array])

    # Post-processing
    # Deviatoric stress-axial strain
    saved_stress_np = saved_stress.numpy()
    Q_invariant = -saved_stress_np[0][2,2]-(-saved_stress_np[0][0,0])
    if step==0:
        print(0., -target_stress_zz-(-target_stress_xx))
    print((step+1)*loading_rate*100., Q_invariant)

    # Volumetric strain-axial strain
    new_total_strain_array_np = new_total_strain_array.numpy()
    volumetric_strain = -1. * (new_total_strain_array_np[0] + new_total_strain_array_np[1] + new_total_strain_array_np[2]) * 100.0 - (-initial_volumetric_strain * 100.0)
    # if step==0:
    #     print(0., 0.)
    # print((step+1)*loading_rate*100.0, volumetric_strain)





