import warp as wp
import numpy as np

import warp.sparse as wps
import warp.optim.linear

# Author: Yidong Zhao (ydzhao94@gmail.com)
# Triaxial test for Drucker-Prager

wp.init()

# ===================== Material properties =====================
youngs_modulus = 25000.0 # kPa
poisson_ratio = 0.3
lame_lambda = youngs_modulus*poisson_ratio / ((1.0+poisson_ratio) * (1.0-2.0*poisson_ratio))
lame_mu = youngs_modulus / (2.0*(1.0+poisson_ratio))

friction_angle = 35.0
dilation_angle = 5.0
cohesion = 0.0
shape_factor = 0.0

# ===================== Loading conditions =====================
final_axial_strain = 0.1
n_steps = 100
loading_rate = final_axial_strain/n_steps
target_stress_xx = -150.0 # Initial radial target stress
target_stress_yy = target_stress_xx # Initial radial target stress
target_stress_zz = -150.0 # Initial axial target stress

tol = 1e-10

n_iter = 10 # Maximum iteration number for global Newton solver
n_iter_local = 10 # Maximum iteration number for local return mapping

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

# Arrays for saving history values during local iteration.
real_strain_history = wp.zeros(shape=n_iter_local+1, dtype=wp.mat33d, requires_grad=True) # This is important to ensure correct gradient calculation
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



# --------------------- DruckerPrager kernels ---------------------
@wp.func
def yield_function_DruckerPrager(P_trial: wp.float64,
                                 Q_trial: wp.float64,
                                 friction_angle: wp.float64,
                                 dilation_angle: wp.float64,
                                 cohesion: wp.float64,
                                 shape_factor: wp.float64) -> wp.float64:
    
    float64_pi = wp.float64(3.14159265358979)
    
    cos_angle = wp.cos(friction_angle*float64_pi/wp.float64(180.))
    sin_angle = wp.sin(friction_angle*float64_pi/wp.float64(180.))

    A = wp.float64(2.)*wp.sqrt(wp.float64(6.)) * cohesion * cos_angle/(wp.float64(3.)-sin_angle)
    B = wp.float64(2.)*wp.sqrt(wp.float64(6.)) * sin_angle/(wp.float64(3.)-sin_angle)

    omega_F = wp.float64(2.)/wp.float64(3.) * Q_trial*Q_trial + shape_factor*shape_factor * A*A

    return wp.sqrt(omega_F) - (A-B*P_trial)

@wp.func
def grad_potential_DruckerPrager(tau0_trial: wp.float64,
                                 tau1_trial: wp.float64,
                                 tau2_trial: wp.float64,
                                 Q_trial: wp.float64,
                                 dilation_angle: wp.float64,
                                 cohesion: wp.float64,
                                 shape_factor: wp.float64) -> wp.vec3d:

    float64_pi = wp.float64(3.14159265358979)
    
    cos_dilation = wp.cos(dilation_angle*float64_pi/wp.float64(180.))
    sin_dilation = wp.sin(dilation_angle*float64_pi/wp.float64(180.))

    A = wp.float64(2.)*wp.sqrt(wp.float64(6.)) * cohesion * cos_dilation/(wp.float64(3.)-sin_dilation)
    B = wp.float64(2.)*wp.sqrt(wp.float64(6.)) * sin_dilation/(wp.float64(3.)-sin_dilation)

    omega_F = wp.float64(2.)/wp.float64(3.) * Q_trial*Q_trial + shape_factor*shape_factor * A*A


    dFdP = B
    dFdQ = wp.float64(2.)/wp.float64(3.) * Q_trial/wp.sqrt(omega_F)

    Q_with_threshold = Q_trial
    sign_Q = wp.float64(1.)
    if Q_with_threshold<wp.float64(0.):
        sign_Q = wp.float64(-1.)
    if wp.abs(Q_with_threshold)<wp.float64(1e-10):
        Q_with_threshold = wp.float64(1e-10) * sign_Q

    gradQ = wp.vec3d(
            wp.float64(0.5)/Q_with_threshold * (wp.float64(2.)*tau0_trial - tau1_trial - tau2_trial),
            wp.float64(0.5)/Q_with_threshold * (wp.float64(2.)*tau1_trial - tau0_trial - tau2_trial),
            wp.float64(0.5)/Q_with_threshold * (wp.float64(2.)*tau2_trial - tau0_trial - tau1_trial)
            )

    grad_g = wp.vec3d(
             dFdP*wp.float64(1.)/wp.float64(3.) + dFdQ*gradQ[0],
             dFdP*wp.float64(1.)/wp.float64(3.) + dFdQ*gradQ[1],
             dFdP*wp.float64(1.)/wp.float64(3.) + dFdQ*gradQ[2],
             )

    return grad_g

@wp.func
def grad_yield_DruckerPrager(tau0_trial: wp.float64, #tau_trial: wp.mat33d,
                             tau1_trial: wp.float64,
                             tau2_trial: wp.float64,
                             Q_trial: wp.float64,
                             friction_angle: wp.float64,
                             cohesion: wp.float64,
                             shape_factor: wp.float64) -> wp.vec3d:
    
    float64_pi = wp.float64(3.14159265358979)

    cos_angle = wp.cos(friction_angle*float64_pi/wp.float64(180.))
    sin_angle = wp.sin(friction_angle*float64_pi/wp.float64(180.))

    A = wp.float64(2.)*wp.sqrt(wp.float64(6.)) * cohesion * cos_angle/(wp.float64(3.)-sin_angle)
    B = wp.float64(2.)*wp.sqrt(wp.float64(6.)) * sin_angle/(wp.float64(3.)-sin_angle)

    omega_F = wp.float64(2.)/wp.float64(3.) * Q_trial*Q_trial + shape_factor*shape_factor * A*A


    dFdP = B
    dFdQ = wp.float64(2.)/wp.float64(3.) * Q_trial/wp.sqrt(omega_F)

    Q_with_threshold = Q_trial
    sign_Q = wp.float64(1.)
    if Q_with_threshold<wp.float64(0.):
        sign_Q = wp.float64(-1.)
    if wp.abs(Q_with_threshold)<wp.float64(1e-10):
        Q_with_threshold = wp.float64(1e-10) * sign_Q

    gradQ = wp.vec3d(
            wp.float64(0.5)/Q_with_threshold * (wp.float64(2.)*tau0_trial - tau1_trial - tau2_trial),
            wp.float64(0.5)/Q_with_threshold * (wp.float64(2.)*tau1_trial - tau0_trial - tau2_trial),
            wp.float64(0.5)/Q_with_threshold * (wp.float64(2.)*tau2_trial - tau0_trial - tau1_trial)
            )

    grad_f = wp.vec3d(
             dFdP*wp.float64(1.)/wp.float64(3.) + dFdQ*gradQ[0],
             dFdP*wp.float64(1.)/wp.float64(3.) + dFdQ*gradQ[1],
             dFdP*wp.float64(1.)/wp.float64(3.) + dFdQ*gradQ[2],
             )

    return grad_f

@wp.func
def grad_yield_epsilon_DruckerPrager(elastic_a: wp.mat33d,
                                     grad_f: wp.vec3d) -> wp.vec3d:

    grad_f_eps = wp.vec3d(
                 grad_f[0]*elastic_a[0,0] + grad_f[1]*elastic_a[1,0] + grad_f[2]*elastic_a[2,0],
                 grad_f[0]*elastic_a[0,1] + grad_f[1]*elastic_a[1,1] + grad_f[2]*elastic_a[2,1],
                 grad_f[0]*elastic_a[0,2] + grad_f[1]*elastic_a[1,2] + grad_f[2]*elastic_a[2,2]
                 )

    return grad_f_eps

@wp.func
def hess_potential_DruckerPrager(S_trial: wp.mat33d,
                                 Q_trial: wp.float64,
                                 dilation_angle: wp.float64,
                                 cohesion: wp.float64,
                                 shape_factor: wp.float64) -> wp.mat33d:

    float64_pi = wp.float64(3.14159265358979)

    trS2 = wp.ddot(S_trial, S_trial) # Check
    chi = wp.sqrt(trS2)
    if wp.abs(chi)<wp.float64(1e-10):
        chi = wp.float64(1e-10)
    const1 = wp.sqrt(wp.float64(3.)/wp.float64(2.)) / chi

    gradQ = wp.vec3d(
            const1*S_trial[0,0], 
            const1*S_trial[1,1],
            const1*S_trial[2,2]
            )

    hessQ = wp.mat33d(
            const1*(wp.float64(1.) - wp.float64(1.)/wp.float64(3.) - S_trial[0,0]*S_trial[0,0]/(chi*chi)), const1*(wp.float64(0.) - wp.float64(1.)/wp.float64(3.) - S_trial[0,0]*S_trial[1,1]/(chi*chi)), const1*(wp.float64(0.) - wp.float64(1.)/wp.float64(3.) - S_trial[0,0]*S_trial[2,2]/(chi*chi)),
            const1*(wp.float64(0.) - wp.float64(1.)/wp.float64(3.) - S_trial[1,1]*S_trial[0,0]/(chi*chi)), const1*(wp.float64(1.) - wp.float64(1.)/wp.float64(3.) - S_trial[1,1]*S_trial[1,1]/(chi*chi)), const1*(wp.float64(0.) - wp.float64(1.)/wp.float64(3.) - S_trial[1,1]*S_trial[2,2]/(chi*chi)),
            const1*(wp.float64(0.) - wp.float64(1.)/wp.float64(3.) - S_trial[2,2]*S_trial[0,0]/(chi*chi)), const1*(wp.float64(0.) - wp.float64(1.)/wp.float64(3.) - S_trial[2,2]*S_trial[1,1]/(chi*chi)), const1*(wp.float64(1.) - wp.float64(1.)/wp.float64(3.) - S_trial[2,2]*S_trial[2,2]/(chi*chi))
            )

    cos_dilation = wp.cos(dilation_angle*float64_pi/wp.float64(180.))
    sin_dilation = wp.sin(dilation_angle*float64_pi/wp.float64(180.))

    A = wp.float64(2.)*wp.sqrt(wp.float64(6.)) * cohesion * cos_dilation/(wp.float64(3.)-sin_dilation)
    B = wp.float64(2.)*wp.sqrt(wp.float64(6.)) * sin_dilation/(wp.float64(3.)-sin_dilation)

    omega_F = wp.float64(2.)/wp.float64(3.) * Q_trial*Q_trial + shape_factor*shape_factor * A*A

    dFdQ = wp.float64(2.)/wp.float64(3.) * Q_trial/wp.sqrt(omega_F)
    d2FdQ2 = wp.float64(2.)/wp.float64(3.)/wp.sqrt(omega_F) - wp.float64(4.)/wp.float64(9.) * Q_trial*Q_trial / (omega_F*wp.sqrt(omega_F))

    gradQ_dyad_gradQ = wp.outer(gradQ, gradQ)

    hess_g = dFdQ*hessQ + d2FdQ2*gradQ_dyad_gradQ

    return hess_g

@wp.func
def hess_potential_epsilon_DruckerPrager(hess_g: wp.mat33d,
                                         elastic_a: wp.mat33d) -> wp.mat33d:
    hess_g_eps_00 = hess_g[0,0]*elastic_a[0,0] + hess_g[0,1]*elastic_a[1,0] + hess_g[0,2]*elastic_a[2,0]
    hess_g_eps_01 = hess_g[0,0]*elastic_a[0,1] + hess_g[0,1]*elastic_a[1,1] + hess_g[0,2]*elastic_a[2,1]
    hess_g_eps_02 = hess_g[0,0]*elastic_a[0,2] + hess_g[0,1]*elastic_a[1,2] + hess_g[0,2]*elastic_a[2,2]

    hess_g_eps_10 = hess_g[1,0]*elastic_a[0,0] + hess_g[1,1]*elastic_a[1,0] + hess_g[1,2]*elastic_a[2,0]
    hess_g_eps_11 = hess_g[1,0]*elastic_a[0,1] + hess_g[1,1]*elastic_a[1,1] + hess_g[1,2]*elastic_a[2,1]
    hess_g_eps_12 = hess_g[1,0]*elastic_a[0,2] + hess_g[1,1]*elastic_a[1,2] + hess_g[1,2]*elastic_a[2,2]

    hess_g_eps_20 = hess_g[2,0]*elastic_a[0,0] + hess_g[2,1]*elastic_a[1,0] + hess_g[2,2]*elastic_a[2,0]
    hess_g_eps_21 = hess_g[2,0]*elastic_a[0,1] + hess_g[2,1]*elastic_a[1,1] + hess_g[2,2]*elastic_a[2,1]
    hess_g_eps_22 = hess_g[2,0]*elastic_a[0,2] + hess_g[2,1]*elastic_a[1,2] + hess_g[2,2]*elastic_a[2,2]


    hess_g_eps = wp.mat33d(
                 hess_g_eps_00, hess_g_eps_01, hess_g_eps_02,
                 hess_g_eps_10, hess_g_eps_11, hess_g_eps_12,
                 hess_g_eps_20, hess_g_eps_21, hess_g_eps_22
                 )

    return hess_g_eps

@wp.func
def return_mapping_DruckerPrager(trial_strain: wp.mat33d,
                                 lame_lambda: wp.float64,
                                 lame_mu: wp.float64,
                                 friction_angle: wp.float64,
                                 dilation_angle: wp.float64,
                                 cohesion: wp.float64,
                                 shape_factor: wp.float64,
                                 tol: wp.float64,
                                 real_strain_history: wp.array(dtype=wp.mat33d),
                                 delta_lambda_history: wp.array(dtype=wp.float64),
                                 ):
    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)

    real_strain = trial_strain

    # Construct elastic_a
    elastic_a_tmp1 = lame_lambda * wp.mat33d(wp.float64(1.), wp.float64(1.), wp.float64(1.),
                                            wp.float64(1.), wp.float64(1.), wp.float64(1.),
                                            wp.float64(1.), wp.float64(1.), wp.float64(1.))
    elastic_a_tmp2 = wp.float64(2.) * lame_mu * wp.identity(n=3, dtype=wp.float64)
    elastic_a = elastic_a_tmp1 + elastic_a_tmp2

    # Calculate trial stress
    eps_v = wp.trace(trial_strain)
    tau_trial = lame_lambda*eps_v*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*trial_strain

    # Get P and Q invariants
    P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
    S_trial = tau_trial - P_trial*wp.identity(n=3, dtype=wp.float64)
    S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
    Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))

    # Check yield
    yield_y = yield_function_DruckerPrager(P_trial, Q_trial, friction_angle, dilation_angle, cohesion, shape_factor)

    if yield_y<wp.float64(1e-10):
        pass
    elif P_trial>=wp.float64(0.): # Return to the tip. Ignored here
        pass
        # TODO
    else: # Plasticity
        # Local newton iteration
        delta_lambda = wp.float64(0.) 
        delta_lambda_history[0] = delta_lambda
        real_strain_history[0] = trial_strain

        for local_iter in range(10):
            grad_g = grad_potential_DruckerPrager(tau_trial[0,0], tau_trial[1,1], tau_trial[2,2], Q_trial, dilation_angle, cohesion, shape_factor)
            grad_f = grad_yield_DruckerPrager(tau_trial[0,0], tau_trial[1,1], tau_trial[2,2], Q_trial, friction_angle, cohesion, shape_factor)
            grad_f_eps = grad_yield_epsilon_DruckerPrager(elastic_a, grad_f)

            hess_g = hess_potential_DruckerPrager(S_trial, Q_trial, dilation_angle, cohesion, shape_factor)
            hess_g_eps = hess_potential_epsilon_DruckerPrager(hess_g, elastic_a)

            yield_y = yield_function_DruckerPrager(P_trial, Q_trial, friction_angle, dilation_angle, cohesion, shape_factor)

            residual = wp.vec4d(
                       real_strain_history[local_iter][0,0] - trial_strain[0,0] + delta_lambda*grad_g[0],
                       real_strain_history[local_iter][1,1] - trial_strain[1,1] + delta_lambda*grad_g[1],
                       real_strain_history[local_iter][2,2] - trial_strain[2,2] + delta_lambda*grad_g[2],
                       yield_y
                       )

            residual_norm = wp.sqrt(residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2] + residual[3]*residual[3])

            if residual_norm<tol: # TODO: this will break up the differentiability. Maybe related to this(?): https://github.com/NVIDIA/warp/issues/140#issuecomment-1682675942
                pass

            # Assemble Jacobian
            jacobian = wp.mat44d(
                       wp.float64(1.) + delta_lambda*hess_g_eps[0,0], delta_lambda*hess_g_eps[0,1], delta_lambda*hess_g_eps[0,2], grad_g[0],
                       delta_lambda*hess_g_eps[1,0], wp.float64(1.) + delta_lambda*hess_g_eps[1,1], delta_lambda*hess_g_eps[1,2], grad_g[1],
                       delta_lambda*hess_g_eps[2,0], delta_lambda*hess_g_eps[2,1], wp.float64(1.) + delta_lambda*hess_g_eps[2,2], grad_g[2],
                       grad_f_eps[0], grad_f_eps[1], grad_f_eps[2], wp.float64(0.)
                       )
            xdelta = wp.inverse(jacobian) @ residual

            delta_strain = wp.mat33d(
                           -xdelta[0], wp.float64(0.), wp.float64(0.),
                           wp.float64(0.), -xdelta[1], wp.float64(0.),
                           wp.float64(0.), wp.float64(0.), -xdelta[2]
                           )
            real_strain_history[local_iter+1] = real_strain_history[local_iter] + delta_strain

            delta_lambda_history[local_iter+1] = delta_lambda_history[local_iter] - xdelta[3]
            delta_lambda = delta_lambda_history[local_iter+1]

            # Update stress
            eps_v = wp.trace(real_strain_history[local_iter+1])
            tau_trial = lame_lambda*eps_v*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*real_strain_history[local_iter+1]

            P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
            S_trial = tau_trial - P_trial*wp.identity(n=3, dtype=wp.float64)
            S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
            Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))

            real_strain = wp.mat33d(
                          real_strain_history[local_iter+1][0,0], wp.float64(0.), wp.float64(0.),
                          wp.float64(0.), real_strain_history[local_iter+1][1,1], wp.float64(0.),
                          wp.float64(0.), wp.float64(0.), real_strain_history[local_iter+1][2,2]
                          )

    return real_strain



@wp.kernel
def calculate_stress_residual_DruckerPrager(lame_lambda: wp.float64,
                                            lame_mu: wp.float64,
                                            friction_angle: wp.float64,
                                            dilation_angle: wp.float64,
                                            cohesion: wp.float64,
                                            shape_factor: wp.float64,
                                            new_trial_strain_array: wp.array(dtype=wp.float64),
                                            new_total_strain_array: wp.array(dtype=wp.float64),
                                            new_elastic_strain_array: wp.array(dtype=wp.float64),
                                            real_strain_history: wp.array(dtype=wp.mat33d),
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

    # Get real strain through return mapping
    e_real = return_mapping_DruckerPrager(trial_strain, lame_lambda, lame_mu, friction_angle, dilation_angle, cohesion, shape_factor, tol, real_strain_history, delta_lambda_history)
    
    new_elastic_strain_array[0] = e_real[0,0]
    new_elastic_strain_array[1] = e_real[1,1]
    new_elastic_strain_array[2] = e_real[2,2]

    # Get new stress
    # Hencky elasticity
    e_trace = wp.trace(e_real)
    stress_principal = lame_lambda*e_trace*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*e_real
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
# Initialize the elastic cto
wp.launch(kernel=initialize_stiffness,
          dim=1,
          inputs=[rows, cols, vals, lame_lambda, lame_mu])
wps.bsr_set_from_triplets(stiffness, rows, cols, vals, prune_numerical_zeros=False) # If setting prune_numerical_zeros==True (by default), bsr_matrix will contain NaN. See this: https://github.com/NVIDIA/warp/issues/293

# Set initial stress
wp.launch(kernel=set_initial_stress,
          dim=1,
          inputs=[rhs, target_stress_xx, target_stress_yy, target_stress_zz])


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
    # print('Load step:', step+1)
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
        delta_lambda_history.zero_()

        saved_local_residual.zero_()
        saved_stress.zero_()

        tape = wp.Tape()
        with tape:
            
            wp.launch(kernel=calculate_stress_residual_DruckerPrager,
                      dim=1,
                      inputs=[lame_lambda, lame_mu, friction_angle, dilation_angle, cohesion, shape_factor, new_trial_strain_array, new_total_strain_array, new_elastic_strain_array, real_strain_history, delta_lambda_history, target_stress_xx, target_stress_yy, target_stress_zz, rhs, saved_local_residual, saved_stress, tol])


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


    # Set new_trial_strain_array to elastic strain
    wp.launch(kernel=set_new_trial_strain_array_to_elastic_strain,
              dim=1,
              inputs=[new_trial_strain_array, new_elastic_strain_array])

    # Post-processing
    # Deviatoric stress-axial strain
    saved_stress_np = saved_stress.numpy()
    P_invariant = -1./3. * (saved_stress_np[0][0,0] + saved_stress_np[0][1,1] + saved_stress_np[0][2,2])
    Q_invariant = -saved_stress_np[0][2,2]-(-saved_stress_np[0][0,0])
    if step==0:
        print(-1./3.*(target_stress_xx+target_stress_yy+target_stress_zz), target_stress_zz-target_stress_xx)
    print(P_invariant, Q_invariant)

    

