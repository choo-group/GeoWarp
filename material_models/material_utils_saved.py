import warp as wp
import numpy as np

vec15d = wp.types.vector(length=15, dtype=wp.float64)
vec15_mat33d = wp.types.vector(length=15, dtype=wp.mat33d)

@wp.func
def get_cauchy_stress_neohookean(particle_F_3d: wp.mat33d, 
                                 lame_lambda: wp.float64,
                                 lame_mu: wp.float64) -> wp.mat33d:
    
    # calculate stress
    particle_F_inv = wp.inverse(particle_F_3d)
    particle_J = wp.determinant(particle_F_3d)
    particle_PK1_stress = lame_mu * (particle_F_3d - wp.transpose(particle_F_inv)) + lame_lambda * wp.log(particle_J) * wp.transpose(particle_F_inv)

    particle_Cauchy_stress = wp.float64(1.)/particle_J * particle_PK1_stress @ wp.transpose(particle_F_3d)

    return particle_Cauchy_stress


@wp.func
def get_cauchy_stress_hencky(particle_F_3d: wp.mat33d, 
                             lame_lambda: wp.float64,
                             lame_mu: wp.float64) -> wp.mat33d:

    particle_J = wp.determinant(particle_F_3d)
    
    U = wp.mat33d()
    V = wp.mat33d()
    sig = wp.vec3d()
    wp.svd3(particle_F_3d, U, sig, V)

    e_trial = wp.matrix(
              wp.log(sig[0]), wp.float64(0.), wp.float64(0.),
              wp.float64(0.), wp.log(sig[1]), wp.float64(0.),
              wp.float64(0.), wp.float64(0.), wp.log(sig[2]),
              shape=(3,3)
        )
    e_real = e_trial

    e_trace = wp.trace(e_real)

    Kirchhoff_principal = lame_lambda*e_trace*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*e_real
    Kirchhoff_stress = U @ (Kirchhoff_principal) @ wp.transpose(U)

    particle_Cauchy_stress = Kirchhoff_stress/particle_J

    return particle_Cauchy_stress


@wp.func
def return_mapping_J2(trial_strain: wp.mat33d, # trial principal strain
                      lame_lambda: wp.float64,
                      lame_mu: wp.float64,
                      kappa: wp.float64) -> wp.mat33d:
    real_strain = trial_strain

    # Calculate trial stress
    eps_v = wp.trace(trial_strain)
    tau_trial = lame_lambda*eps_v*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*trial_strain

    # Get P and S
    P = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
    S_trial = tau_trial - P*wp.identity(n=3, dtype=wp.float64)
    S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))

    if S_trial_norm<=kappa:
        pass
    else: # Yield
        n = S_trial/S_trial_norm

        delta_lambda = (S_trial_norm - kappa)/(wp.float64(2.)*lame_mu)

        real_strain = trial_strain - delta_lambda*n

    return real_strain


@wp.func
def return_mapping_DP_no_iteration(trial_strain: wp.mat33d, # trial principal strain
                                   lame_lambda: wp.float64,
                                   lame_mu: wp.float64,
                                   friction_angle: wp.float64,
                                   dilation_angle: wp.float64,
                                   cohesion: wp.float64,
                                   shape_factor: wp.float64,
                                   tol: wp.float64,
                                   real_strain_array: wp.array(dtype=wp.mat33d)) -> wp.mat33d:
    
    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.14159265358979)

    real_strain = trial_strain

    eps_v = wp.trace(trial_strain)
    tau_trial = lame_lambda*eps_v*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*trial_strain

    # Get P and S
    P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
    S_trial = tau_trial - P_trial*wp.identity(n=3, dtype=wp.float64)
    S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
    Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))

    # return mapping
    friction_angle_coefficient = wp.float64(2.)*wp.sqrt(wp.float64(6.))*wp.sin(friction_angle*float64_pi/wp.float64(180.)) / (wp.float64(3.)-wp.sin(friction_angle*float64_pi/wp.float64(180.)))
    yield_function = wp.sqrt(wp.float64(2.)/wp.float64(3.)) * Q_trial + friction_angle_coefficient * P_trial

    if yield_function<=wp.float64(0.):
        pass
    elif P_trial>=wp.float64(0.):
        pass
    else:
        delta_lambda = (yield_function) / (wp.float64(2.) * lame_mu)

        n = wp.mat33d()
        if S_trial_norm>wp.float64(0.):
            n = S_trial / S_trial_norm
        real_strain = trial_strain - delta_lambda * n

    return real_strain


@wp.func
def return_mapping_DP(trial_strain: wp.mat33d, # trial principal strain
                      lame_lambda: wp.float64,
                      lame_mu: wp.float64,
                      friction_angle: wp.float64,
                      dilation_angle: wp.float64,
                      cohesion: wp.float64,
                      shape_factor: wp.float64,
                      tol: wp.float64,
                      real_strain_array: wp.array(dtype=wp.mat33d)) -> wp.mat33d:

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

    # P_trial_array[0] = P_trial
    # Q_trial_array[0] = Q_trial
    # tau_trial_array[0] = tau_trial

    # Check yield
    yield_y = yield_function_DP(P_trial, Q_trial, friction_angle, dilation_angle, cohesion, shape_factor)

    if yield_y<wp.float64(1e-10):
        pass
    elif P_trial>=wp.float64(0.):
        pass
        # TODO
    else: # Plasticity
        # Local newton iteration

        
        delta_lambda = wp.float64(0.) # Local iter converges weel, but seems this will affect the outer gradient calculation
        # delta_lambda_array[0] = delta_lambda

        real_strain_array[0] = trial_strain


        test = wp.mat33d()
        convergence_flag = wp.float64(0.0)

        for local_iter in range(10):

            grad_g = grad_potential_DP(tau_trial[0,0], tau_trial[1,1], tau_trial[2,2], Q_trial, dilation_angle, cohesion, shape_factor)
            grad_f = grad_yield_DP(tau_trial[0,0], tau_trial[1,1], tau_trial[2,2], Q_trial, friction_angle, cohesion, shape_factor)
            grad_f_eps = grad_yield_epsilon_DP(elastic_a, grad_f)

            hess_g = hess_potential_DP(S_trial, Q_trial, dilation_angle, cohesion, shape_factor)
            hess_g_eps = hess_potential_epsilon_DP(hess_g, elastic_a)

            yield_y = yield_function_DP(P_trial, Q_trial, friction_angle, dilation_angle, cohesion, shape_factor)

            residual = wp.vec4d(
                       real_strain_array[local_iter][0,0] - trial_strain[0,0] + delta_lambda*grad_g[0],
                       real_strain_array[local_iter][1,1] - trial_strain[1,1] + delta_lambda*grad_g[1],
                       real_strain_array[local_iter][2,2] - trial_strain[2,2] + delta_lambda*grad_g[2],
                       yield_y
                       )

            residual_norm = wp.sqrt(residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2] + residual[3]*residual[3])

            if residual_norm<tol: # TODO: this will break up the differentiability. Maybe related to this(?): https://github.com/NVIDIA/warp/issues/140#issuecomment-1682675942
                convergence_flag = wp.float64(1.0)
                # break



            # Assemble Jacobian
            jacobian = wp.mat44d(
                       wp.float64(1.) + delta_lambda*hess_g_eps[0,0], delta_lambda*hess_g_eps[0,1], delta_lambda*hess_g_eps[0,2], grad_g[0],
                       delta_lambda*hess_g_eps[1,0], wp.float64(1.) + delta_lambda*hess_g_eps[1,1], delta_lambda*hess_g_eps[1,2], grad_g[1],
                       delta_lambda*hess_g_eps[2,0], delta_lambda*hess_g_eps[2,1], wp.float64(1.) + delta_lambda*hess_g_eps[2,2], grad_g[2],
                       grad_f_eps[0], grad_f_eps[1], grad_f_eps[2], wp.float64(0.)
                       )
            xdelta = wp.inverse(jacobian) @ residual

        


            

            # NO GLOBAL ARRAY
            # real_strain[0,0] = real_strain[0,0] - xdelta[0] # NOTE: THIS DOES NOT UPDATE THE MATRIX
            # real_strain[1,1] = real_strain[1,1] - xdelta[1]
            # real_strain[2,2] = real_strain[2,2] - xdelta[2]

            delta_strain = wp.mat33d(
                           -xdelta[0], wp.float64(0.), wp.float64(0.),
                           wp.float64(0.), -xdelta[1], wp.float64(0.),
                           wp.float64(0.), wp.float64(0.), -xdelta[2]
                           )
            real_strain_array[local_iter+1] = real_strain_array[local_iter] + delta_strain

            delta_lambda = delta_lambda - xdelta[3]
            # delta_lambda_array[local_iter+1] = delta_lambda_array[local_iter] - xdelta[3]

            tmp = wp.mat33d(wp.float64(1.), wp.float64(0.), wp.float64(0.),
                            wp.float64(0.), wp.float64(0.), wp.float64(0.),
                            wp.float64(0.), wp.float64(0.), wp.float64(0.))
            test = test + tmp



            # Update stress
            eps_v = wp.trace(real_strain_array[local_iter+1])
            tau_trial = lame_lambda*eps_v*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*real_strain_array[local_iter+1]

            # P_trial_array[local_iter+1] = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial_array[local_iter+1])
            # S_trial = tau_trial_array[local_iter+1] - P_trial_array[local_iter+1]*wp.identity(n=3, dtype=wp.float64)
            # S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
            # Q_trial_array[local_iter+1] = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))

            P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
            S_trial = tau_trial - P_trial*wp.identity(n=3, dtype=wp.float64)
            S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
            Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))


            real_strain = wp.mat33d(
                          real_strain_array[local_iter+1][0,0], wp.float64(0.), wp.float64(0.),
                          wp.float64(0.), real_strain_array[local_iter+1][1,1], wp.float64(0.),
                          wp.float64(0.), wp.float64(0.), real_strain_array[local_iter+1][2,2]
                          )

        # print(convergence_flag)

        if convergence_flag<wp.float64(0.5):
            print('not converge!!')



        # # NOTE: WITHOU LOOP, CONVERGENCE IS VERY GOOD (3 ITERATIONS IN TOTAL)
        # grad_g = grad_potential_DP(tau_trial, Q_trial, dilation_angle, cohesion, shape_factor)
        # grad_f = grad_yield_DP(tau_trial, Q_trial, friction_angle, cohesion, shape_factor)
        # grad_f_eps = grad_yield_epsilon_DP(elastic_a, grad_f)

        # hess_g = hess_potential_DP(S_trial, Q_trial, dilation_angle, cohesion, shape_factor)
        # hess_g_eps = hess_potential_epsilon_DP(hess_g, elastic_a) 

        # yield_y_iter = yield_function_DP(P_trial, Q_trial, friction_angle, dilation_angle, cohesion, shape_factor)

        # residual = wp.vec4d(
        #            real_strain[0,0] - trial_strain[0,0] ,
        #            real_strain[1,1] - trial_strain[1,1] ,
        #            real_strain[2,2] - trial_strain[2,2] ,
        #            yield_y_iter
        #            )

        # residual_norm = wp.sqrt(residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2] + residual[3]*residual[3])




        # # Assemble Jacobian
        # jacobian = wp.mat44d(
        #            wp.float64(1.), wp.float64(0.), wp.float64(0.), grad_g[0],
        #            wp.float64(0.), wp.float64(1.), wp.float64(0.), grad_g[1],
        #            wp.float64(0.), wp.float64(0.), wp.float64(1.), grad_g[2],
        #            grad_f_eps[0], grad_f_eps[1], grad_f_eps[2], wp.float64(0.)
        #            )
        # xdelta = wp.inverse(jacobian) @ residual

    

        # delta_strain = wp.mat33d(
        #                -xdelta[0], wp.float64(0.), wp.float64(0.),
        #                wp.float64(0.), -xdelta[1], wp.float64(0.),
        #                wp.float64(0.), wp.float64(0.), -xdelta[2]
        #                )
        # real_strain = trial_strain + delta_strain

        # delta_lambda = - xdelta[3]

        # tmp = wp.mat33d(wp.float64(1.), wp.float64(0.), wp.float64(0.),
        #                 wp.float64(0.), wp.float64(0.), wp.float64(0.),
        #                 wp.float64(0.), wp.float64(0.), wp.float64(0.))
        # test = test + tmp


        # # print(local_iter)

        

        # # print(test)

        # real_strain = real_strain_array[1]


    return real_strain


@wp.kernel
def initialize_pi_for_NorSand(pi_initial: wp.float64,
                              saved_pi: wp.array(dtype=wp.float64)):
    saved_pi[0] = pi_initial

@wp.kernel
def set_old_pi_to_new_NorSand(saved_pi: wp.array(dtype=wp.float64),
                              old_pi: wp.array(dtype=wp.float64)):
    old_pi[0] = saved_pi[0]


@wp.func
def return_mapping_NorSand(trial_strain: wp.mat33d, # trial principal strain
                           total_strain: wp.mat33d,
                           lame_lambda: wp.float64,
                           lame_mu: wp.float64,
                           M: wp.float64,
                           N: wp.float64,
                           saved_pi: wp.array(dtype=wp.float64),
                           old_pi_array: wp.array(dtype=wp.float64),
                           tilde_lambda: wp.float64,
                           beta: wp.float64,
                           v_c0: wp.float64,
                           v_0: wp.float64,
                           h: wp.float64,
                           tol: wp.float64,
                           real_strain_array: wp.array(dtype=wp.mat33d),
                           pi_array: wp.array(dtype=wp.float64),
                           delta_lambda_array: wp.array(dtype=wp.float64),
                           saved_local_residual: wp.array(dtype=wp.float64),
                           saved_residual: wp.array(dtype=wp.vec4d),
                           saved_H: wp.array(dtype=wp.float64),
                           Ir: wp.float64,
                           poisson_ratio: wp.float64,
                           saved_p: wp.float64) -> wp.mat33d:

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    # K = lame_lambda + wp.float64(2.0)/wp.float64(3.0)*lame_mu

    # pressure-dependent elasticity
    G_elasticity = Ir * (-saved_p)
    K_elasticity = (wp.float64(2.0)*(wp.float64(1.0)+poisson_ratio))/(wp.float64(3.0)*(wp.float64(1.0)-wp.float64(2.0)*poisson_ratio)) * G_elasticity
    lame_lambda_new = K_elasticity - wp.float64(2.0)/wp.float64(3.0) * G_elasticity
    lame_mu_new = G_elasticity


    alpha_bar = wp.float64(-3.5)/beta
    xi = wp.float64(0.1) # parameter for the plastic potential function cap. See Eq. (2.76) of Borja, Andrade (2006) for more details
    old_pi = old_pi_array[0]
    pi_array[0] = old_pi

    # if saved_pi[0]<wp.float64(-223):
    #     h = wp.float64(0.0)

    # saved_H[0] = old_pi




    real_strain = trial_strain

    # Construct elastic_a
    elastic_a_tmp1 = lame_lambda_new * wp.mat33d(wp.float64(1.), wp.float64(1.), wp.float64(1.),
                                            wp.float64(1.), wp.float64(1.), wp.float64(1.),
                                            wp.float64(1.), wp.float64(1.), wp.float64(1.))
    elastic_a_tmp2 = wp.float64(2.) * lame_mu_new * wp.identity(n=3, dtype=wp.float64)
    elastic_a = elastic_a_tmp1 + elastic_a_tmp2

    # Calculate trial stress
    eps_v = wp.trace(trial_strain)
    eps_s = wp.sqrt(wp.float64(2.0)/wp.float64(9.0) * ((trial_strain[0,0]-trial_strain[1,1])*(trial_strain[0,0]-trial_strain[1,1]) + (trial_strain[1,1]-trial_strain[2,2])*(trial_strain[1,1]-trial_strain[2,2]) + (trial_strain[0,0]-trial_strain[2,2])*(trial_strain[0,0]-trial_strain[2,2])))
    tau_trial = lame_lambda_new*eps_v*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu_new*trial_strain

    # Get deviatoric direction
    n_e_trial_deviatoric = wp.vec3d(
                           trial_strain[0,0] - wp.float64(1.0)/wp.float64(3.0)*eps_v,
                           trial_strain[1,1] - wp.float64(1.0)/wp.float64(3.0)*eps_v,
                           trial_strain[2,2] - wp.float64(1.0)/wp.float64(3.0)*eps_v,
                           )
    n_e_trial_deviatoric = wp.normalize(n_e_trial_deviatoric)

    # Get P and Q invariants
    P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
    S_trial = tau_trial - P_trial*wp.identity(n=3, dtype=wp.float64)
    S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
    Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))

    # Get new volume
    eps_v_total = wp.trace(total_strain)
    new_J = wp.exp(eps_v_total)
    new_v = new_J * v_0

    # Check whether eps_v > 0
    if eps_v > tol:
        real_strain = wp.mat33d()
        # saved_H[0] = old_pi
    else:
        # Check yield
        yield_y = yield_function_NorSand(P_trial, Q_trial, M, N, old_pi)

        if yield_y<wp.float64(1e-10):
            saved_H[0] = old_pi
            pass # elasticity
        else:
            # plasticity

            # Local newton iteration
            delta_lambda = wp.float64(0.)
            real_strain_array[0] = trial_strain

            convergence_flag = wp.float64(0.0)

            eta = eta_function_NorSand(M, N, P_trial, old_pi)
            if False: #eta < xi * M:
                pass #todo
            else:
                # for local_iter in range(15):

                #     eps_real_v = wp.trace(real_strain_array[local_iter])
                #     eps_real_s = wp.sqrt(wp.float64(2.0)/wp.float64(9.0) * ((real_strain_array[local_iter][0,0]-real_strain_array[local_iter][1,1])*(real_strain_array[local_iter][0,0]-real_strain_array[local_iter][1,1]) + (real_strain_array[local_iter][1,1]-real_strain_array[local_iter][2,2])*(real_strain_array[local_iter][1,1]-real_strain_array[local_iter][2,2]) + (real_strain_array[local_iter][0,0]-real_strain_array[local_iter][2,2])*(real_strain_array[local_iter][0,0]-real_strain_array[local_iter][2,2])))

                #     # sub iteration for pi
                #     psi_i = wp.float64(0.0)
                #     pi_star = wp.float64(0.0)
                #     for local_iter_p in range(15):
                #         psi_i = new_v - v_c0 + tilde_lambda * wp.log(-pi_array[local_iter*15 + local_iter_p])
                #         pi_star = wp.float64(0.0)
                #         if wp.abs(N)<tol:
                #             pi_star = P_trial * wp.exp(alpha_bar * psi_i / M)
                #         else:
                #             pi_star = P_trial * wp.pow(wp.float64(1.0) - alpha_bar*psi_i*N/M, (N-wp.float64(1.0)/N))

                #         local_residual = pi_array[local_iter*15 + local_iter_p] - old_pi + h*(pi_array[local_iter*15 + local_iter_p] - pi_star) * delta_lambda

                #         if wp.abs(local_residual)<tol:
                #             pi_array[local_iter*15 + local_iter_p + 1] = pi_array[local_iter*15 + local_iter_p]
                #         else:
                #             local_jacobian = wp.float64(1.0) + h*delta_lambda*(wp.float64(1.0) - (tilde_lambda*alpha_bar*(wp.float64(1.0)-N))/(M-alpha_bar*psi_i*N)*(pi_star/pi_array[local_iter*15 + local_iter_p]))
                #             pi_array[local_iter*15 + local_iter_p + 1] = pi_array[local_iter*15 + local_iter_p] - local_residual/local_jacobian

                #     new_pi = pi_array[local_iter*15 + 15]
                    
                #     # Jacobian, see section 2.6 of Borja, Andrade for details
                #     c = wp.float64(1.0) + h*delta_lambda*(wp.float64(1.0) - (tilde_lambda*alpha_bar*(wp.float64(1.0)-N))/(M-alpha_bar*psi_i*N) * (pi_star/new_pi))
                #     D = wp.mat33d(
                #         K, wp.float64(0.0), wp.float64(0.0),
                #         wp.float64(0.0), wp.float64(3.0)*lame_mu, wp.float64(0.0),
                #         wp.float64(1.0)/c * h * delta_lambda * (pi_star/P_trial) * K, wp.float64(0.0), wp.float64(-1.0)/c * h * (new_pi - pi_star)
                #         )

                #     H = wp.vec3d(
                #         wp.float64(-1.0)/(wp.float64(1.0)-N) * (M/P_trial) * wp.pow(P_trial/new_pi, N/(wp.float64(1.0)-N)),
                #         wp.float64(0.0),
                #         wp.float64(1.0)/(wp.float64(1.0)-N) * (M/P_trial) * wp.pow(P_trial/new_pi, wp.float64(1.0)/(wp.float64(1.0)-N))
                #         )

                #     G = H @ D

                #     dFdP = wp.float64(0.0)
                #     dFdQ = wp.float64(1.0)
                #     if wp.abs(N)<1e-8:
                #         dFdP = M * wp.log(new_pi/P_trial)
                #     else:
                #         dFdP = M/N * (wp.float64(1.0) - wp.pow(P_trial/new_pi, N/(wp.float64(1.0)-N)))

                #     dFdPi = M * wp.pow(P_trial/new_pi, wp.float64(1.0)/(wp.float64(1.0)-N))

                #     yield_y = yield_function_NorSand(P_trial, Q_trial, M, N, new_pi)

                #     residual = wp.vec3d(
                #                eps_real_v - eps_v + delta_lambda * beta * dFdP,
                #                eps_real_s - eps_s + delta_lambda * dFdQ,
                #                yield_y
                #                )

                #     jacobian = wp.mat33d(
                #                wp.float64(1.0)+delta_lambda*beta*G[0], delta_lambda*beta*G[1], beta*(dFdP+delta_lambda*G[2]),
                #                wp.float64(0.0), wp.float64(1.0), dFdQ,
                #                D[0,0]*dFdP+D[1,0]*dFdQ+D[2,0]*dFdPi, D[0,1]*dFdP+D[1,1]*dFdQ+D[2,1]*dFdPi, D[2,2]*dFdPi
                #                )
                #     xdelta = wp.inverse(jacobian) @ residual

                #     new_eps_real_v = eps_real_v - xdelta[0]
                #     new_eps_real_s = eps_real_s - xdelta[1]
                #     delta_lambda -= xdelta[2]


                #     # Update strain
                #     real_strain_array[local_iter+1] = wp.mat33d(
                #                                       new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[0]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0), wp.float64(0.0),
                #                                       wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[1]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0),
                #                                       wp.float64(0.0), wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[2]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0))
                #                                       )

                #     # Update stress
                #     eps_v_tmp = wp.trace(real_strain_array[local_iter+1])
                #     tau_tmp = lame_lambda*eps_v_tmp*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*real_strain_array[local_iter+1]

                #     P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_tmp)
                #     S_trial = tau_tmp - P_trial*wp.identity(n=3, dtype=wp.float64)
                #     S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
                #     Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))


                #     real_strain = wp.mat33d(
                #                   real_strain_array[local_iter+1][0,0], wp.float64(0.), wp.float64(0.),
                #                   wp.float64(0.), real_strain_array[local_iter+1][1,1], wp.float64(0.),
                #                   wp.float64(0.), wp.float64(0.), real_strain_array[local_iter+1][2,2]
                #                   )



                # for iter_i in range(225):

                #     local_iter = iter_i/15
                #     local_iter_p = iter_i%15

                    
                #     # sub iteration for pi
                #     psi_i = wp.float64(0.0)
                #     pi_star = wp.float64(0.0)
                    
                #     psi_i = new_v - v_c0 + tilde_lambda * wp.log(-pi_array[local_iter*15 + local_iter_p])
                #     pi_star = wp.float64(0.0)
                #     if wp.abs(N)<tol:
                #         pi_star = P_trial * wp.exp(alpha_bar * psi_i / M)
                #     else:
                #         pi_star = P_trial * wp.pow(wp.float64(1.0) - alpha_bar*psi_i*N/M, ((N-wp.float64(1.0))/N))

                #     local_residual = pi_array[local_iter*15 + local_iter_p] - old_pi + h*(pi_array[local_iter*15 + local_iter_p] - pi_star) * delta_lambda_array[local_iter]

                #     if wp.abs(local_residual)<tol:
                #         pi_array[local_iter*15 + local_iter_p + 1] = pi_array[local_iter*15 + local_iter_p]
                #     else:
                #         local_jacobian = wp.float64(1.0) + h*delta_lambda_array[local_iter]*(wp.float64(1.0) - (tilde_lambda*alpha_bar*(wp.float64(1.0)-N))/(M-alpha_bar*psi_i*N)*(pi_star/pi_array[local_iter*15 + local_iter_p]))
                #         pi_array[local_iter*15 + local_iter_p + 1] = pi_array[local_iter*15 + local_iter_p] - local_residual/local_jacobian


                #     if local_iter_p==14:
                #         eps_real_v = wp.trace(real_strain_array[local_iter])
                #         eps_real_s = wp.sqrt(wp.float64(2.0)/wp.float64(9.0) * ((real_strain_array[local_iter][0,0]-real_strain_array[local_iter][1,1])*(real_strain_array[local_iter][0,0]-real_strain_array[local_iter][1,1]) + (real_strain_array[local_iter][1,1]-real_strain_array[local_iter][2,2])*(real_strain_array[local_iter][1,1]-real_strain_array[local_iter][2,2]) + (real_strain_array[local_iter][0,0]-real_strain_array[local_iter][2,2])*(real_strain_array[local_iter][0,0]-real_strain_array[local_iter][2,2])))

                #         new_pi = pi_array[local_iter*15 + 15]
                        
                #         # Jacobian, see section 2.6 of Borja, Andrade for details
                #         c = wp.float64(1.0) + h*delta_lambda_array[local_iter]*(wp.float64(1.0) - (tilde_lambda*alpha_bar*(wp.float64(1.0)-N))/(M-alpha_bar*psi_i*N) * (pi_star/new_pi))
                #         D = wp.mat33d(
                #             K, wp.float64(0.0), wp.float64(0.0),
                #             wp.float64(0.0), wp.float64(3.0)*lame_mu, wp.float64(0.0),
                #             wp.float64(1.0)/c * h * delta_lambda_array[local_iter] * (pi_star/P_trial) * K, wp.float64(0.0), wp.float64(-1.0)/c * h * (new_pi - pi_star)
                #             )

                #         H = wp.vec3d(
                #             wp.float64(-1.0)/(wp.float64(1.0)-N) * (M/P_trial) * wp.pow(P_trial/new_pi, N/(wp.float64(1.0)-N)),
                #             wp.float64(0.0),
                #             wp.float64(1.0)/(wp.float64(1.0)-N) * (M/P_trial) * wp.pow(P_trial/new_pi, wp.float64(1.0)/(wp.float64(1.0)-N))
                #             )

                #         G = H @ D

                #         dFdP = wp.float64(0.0)
                #         dFdQ = wp.float64(1.0)
                #         if wp.abs(N)<1e-8:
                #             dFdP = M * wp.log(new_pi/P_trial)
                #         else:
                #             dFdP = M/N * (wp.float64(1.0) - wp.pow(P_trial/new_pi, N/(wp.float64(1.0)-N)))

                #         dFdPi = M * wp.pow(P_trial/new_pi, wp.float64(1.0)/(wp.float64(1.0)-N))

                #         yield_y = yield_function_NorSand(P_trial, Q_trial, M, N, new_pi)

                #         residual = wp.vec3d(
                #                    eps_real_v - eps_v + delta_lambda_array[local_iter] * beta * dFdP,
                #                    eps_real_s - eps_s + delta_lambda_array[local_iter] * dFdQ,
                #                    yield_y
                #                    )

                #         jacobian = wp.mat33d(
                #                    wp.float64(1.0)+delta_lambda_array[local_iter]*beta*G[0], delta_lambda_array[local_iter]*beta*G[1], beta*(dFdP+delta_lambda_array[local_iter]*G[2]),
                #                    wp.float64(0.0), wp.float64(1.0), dFdQ,
                #                    D[0,0]*dFdP+D[1,0]*dFdQ+D[2,0]*dFdPi, D[0,1]*dFdP+D[1,1]*dFdQ+D[2,1]*dFdPi, D[2,2]*dFdPi
                #                    )
                #         xdelta = wp.inverse(jacobian) @ residual

                #         new_eps_real_v = eps_real_v - xdelta[0]
                #         new_eps_real_s = eps_real_s - xdelta[1]
                #         delta_lambda_array[local_iter+1] = delta_lambda_array[local_iter] - xdelta[2]


                #         # Update strain
                #         real_strain_array[local_iter+1] = wp.mat33d(
                #                                           new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[0]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0), wp.float64(0.0),
                #                                           wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[1]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0),
                #                                           wp.float64(0.0), wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[2]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0))
                #                                           )

                #         # Update stress
                #         eps_v_tmp = wp.trace(real_strain_array[local_iter+1])
                #         tau_tmp = lame_lambda*eps_v_tmp*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*real_strain_array[local_iter+1]

                #         P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_tmp)
                #         S_trial = tau_tmp - P_trial*wp.identity(n=3, dtype=wp.float64)
                #         S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
                #         Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))


                #         real_strain = wp.mat33d(
                #                       real_strain_array[local_iter+1][0,0], wp.float64(0.), wp.float64(0.),
                #                       wp.float64(0.), real_strain_array[local_iter+1][1,1], wp.float64(0.),
                #                       wp.float64(0.), wp.float64(0.), real_strain_array[local_iter+1][2,2]
                #                       )







                # # using delta_lambda
                # for iter_i in range(225):

                #     local_iter = iter_i/15
                #     local_iter_p = iter_i%15

                    
                #     # sub iteration for pi
                #     psi_i = wp.float64(0.0)
                #     pi_star = wp.float64(0.0)
                    
                #     psi_i = new_v - v_c0 + tilde_lambda * wp.log(-pi_array[local_iter*15 + local_iter_p])
                #     pi_star = wp.float64(0.0)
                #     if wp.abs(N)<tol:
                #         pi_star = P_trial * wp.exp(alpha_bar * psi_i / M)
                #     else:
                #         pi_star = P_trial * wp.pow(wp.float64(1.0) - alpha_bar*psi_i*N/M, ((N-wp.float64(1.0))/N))

                #     local_residual = pi_array[local_iter*15 + local_iter_p] - old_pi + h*(pi_array[local_iter*15 + local_iter_p] - pi_star) * delta_lambda

                #     if wp.abs(local_residual)<tol:
                #         pi_array[local_iter*15 + local_iter_p + 1] = pi_array[local_iter*15 + local_iter_p]
                #     else:
                #         local_jacobian = wp.float64(1.0) + h*delta_lambda*(wp.float64(1.0) - (tilde_lambda*alpha_bar*(wp.float64(1.0)-N))/(M-alpha_bar*psi_i*N)*(pi_star/pi_array[local_iter*15 + local_iter_p]))
                #         pi_array[local_iter*15 + local_iter_p + 1] = pi_array[local_iter*15 + local_iter_p] - local_residual/local_jacobian


                #     if local_iter_p==14:
                #         eps_real_v = wp.trace(real_strain_array[local_iter])
                #         eps_real_s = wp.sqrt(wp.float64(2.0)/wp.float64(9.0) * ((real_strain_array[local_iter][0,0]-real_strain_array[local_iter][1,1])*(real_strain_array[local_iter][0,0]-real_strain_array[local_iter][1,1]) + (real_strain_array[local_iter][1,1]-real_strain_array[local_iter][2,2])*(real_strain_array[local_iter][1,1]-real_strain_array[local_iter][2,2]) + (real_strain_array[local_iter][0,0]-real_strain_array[local_iter][2,2])*(real_strain_array[local_iter][0,0]-real_strain_array[local_iter][2,2])))

                #         new_pi = pi_array[local_iter*15 + 15]
                        
                #         # Jacobian, see section 2.6 of Borja, Andrade for details
                #         c = wp.float64(1.0) + h*delta_lambda*(wp.float64(1.0) - (tilde_lambda*alpha_bar*(wp.float64(1.0)-N))/(M-alpha_bar*psi_i*N) * (pi_star/new_pi))
                #         D = wp.mat33d(
                #             K, wp.float64(0.0), wp.float64(0.0),
                #             wp.float64(0.0), wp.float64(3.0)*lame_mu, wp.float64(0.0),
                #             wp.float64(1.0)/c * h * delta_lambda * (pi_star/P_trial) * K, wp.float64(0.0), wp.float64(-1.0)/c * h * (new_pi - pi_star)
                #             )

                #         H = wp.vec3d(
                #             wp.float64(-1.0)/(wp.float64(1.0)-N) * (M/P_trial) * wp.pow(P_trial/new_pi, N/(wp.float64(1.0)-N)),
                #             wp.float64(0.0),
                #             wp.float64(1.0)/(wp.float64(1.0)-N) * (M/P_trial) * wp.pow(P_trial/new_pi, wp.float64(1.0)/(wp.float64(1.0)-N))
                #             )

                #         G = H @ D

                #         dFdP = wp.float64(0.0)
                #         dFdQ = wp.float64(1.0)
                #         if wp.abs(N)<1e-8:
                #             dFdP = M * wp.log(new_pi/P_trial)
                #         else:
                #             dFdP = M/N * (wp.float64(1.0) - wp.pow(P_trial/new_pi, N/(wp.float64(1.0)-N)))

                #         dFdPi = M * wp.pow(P_trial/new_pi, wp.float64(1.0)/(wp.float64(1.0)-N))

                #         yield_y = yield_function_NorSand(P_trial, Q_trial, M, N, new_pi)

                #         residual = wp.vec3d(
                #                    eps_real_v - eps_v + delta_lambda * beta * dFdP,
                #                    eps_real_s - eps_s + delta_lambda * dFdQ,
                #                    yield_y
                #                    )

                #         jacobian = wp.mat33d(
                #                    wp.float64(1.0)+delta_lambda*beta*G[0], delta_lambda*beta*G[1], beta*(dFdP+delta_lambda*G[2]),
                #                    wp.float64(0.0), wp.float64(1.0), dFdQ,
                #                    D[0,0]*dFdP+D[1,0]*dFdQ+D[2,0]*dFdPi, D[0,1]*dFdP+D[1,1]*dFdQ+D[2,1]*dFdPi, D[2,2]*dFdPi
                #                    )
                #         xdelta = wp.inverse(jacobian) @ residual

                #         new_eps_real_v = eps_real_v
                #         new_eps_real_s = eps_real_s
                #         residual_norm = wp.sqrt(wp.dot(residual, residual))
                #         if residual_norm>1e-8:
                #             new_eps_real_v = eps_real_v - xdelta[0]
                #             new_eps_real_s = eps_real_s - xdelta[1]
                #             delta_lambda = delta_lambda - xdelta[2]


                #         # Update strain
                #         real_strain_array[local_iter+1] = wp.mat33d(
                #                                           new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[0]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0), wp.float64(0.0),
                #                                           wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[1]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0),
                #                                           wp.float64(0.0), wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[2]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0))
                #                                           )

                #         # Update stress
                #         eps_v_tmp = wp.trace(real_strain_array[local_iter+1])
                #         tau_tmp = lame_lambda*eps_v_tmp*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*real_strain_array[local_iter+1]

                #         P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_tmp)
                #         S_trial = tau_tmp - P_trial*wp.identity(n=3, dtype=wp.float64)
                #         S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
                #         Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))


                #         real_strain = wp.mat33d(
                #                       real_strain_array[local_iter+1][0,0], wp.float64(0.), wp.float64(0.),
                #                       wp.float64(0.), real_strain_array[local_iter+1][1,1], wp.float64(0.),
                #                       wp.float64(0.), wp.float64(0.), real_strain_array[local_iter+1][2,2]
                #                       )


                #         if iter_i==224:
                #             saved_local_residual[0] = local_residual
                #             saved_residual[0] = residual





                # saved_H[0] = old_pi

                # Treat pi as an independent variable
                for local_iter in range(15):
                    eps_real_v = wp.trace(real_strain_array[local_iter])
                    eps_real_s = wp.sqrt(wp.float64(2.0)/wp.float64(9.0) * ((real_strain_array[local_iter][0,0]-real_strain_array[local_iter][1,1])*(real_strain_array[local_iter][0,0]-real_strain_array[local_iter][1,1]) + (real_strain_array[local_iter][1,1]-real_strain_array[local_iter][2,2])*(real_strain_array[local_iter][1,1]-real_strain_array[local_iter][2,2]) + (real_strain_array[local_iter][0,0]-real_strain_array[local_iter][2,2])*(real_strain_array[local_iter][0,0]-real_strain_array[local_iter][2,2])))
                    new_pi = pi_array[local_iter]

                    dFdP = wp.float64(0.0)
                    dFdQ = wp.float64(1.0)
                    if wp.abs(N)<1e-8:
                        dFdP = M * wp.log(new_pi/P_trial)
                    else:
                        dFdP = M/N * (wp.float64(1.0) - wp.pow(P_trial/new_pi, N/(wp.float64(1.0)-N)))


                    # residual
                    psi_i = new_v - v_c0 + tilde_lambda * wp.log(-new_pi)
                    pi_star = wp.float64(0.0)
                    if wp.abs(N)<tol:
                        pi_star = P_trial * wp.exp(alpha_bar * psi_i / M)
                    else:
                        pi_star = P_trial * wp.pow(wp.float64(1.0) - alpha_bar*psi_i*N/M, ((N-wp.float64(1.0))/N))

                    yield_y = yield_function_NorSand(P_trial, Q_trial, M, N, new_pi)
                    H = new_pi - old_pi + h*(new_pi-pi_star)*delta_lambda

                    residual = wp.vec4d(
                               eps_real_v - eps_v + delta_lambda * beta * dFdP,
                               eps_real_s - eps_s + delta_lambda * dFdQ,
                               yield_y,
                               H
                               )

                    # local jacobian
                    dEtadP = dEtadP_function_NorSand(M, N, P_trial, new_pi)
                    dEtadPi = dEtadPi_function_NorSand(M, N, P_trial, new_pi)
                    d2F_dPdepsv = dEtadP * K_elasticity
                    d2F_dPdPi = dEtadPi

                    eta = eta_function_NorSand(M, N, P_trial, new_pi)
                    dFdepsv = (dEtadP*P_trial + eta) * K_elasticity
                    dFdepss = wp.float64(3.0) * lame_mu_new
                    dFdPi = P_trial * dEtadPi

                    dPistar_dP = dPistar_dP_function_NorSand(M, N, alpha_bar, psi_i)
                    dPsii_dPi = tilde_lambda/new_pi
                    dPistar_dPi = dPistar_dPi_function_NorSand(M, N, alpha_bar, P_trial, psi_i, dPsii_dPi)
                    dHdepsv = -h*delta_lambda*dPistar_dP*K_elasticity
                    dHdPi = wp.float64(1.0) + h*delta_lambda*(wp.float64(1.0) - dPistar_dPi)


                    jacobian = wp.mat44d(
                               wp.float64(1.0)+delta_lambda*beta*d2F_dPdepsv, wp.float64(0.0), beta*dFdP, delta_lambda*beta*d2F_dPdPi,
                               wp.float64(0.0), wp.float64(1.0), dFdQ, wp.float64(0.0),
                               dFdepsv, dFdepss, wp.float64(0.0), dFdPi,
                               dHdepsv, wp.float64(0.0), h*(new_pi-pi_star), dHdPi
                               )

                    xdelta = wp.inverse(jacobian) @ residual

                    new_eps_real_v = eps_real_v
                    new_eps_real_s = eps_real_s
                    residual_norm = wp.sqrt(wp.dot(residual, residual))

                    new_eps_real_v = eps_real_v - xdelta[0]
                    new_eps_real_s = eps_real_s - xdelta[1]
                    delta_lambda = delta_lambda - xdelta[2]
                    pi_array[local_iter+1] = pi_array[local_iter] - xdelta[3]

                    # Update strain
                    real_strain_array[local_iter+1] = wp.mat33d(
                                                      new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[0]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0), wp.float64(0.0),
                                                      wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[1]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0),
                                                      wp.float64(0.0), wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[2]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0))
                                                      )

                    # Update stress
                    eps_v_tmp = wp.trace(real_strain_array[local_iter+1])
                    tau_tmp = lame_lambda_new*eps_v_tmp*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu_new*real_strain_array[local_iter+1]

                    P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_tmp)
                    S_trial = tau_tmp - P_trial*wp.identity(n=3, dtype=wp.float64)
                    S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
                    Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))


                    real_strain = wp.mat33d(
                              real_strain_array[local_iter+1][0,0], wp.float64(0.), wp.float64(0.),
                              wp.float64(0.), real_strain_array[local_iter+1][1,1], wp.float64(0.),
                              wp.float64(0.), wp.float64(0.), real_strain_array[local_iter+1][2,2]
                              )


                    saved_residual[local_iter] = residual

                    # Hardening modulus
                    # saved_H[0] = old_pi #new_pi - pi_star==0 #M*h*wp.pow(P_trial/new_pi, wp.float64(1.0)/(wp.float64(1.0)-N))*(new_pi - pi_star)==0






                # # no iteration on pi
                # new_pi = saved_pi[0]
                # for local_iter in range(15):

                    
                #     # sub iteration for pi
                #     psi_i = wp.float64(0.0)
                #     pi_star = wp.float64(0.0)
                    
                #     psi_i = new_v - v_c0 + tilde_lambda * wp.log(-new_pi)
                #     pi_star = wp.float64(0.0)
                #     if wp.abs(N)<tol:
                #         pi_star = P_trial * wp.exp(alpha_bar * psi_i / M)
                #     else:
                #         pi_star = P_trial * wp.pow(wp.float64(1.0) - alpha_bar*psi_i*N/M, (N-wp.float64(1.0)/N))


                #     eps_real_v = wp.trace(real_strain_array[local_iter])
                #     eps_real_s = wp.sqrt(wp.float64(2.0)/wp.float64(9.0) * ((real_strain_array[local_iter][0,0]-real_strain_array[local_iter][1,1])*(real_strain_array[local_iter][0,0]-real_strain_array[local_iter][1,1]) + (real_strain_array[local_iter][1,1]-real_strain_array[local_iter][2,2])*(real_strain_array[local_iter][1,1]-real_strain_array[local_iter][2,2]) + (real_strain_array[local_iter][0,0]-real_strain_array[local_iter][2,2])*(real_strain_array[local_iter][0,0]-real_strain_array[local_iter][2,2])))

                    
                #     # Jacobian, see section 2.6 of Borja, Andrade for details
                #     c = wp.float64(1.0) + h*delta_lambda*(wp.float64(1.0) - (tilde_lambda*alpha_bar*(wp.float64(1.0)-N))/(M-alpha_bar*psi_i*N) * (pi_star/new_pi))
                #     D = wp.mat33d(
                #         K, wp.float64(0.0), wp.float64(0.0),
                #         wp.float64(0.0), wp.float64(3.0)*lame_mu, wp.float64(0.0),
                #         wp.float64(1.0)/c * h * delta_lambda * (pi_star/P_trial) * K, wp.float64(0.0), wp.float64(-1.0)/c * h * (new_pi - pi_star)
                #         )

                #     H = wp.vec3d(
                #         wp.float64(-1.0)/(wp.float64(1.0)-N) * (M/P_trial) * wp.pow(P_trial/new_pi, N/(wp.float64(1.0)-N)),
                #         wp.float64(0.0),
                #         wp.float64(1.0)/(wp.float64(1.0)-N) * (M/P_trial) * wp.pow(P_trial/new_pi, wp.float64(1.0)/(wp.float64(1.0)-N))
                #         )

                #     G = H @ D

                #     dFdP = wp.float64(0.0)
                #     dFdQ = wp.float64(1.0)
                #     if wp.abs(N)<1e-8:
                #         dFdP = M * wp.log(new_pi/P_trial)
                #     else:
                #         dFdP = M/N * (wp.float64(1.0) - wp.pow(P_trial/new_pi, N/(wp.float64(1.0)-N)))

                #     dFdPi = M * wp.pow(P_trial/new_pi, wp.float64(1.0)/(wp.float64(1.0)-N))

                #     yield_y = yield_function_NorSand(P_trial, Q_trial, M, N, new_pi)

                #     residual = wp.vec3d(
                #                eps_real_v - eps_v + delta_lambda * beta * dFdP,
                #                eps_real_s - eps_s + delta_lambda * dFdQ,
                #                yield_y
                #                )

                #     jacobian = wp.mat33d(
                #                wp.float64(1.0)+delta_lambda*beta*G[0], delta_lambda*beta*G[1], beta*(dFdP+delta_lambda*G[2]),
                #                wp.float64(0.0), wp.float64(1.0), dFdQ,
                #                D[0,0]*dFdP+D[1,0]*dFdQ+D[2,0]*dFdPi, D[0,1]*dFdP+D[1,1]*dFdQ+D[2,1]*dFdPi, D[2,2]*dFdPi
                #                )
                #     xdelta = wp.inverse(jacobian) @ residual

                #     new_eps_real_v = eps_real_v - xdelta[0]
                #     new_eps_real_s = eps_real_s - xdelta[1]
                #     delta_lambda -= xdelta[2]


                #     # Update strain
                #     real_strain_array[local_iter+1] = wp.mat33d(
                #                                       new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[0]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0), wp.float64(0.0),
                #                                       wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[1]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0)), wp.float64(0.0),
                #                                       wp.float64(0.0), wp.float64(0.0), new_eps_real_v/wp.float64(3.0)+new_eps_real_s*n_e_trial_deviatoric[2]*wp.sqrt(wp.float64(3.0)/wp.float64(2.0))
                #                                       )

                #     # Update stress
                #     eps_v_tmp = wp.trace(real_strain_array[local_iter+1])
                #     tau_tmp = lame_lambda*eps_v_tmp*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*real_strain_array[local_iter+1]

                #     P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_tmp)
                #     S_trial = tau_tmp - P_trial*wp.identity(n=3, dtype=wp.float64)
                #     S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
                #     Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))


                #     real_strain = wp.mat33d(
                #                   real_strain_array[local_iter+1][0,0], wp.float64(0.), wp.float64(0.),
                #                   wp.float64(0.), real_strain_array[local_iter+1][1,1], wp.float64(0.),
                #                   wp.float64(0.), wp.float64(0.), real_strain_array[local_iter+1][2,2]
                #                   )



            # saved_pi[0] = pi_array[14*15 + 14+1]

            saved_pi[0] = pi_array[15]
            # saved_H[0] = old_pi

    return real_strain







# # TODO: MAKE IT WORK
# @wp.kernel
# def return_mapping_DP_kernel(new_strain_vector: wp.array(dtype=wp.float64), # Voigt notation
#                              lame_lambda: wp.float64,
#                              lame_mu: wp.float64,
#                              friction_angle: wp.float64,
#                              dilation_angle: wp.float64,
#                              cohesion: wp.float64,
#                              shape_factor: wp.float64,
#                              tol: wp.float64,
#                              target_stress_xx: wp.float64,
#                              target_stress_yy: wp.float64,
#                              rhs: wp.array(dtype=wp.float64),
#                              saved_stress: wp.array(dtype=wp.mat33d),
#                              real_strain_array: wp.array(dtype=wp.mat33d),
#                              delta_lambda_array: wp.array(dtype=wp.float64)):

#     float64_one = wp.float64(1.0)
#     float64_zero = wp.float64(0.0)
    
#     trial_strain = wp.mat33d(
#                    new_strain_vector[0], float64_zero, float64_zero,
#                    float64_zero, new_strain_vector[1], float64_zero,
#                    float64_zero, float64_zero, new_strain_vector[2]
#                    )

#     real_strain = trial_strain

#     # Construct elastic_a
#     elastic_a_tmp1 = lame_lambda * wp.mat33d(wp.float64(1.), wp.float64(1.), wp.float64(1.),
#                                             wp.float64(1.), wp.float64(1.), wp.float64(1.),
#                                             wp.float64(1.), wp.float64(1.), wp.float64(1.))
#     elastic_a_tmp2 = wp.float64(2.) * lame_mu * wp.identity(n=3, dtype=wp.float64)
#     elastic_a = elastic_a_tmp1 + elastic_a_tmp2

#     # Calculate trial stress
#     eps_v = wp.trace(trial_strain)
#     tau_trial = lame_lambda*eps_v*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*trial_strain

#     # Get P and Q invariants
#     P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
#     S_trial = tau_trial - P_trial*wp.identity(n=3, dtype=wp.float64)
#     S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
#     Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))

#     # Check yield
#     yield_y = yield_function_DP(P_trial, Q_trial, friction_angle, dilation_angle, cohesion, shape_factor)

#     if yield_y<wp.float64(1e-10):
#         pass
#     elif P_trial>=wp.float64(0.):
#         pass
#         # TODO
#     else: # Plasticity
#         # Local newton iteration

        
#         # tau_trial_array[0] = tau_trial
#         # P_trial_array[0] = P_trial
#         # Q_trial_array[0] = Q_trial
#         # S_trial_array[0] = S_trial
#         real_strain_array[0] = trial_strain
#         delta_lambda_array[0] = wp.float64(0.)

#         # delta_lambda = wp.float64(0.) # Local iter converges weel, but seems this will affect the outer gradient calculation
#         test = wp.mat33d()

#         for local_iter in range(1):
#             grad_g = grad_potential_DP(tau_trial, Q_trial, dilation_angle, cohesion, shape_factor)
#             grad_f = grad_yield_DP(tau_trial, Q_trial, friction_angle, cohesion, shape_factor)
#             grad_f_eps = grad_yield_epsilon_DP(elastic_a, grad_f)

#             hess_g = hess_potential_DP(S_trial, Q_trial, dilation_angle, cohesion, shape_factor)
#             hess_g_eps = hess_potential_epsilon_DP(hess_g, elastic_a) 

#             yield_y_iter = yield_function_DP(P_trial, Q_trial, friction_angle, dilation_angle, cohesion, shape_factor)

#             residual = wp.vec4d(
#                        real_strain_array[local_iter][0,0] - trial_strain[0,0] + delta_lambda_array[local_iter]*grad_g[0],
#                        real_strain_array[local_iter][1,1] - trial_strain[1,1] + delta_lambda_array[local_iter]*grad_g[1],
#                        real_strain_array[local_iter][2,2] - trial_strain[2,2] + delta_lambda_array[local_iter]*grad_g[2],
#                        yield_y_iter
#                        )

#             residual_norm = wp.sqrt(residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2] + residual[3]*residual[3])

#             # if residual_norm<tol:
#             #     break



#             # Assemble Jacobian
#             jacobian = wp.mat44d(
#                        wp.float64(1.) + delta_lambda_array[local_iter]*hess_g_eps[0,0], delta_lambda_array[local_iter]*hess_g_eps[0,1], delta_lambda_array[local_iter]*hess_g_eps[0,2], grad_g[0],
#                        delta_lambda_array[local_iter]*hess_g_eps[1,0], wp.float64(1.) + delta_lambda_array[local_iter]*hess_g_eps[1,1], delta_lambda_array[local_iter]*hess_g_eps[1,2], grad_g[1],
#                        delta_lambda_array[local_iter]*hess_g_eps[2,0], delta_lambda_array[local_iter]*hess_g_eps[2,1], wp.float64(1.) + delta_lambda_array[local_iter]*hess_g_eps[2,2], grad_g[2],
#                        grad_f_eps[0], grad_f_eps[1], grad_f_eps[2], wp.float64(0.)
#                        )
#             xdelta = wp.inverse(jacobian) @ residual

        


#             # Update variables
#             delta_strain = wp.mat33d(
#                            -xdelta[0], wp.float64(0.), wp.float64(0.),
#                            wp.float64(0.), -xdelta[1], wp.float64(0.),
#                            wp.float64(0.), wp.float64(0.), -xdelta[2]
#                            )
#             real_strain_array[local_iter+1] = real_strain_array[local_iter] + delta_strain

#             delta_lambda_array[local_iter+1] = delta_lambda_array[local_iter] - xdelta[3]

#             # NO GLOBAL ARRAY
#             # real_strain[0,0] = real_strain[0,0] - xdelta[0] # NOTE: THIS DOES NOT UPDATE THE MATRIX
#             # real_strain[1,1] = real_strain[1,1] - xdelta[1]
#             # real_strain[2,2] = real_strain[2,2] - xdelta[2]

#             # delta_strain = wp.mat33d(
#             #                -xdelta[0], wp.float64(0.), wp.float64(0.),
#             #                wp.float64(0.), -xdelta[1], wp.float64(0.),
#             #                wp.float64(0.), wp.float64(0.), -xdelta[2]
#             #                )
#             # real_strain = real_strain + delta_strain

#             # delta_lambda = delta_lambda - xdelta[3]

#             tmp = wp.mat33d(wp.float64(1.), wp.float64(0.), wp.float64(0.),
#                             wp.float64(0.), wp.float64(0.), wp.float64(0.),
#                             wp.float64(0.), wp.float64(0.), wp.float64(0.))
#             test = test + tmp


#             # print(local_iter)

#             # # Update stress
#             # eps_v_iter = wp.trace(real_strain_array[local_iter+1])
#             # tau_trial = lame_lambda*eps_v_iter*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*real_strain_array[local_iter+1]

#             # P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
#             # S_trial = tau_trial - P_trial*wp.identity(n=3, dtype=wp.float64)
#             # S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
#             # Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))

#             # real_strain = wp.mat33d(
#             #               real_strain_array[local_iter+1][0,0], wp.float64(0.), wp.float64(0.),
#             #               wp.float64(0.), real_strain_array[local_iter+1][1,1], wp.float64(0.),
#             #               wp.float64(0.), wp.float64(0.), real_strain_array[local_iter+1][2,2]
#             #               )

#             real_strain = real_strain_array[1]

        
#     # After return mapping
#     e_trace = wp.trace(real_strain)

#     stress_principal = lame_lambda*e_trace*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*real_strain

#     saved_stress[0] = stress_principal

#     # Here assuming the stress only involves normal components (i.e., no shearing)
#     target_stress = wp.matrix(
#                     target_stress_xx, float64_zero, float64_zero,
#                     float64_zero, target_stress_yy, float64_zero,
#                     float64_zero, float64_zero, stress_principal[2,2],
#                     shape=(3,3)
#                     )
#     stress_residual = target_stress - stress_principal

#     # Assemble to rhs
#     wp.atomic_add(rhs, 0, stress_residual[0,0])
#     wp.atomic_add(rhs, 1, stress_residual[1,1])



@wp.func
def yield_function_DP(P_trial: wp.float64,
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
def grad_potential_DP(tau0_trial: wp.float64,#tau_trial: wp.mat33d,
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
def grad_yield_DP(tau0_trial: wp.float64, #tau_trial: wp.mat33d,
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
def grad_yield_epsilon_DP(elastic_a: wp.mat33d,
                          grad_f: wp.vec3d) -> wp.vec3d:

    grad_f_eps = wp.vec3d(
                 grad_f[0]*elastic_a[0,0] + grad_f[1]*elastic_a[1,0] + grad_f[2]*elastic_a[2,0],
                 grad_f[0]*elastic_a[0,1] + grad_f[1]*elastic_a[1,1] + grad_f[2]*elastic_a[2,1],
                 grad_f[0]*elastic_a[0,2] + grad_f[1]*elastic_a[1,2] + grad_f[2]*elastic_a[2,2]
                 )

    return grad_f_eps


@wp.func
def hess_potential_DP(S_trial: wp.mat33d,
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
def hess_potential_epsilon_DP(hess_g: wp.mat33d,
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
def yield_function_NorSand(P_trial: wp.float64, 
                           Q_trial: wp.float64, 
                           M: wp.float64, 
                           N: wp.float64, 
                           pi_variable: wp.float64):
    
    eta = eta_function_NorSand(M, N, P_trial, pi_variable)

    return Q_trial + eta * P_trial


@wp.func
def eta_function_NorSand(M: wp.float64, 
                         N: wp.float64,
                         P_trial: wp.float64,
                         pi_variable: wp.float64):
    eta = wp.float64(0.0)
    if wp.abs(N) < 1e-8:
        eta = M * (wp.float64(1.0) + wp.log(pi_variable/P_trial))
    else:
        eta = (M/N) * (wp.float64(1.0) - (wp.float64(1.0)-N)*wp.pow(P_trial/pi_variable, N/(wp.float64(1.0)-N)))

    return eta

@ wp.func
def dEtadP_function_NorSand(M: wp.float64,
                            N: wp.float64,
                            P_trial: wp.float64,
                            pi_variable: wp.float64):
    dEtadP = wp.float64(0.0)
    if wp.abs(N) < 1e-8:
        dEtadP = M * wp.float64(-1.0)/P_trial
    else:
        dEtadP = M * (wp.float64(-1.0)*wp.pow(P_trial/pi_variable, (wp.float64(2.0)*N-wp.float64(1.0))/(wp.float64(1.0)-N))/pi_variable)

    return dEtadP

@wp.func
def dEtadPi_function_NorSand(M: wp.float64,
                             N: wp.float64,
                             P_trial: wp.float64,
                             pi_variable: wp.float64):
    dEtadPi = wp.float64(0.0)
    if wp.abs(N) < 1e-8:
        dEtadPi = M/pi_variable
    else:
        dEtadPi = M * (wp.float64(-1.0)*wp.pow(P_trial/pi_variable, (wp.float64(2.0)*N-wp.float64(1.0))/(wp.float64(1.0)-N)) * (-P_trial)/wp.pow(pi_variable, wp.float64(2.0)))

    return dEtadPi

@wp.func
def dPistar_dP_function_NorSand(M: wp.float64, 
                                N: wp.float64, 
                                alpha_bar: wp.float64, 
                                psi_i: wp.float64):
    dPistar_dP = wp.float64(0.0)
    if wp.abs(N) < 1e-8:
        dPistar_dP = wp.exp(alpha_bar*psi_i/M)
    else:
        dPistar_dP = wp.pow(wp.float64(1.0)-alpha_bar*psi_i*N/M, (N-wp.float64(1.0))/N)

    return dPistar_dP

@wp.func
def dPistar_dPi_function_NorSand(M: wp.float64, 
                                 N: wp.float64, 
                                 alpha_bar: wp.float64, 
                                 P_trial: wp.float64, 
                                 psi_i: wp.float64, 
                                 dPsii_dPi: wp.float64):
    dPistar_dPi = wp.float64(0.0)
    if wp.abs(N) < 1e-8:
        dPistar_dPi = P_trial * wp.exp(alpha_bar*psi_i/M) * alpha_bar/M * dPsii_dPi
    else:
        dPistar_dPi = P_trial * (N-wp.float64(1.0))/N * wp.pow(wp.float64(1.0)/(wp.float64(1.0)-alpha_bar*psi_i*N/M), wp.float64(1.0)/N) * (-alpha_bar*N/M) * (dPsii_dPi)

    return dPistar_dPi

