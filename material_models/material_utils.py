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
            # P_trial_iter = P_trial_array[local_iter]
            # Q_trial_iter = Q_trial_array[local_iter]
            # tau_trial_iter = tau_trial_array[local_iter]
            # delta_lambda_iter = delta_lambda_array[local_iter]

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



# TODO: MAKE IT WORK
@wp.kernel
def return_mapping_DP_kernel(new_strain_vector: wp.array(dtype=wp.float64), # Voigt notation
                             lame_lambda: wp.float64,
                             lame_mu: wp.float64,
                             friction_angle: wp.float64,
                             dilation_angle: wp.float64,
                             cohesion: wp.float64,
                             shape_factor: wp.float64,
                             tol: wp.float64,
                             target_stress_xx: wp.float64,
                             target_stress_yy: wp.float64,
                             rhs: wp.array(dtype=wp.float64),
                             saved_stress: wp.array(dtype=wp.mat33d),
                             real_strain_array: wp.array(dtype=wp.mat33d),
                             delta_lambda_array: wp.array(dtype=wp.float64)):

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    
    trial_strain = wp.mat33d(
                   new_strain_vector[0], float64_zero, float64_zero,
                   float64_zero, new_strain_vector[1], float64_zero,
                   float64_zero, float64_zero, new_strain_vector[2]
                   )

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
    yield_y = yield_function_DP(P_trial, Q_trial, friction_angle, dilation_angle, cohesion, shape_factor)

    if yield_y<wp.float64(1e-10):
        pass
    elif P_trial>=wp.float64(0.):
        pass
        # TODO
    else: # Plasticity
        # Local newton iteration

        
        # tau_trial_array[0] = tau_trial
        # P_trial_array[0] = P_trial
        # Q_trial_array[0] = Q_trial
        # S_trial_array[0] = S_trial
        real_strain_array[0] = trial_strain
        delta_lambda_array[0] = wp.float64(0.)

        # delta_lambda = wp.float64(0.) # Local iter converges weel, but seems this will affect the outer gradient calculation
        test = wp.mat33d()

        for local_iter in range(1):
            grad_g = grad_potential_DP(tau_trial, Q_trial, dilation_angle, cohesion, shape_factor)
            grad_f = grad_yield_DP(tau_trial, Q_trial, friction_angle, cohesion, shape_factor)
            grad_f_eps = grad_yield_epsilon_DP(elastic_a, grad_f)

            hess_g = hess_potential_DP(S_trial, Q_trial, dilation_angle, cohesion, shape_factor)
            hess_g_eps = hess_potential_epsilon_DP(hess_g, elastic_a) 

            yield_y_iter = yield_function_DP(P_trial, Q_trial, friction_angle, dilation_angle, cohesion, shape_factor)

            residual = wp.vec4d(
                       real_strain_array[local_iter][0,0] - trial_strain[0,0] + delta_lambda_array[local_iter]*grad_g[0],
                       real_strain_array[local_iter][1,1] - trial_strain[1,1] + delta_lambda_array[local_iter]*grad_g[1],
                       real_strain_array[local_iter][2,2] - trial_strain[2,2] + delta_lambda_array[local_iter]*grad_g[2],
                       yield_y_iter
                       )

            residual_norm = wp.sqrt(residual[0]*residual[0] + residual[1]*residual[1] + residual[2]*residual[2] + residual[3]*residual[3])

            # if residual_norm<tol:
            #     break



            # Assemble Jacobian
            jacobian = wp.mat44d(
                       wp.float64(1.) + delta_lambda_array[local_iter]*hess_g_eps[0,0], delta_lambda_array[local_iter]*hess_g_eps[0,1], delta_lambda_array[local_iter]*hess_g_eps[0,2], grad_g[0],
                       delta_lambda_array[local_iter]*hess_g_eps[1,0], wp.float64(1.) + delta_lambda_array[local_iter]*hess_g_eps[1,1], delta_lambda_array[local_iter]*hess_g_eps[1,2], grad_g[1],
                       delta_lambda_array[local_iter]*hess_g_eps[2,0], delta_lambda_array[local_iter]*hess_g_eps[2,1], wp.float64(1.) + delta_lambda_array[local_iter]*hess_g_eps[2,2], grad_g[2],
                       grad_f_eps[0], grad_f_eps[1], grad_f_eps[2], wp.float64(0.)
                       )
            xdelta = wp.inverse(jacobian) @ residual

        


            # Update variables
            delta_strain = wp.mat33d(
                           -xdelta[0], wp.float64(0.), wp.float64(0.),
                           wp.float64(0.), -xdelta[1], wp.float64(0.),
                           wp.float64(0.), wp.float64(0.), -xdelta[2]
                           )
            real_strain_array[local_iter+1] = real_strain_array[local_iter] + delta_strain

            delta_lambda_array[local_iter+1] = delta_lambda_array[local_iter] - xdelta[3]

            # NO GLOBAL ARRAY
            # real_strain[0,0] = real_strain[0,0] - xdelta[0] # NOTE: THIS DOES NOT UPDATE THE MATRIX
            # real_strain[1,1] = real_strain[1,1] - xdelta[1]
            # real_strain[2,2] = real_strain[2,2] - xdelta[2]

            # delta_strain = wp.mat33d(
            #                -xdelta[0], wp.float64(0.), wp.float64(0.),
            #                wp.float64(0.), -xdelta[1], wp.float64(0.),
            #                wp.float64(0.), wp.float64(0.), -xdelta[2]
            #                )
            # real_strain = real_strain + delta_strain

            # delta_lambda = delta_lambda - xdelta[3]

            tmp = wp.mat33d(wp.float64(1.), wp.float64(0.), wp.float64(0.),
                            wp.float64(0.), wp.float64(0.), wp.float64(0.),
                            wp.float64(0.), wp.float64(0.), wp.float64(0.))
            test = test + tmp


            # print(local_iter)

            # # Update stress
            # eps_v_iter = wp.trace(real_strain_array[local_iter+1])
            # tau_trial = lame_lambda*eps_v_iter*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*real_strain_array[local_iter+1]

            # P_trial = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
            # S_trial = tau_trial - P_trial*wp.identity(n=3, dtype=wp.float64)
            # S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))
            # Q_trial = S_trial_norm * wp.sqrt(wp.float64(3.)/wp.float64(2.))

            # real_strain = wp.mat33d(
            #               real_strain_array[local_iter+1][0,0], wp.float64(0.), wp.float64(0.),
            #               wp.float64(0.), real_strain_array[local_iter+1][1,1], wp.float64(0.),
            #               wp.float64(0.), wp.float64(0.), real_strain_array[local_iter+1][2,2]
            #               )

            real_strain = real_strain_array[1]

        
    # After return mapping
    e_trace = wp.trace(real_strain)

    stress_principal = lame_lambda*e_trace*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*real_strain

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




