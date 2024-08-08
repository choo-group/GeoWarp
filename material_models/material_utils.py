import warp as wp
import numpy as np

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
def return_mapping_J2(trial_strain: wp.mat33d,
                      lame_lambda: wp.float64,
                      lame_mu: wp.float64,
                      kappa: wp.float64):
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


