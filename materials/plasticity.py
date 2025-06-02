import sys
sys.path.append('..')

import warp as wp
import numpy as np


@wp.func
def return_mapping_J2(e_trial: wp.mat33d,
					  lame_lambda: wp.float64,
					  lame_mu: wp.float64,
					  kappa: wp.float64
					  ):
	# Get trial Kirchhoff stress
	eps_v = wp.trace(e_trial)
	tau_trial = lame_lambda*eps_v*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*e_trial

	# Get P and S
	P = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
	S_trial = tau_trial - P*wp.identity(n=3, dtype=wp.float64)
	S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))

	e_real = e_trial

	if S_trial_norm>kappa:
		# plasticity
		# radial return mapping
		n = S_trial/S_trial_norm

		delta_lambda = (S_trial_norm - kappa)/(wp.float64(2.)*lame_mu)

		e_real = e_trial - delta_lambda*n
	else:
		# elasticity
		e_real = e_trial

	return e_real