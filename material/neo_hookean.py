import warp as wp
import numpy as np

class NeoHookean:
	def __init__(self,
				 youngs_modulus,
				 poisson_ratio,
				 n_particles
		):
		self.youngs_modulus = youngs_modulus
		self.poisson_ratio = poisson_ratio
		self.lame_lambda = youngs_modulus*poisson_ratio / ((1.0+poisson_ratio) * (1.0-2.0*poisson_ratio))
		self.lame_mu = youngs_modulus / (2.0*(1.0+poisson_ratio))

		self.n_particles = n_particles

		self.deformation_gradient_array = wp.zeros(shape=self.n_particles, dtype=wp.mat33d)
		self.cauchy_stress_array = wp.zeros(shape=self.n_particles, dtype=wp.mat33d)

		wp.launch(kernel=initialization,
				  dim=n_particles,
				  inputs=[self.deformation_gradient_array])

	@wp.kernel
	def initialization(deformation_gradient_array: wp.array(dtype=wp.mat33d)):
	    p = wp.tid()

	    float64_one = wp.float64(1.0)
	    float64_zero = wp.float64(0.0)

	    identity_matrix = wp.matrix(
	                      float64_one, float64_zero, float64_zero,
	                      float64_zero, float64_one, float64_zero,
	                      float64_zero, float64_zero, float64_one,
	                      shape=(3,3)
	        )

	    deformation_gradient_array[p] = identity_matrix



	# update stress from trial deformation gradient
	@wp.kernel
	def update_stress(deformation_gradient_array: wp.array(dtype=wp.mat33d),
					  cauchy_stress_array: wp.array(dtype=wp.mat33d),
					  lame_lambda: wp.float64,
					  lame_mu: wp.float64):
		p = wp.tid()

		particle_F = deformation_gradient_array[p]

		# return mapping (no need for Neo-Hookean)

		# calculate stress
		particle_F_inv = wp.inverse(particle_F)
		particle_J = wp.determinant(particle_F)
		particle_PK1_stress = lame_mu * (particle_F - wp.transpose(particle_F_inv)) + lame_lambda * wp.log(particle_J) * wp.transpose(particle_F_inv)

		particle_Cauchy_stress = wp.float64(1.)/particle_J * particle_PK1_stress @ wp.transpose(particle_F)

		cauchy_stress_array[p] = particle_Cauchy_stress



