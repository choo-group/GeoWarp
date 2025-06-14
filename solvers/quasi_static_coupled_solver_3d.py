import sys
sys.path.append('..')

import warp as wp
import warp.sparse as wps
import warp.optim.linear

import numpy as np

import scipy.linalg as sla 
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pyamg import smoothed_aggregation_solver
import pypardiso

from solvers.sparse_differentiation import pick_grid_nodes_3d, precompute_seed_vectors_coupled_3d, from_jacobian_to_vector_parallel_coupled_3d
from mpm.mpm_utils import init_particles_rectangle_3d, initialization, P2G_coupled_3d, P2G_set_nodal_solution_3d, assemble_residual_coupled_3d_neohookean, set_diagnal_component_for_boundary_and_deactivated_dofs_coupled_3d, from_increment_to_solution, G2P_coupled_3d, initialize_youngs_modulus_diff
from mpm.gimp import initialize_GIMP_lp_3d, update_GIMP_lp_3d


import os

print("\n" + "=" * 72)
print(r"""
   ____ _____ _____        ___    ____  ____  
  / ___| ____/ _ \ \      / / \  |  _ \|  _ \ 
 | |  _|  _|| | | \ \ /\ / / _ \ | |_) | |_) |
 | |_| | |__| |_| |\ V  V / ___ \|  _ <|  __/ 
  \____|_____\___/  \_/\_/_/   \_\_| \_\_|    
""")
print("=" * 72)
print("  PROGRAM INFO")
print("    Module             :: Implicit MPM - Quasi-static u-p coupled solver (3D)")
print("    Authors            :: Yidong Zhao\n")
print("  RUN INFO")
print(f"    Hostname           :: {os.uname().nodename}\n")


@wp.struct
class DofStruct:
	activate_flag_array: wp.array(dtype=wp.bool)
	boundary_flag_array: wp.array(dtype=wp.bool)

@wp.kernel
def get_indentation_loss(indentation_force_array: wp.array(dtype=wp.float64),
								 indentation_force: wp.array(dtype=wp.float64),
								 loss: wp.array(dtype=wp.float64),
								 footing_incr_disp: wp.float64,
								 total_steps: wp.float64
								 ):
	loss[0] = (indentation_force[0]/(wp.abs(footing_incr_disp)*total_steps) - wp.float64(13567.27337196))**wp.float64(2.) # The magic number 13567.27337196 is from E = 1e4 kPa
	
	

class quasi_static_coupled_solver_3d:
	def __init__(self,
				 background_grid_dict,
				 particles_dict,
				 iteration_dict,
				 material_dict,
				 porous_media_dict,
				 load_dict,
				 set_boundary_dofs
				 ):
		# Background grid
		self.n_grid_x = background_grid_dict['n_grid_x']
		self.n_grid_y = background_grid_dict['n_grid_y']
		self.n_grid_z = background_grid_dict['n_grid_z']
		self.n_nodes = (self.n_grid_x+1) * (self.n_grid_y+1) * (self.n_grid_z+1)
		self.n_matrix_size = 4 * self.n_nodes # 3 includes the spatial dimension (2) plus pore water pressure
		self.max_x = background_grid_dict['max_x']
		self.dx = background_grid_dict['dx']
		self.max_y = background_grid_dict['max_y']
		self.dy = background_grid_dict['dy']
		self.max_z = background_grid_dict['max_z']
		self.dz = background_grid_dict['dz']
		self.inv_dx = 1./self.dx
		self.inv_dy = 1./self.dy
		self.inv_dz = 1./self.dz
		self.start_x = background_grid_dict['start_x']
		self.end_x = background_grid_dict['end_x']
		self.start_y = background_grid_dict['start_y']
		self.end_y = background_grid_dict['end_y']
		self.start_z = background_grid_dict['start_z']
		self.end_z = background_grid_dict['end_z']
		self.domain_min_x = background_grid_dict['domain_min_x']
		self.domain_max_x = background_grid_dict['domain_max_x']
		self.domain_min_y = background_grid_dict['domain_min_y']
		self.domain_max_y = background_grid_dict['domain_max_y']
		self.domain_min_z = background_grid_dict['domain_min_z']
		self.domain_max_z = background_grid_dict['domain_max_z']

		# Particles
		self.n_particles = particles_dict['n_particles']
		self.PPD = particles_dict['PPD']
		self.p_vol = particles_dict['p_vol']
		self.p_rho = particles_dict['p_rho']
		self.p_mass = self.p_vol * self.p_rho
		self.GIMP_lp_initial = self.dx/self.PPD/2.0 

		# Newton iteration
		self.n_iter = iteration_dict['n_iter']
		self.tol = iteration_dict['tol']
		self.solver_name = iteration_dict['solver_name']
		self.dt = iteration_dict['dt']

		# Material properties
		self.youngs_modulus = material_dict['youngs_modulus']
		print('self.youngs_modulus:', self.youngs_modulus)
		self.poisson_ratio = material_dict['poisson_ratio']
		self.lame_lambda = self.youngs_modulus*self.poisson_ratio / ((1.0+self.poisson_ratio) * (1.0-2.0*self.poisson_ratio))
		self.lame_mu = self.youngs_modulus / (2.0*(1.0+self.poisson_ratio))
		self.material_name = material_dict['material_name']

		# Porous media properties
		self.phi_initial = porous_media_dict['phi_initial']
		self.mobility_constant = porous_media_dict['mobility_constant']

		# Loading
		self.gravity_mag = load_dict['gravity_mag']
		self.traction_value_x = load_dict['traction_value_x']
		self.traction_value_y = load_dict['traction_value_y']
		self.traction_value_z = load_dict['traction_value_z']
		self.point_load_value_x = load_dict['point_load_value_x']
		self.point_load_value_y = load_dict['point_load_value_y']
		self.point_load_value_z = load_dict['point_load_value_z']
		self.footing_region_min_x = load_dict['footing_region_min_x']
		self.footing_region_max_x = load_dict['footing_region_max_x']
		self.footing_region_min_y = load_dict['footing_region_min_y']
		self.footing_region_max_y = load_dict['footing_region_max_y']
		self.footing_initial_height = load_dict['footing_initial_height']
		self.footing_incr_disp = load_dict['footing_incr_disp']
		self.penalty_factor = load_dict['penalty_factor']

		# Boundary setup function
		self.set_boundary_dofs = set_boundary_dofs

		# DofStruct
		self.dofStruct = DofStruct()
		self.dofStruct.activate_flag_array = wp.zeros(shape=self.n_matrix_size, dtype=wp.bool)
		self.dofStruct.boundary_flag_array = wp.zeros(shape=self.n_matrix_size, dtype=wp.bool)

		# ============ Warp arrays (particles) ============
		particles_pos_np = init_particles_rectangle_3d(self.start_x, self.start_y, self.start_z, self.end_x, self.end_y, self.end_z, self.dx, self.n_grid_x, self.n_grid_y, self.n_grid_z, self.PPD, self.n_particles)
		self.x_particles = wp.from_numpy(particles_pos_np, dtype=wp.vec3d)
		self.x0 = wp.from_numpy(particles_pos_np, dtype=wp.vec3d)
		self.deformation_gradient_total_new = wp.array(shape=self.n_particles, dtype=wp.mat33d)
		self.deformation_gradient_total_old = wp.array(shape=self.n_particles, dtype=wp.mat33d)
		self.left_Cauchy_Green_new = wp.array(shape=self.n_particles, dtype=wp.mat33d)
		self.left_Cauchy_Green_old = wp.array(shape=self.n_particles, dtype=wp.mat33d)
		self.particle_Cauchy_stress_array = wp.array(shape=self.n_particles, dtype=wp.mat33d)
		self.particle_pressure_array = wp.array(shape=self.n_particles, dtype=wp.float64)
		self.GIMP_lp = wp.array(shape=self.n_particles, dtype=wp.vec3d)
		self.particle_external_flag_array = wp.array(shape=self.n_particles, dtype=wp.bool)
		self.particle_traction_flag_array = wp.array(shape=self.n_particles, dtype=wp.bool)
		self.particle_pressure_boundary_flag_array = wp.array(shape=self.n_particles, dtype=wp.bool)

		# Inverse
		self.youngs_modulus_diff = wp.zeros(shape=1, dtype=wp.float64, requires_grad=True)
		self.indentation_force = wp.zeros(shape=1, dtype=wp.float64, requires_grad=True)
		self.indentation_force_array = wp.zeros(shape=25, dtype=wp.float64, requires_grad=True)
		self.indentation_force_list = np.empty(0)
		self.loss = wp.zeros(shape=1, dtype=wp.float64, requires_grad=True)
		self.E_grad = None

		# ============ Warp arrays (grid and matrix system) ============
		self.rhs = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64, requires_grad=True)
		self.old_solution = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64)
		self.new_solution = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64, requires_grad=True)
		self.increment_iteration = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64)

		self.rhs_P2G = wp.zeros(shape=self.n_nodes, dtype=wp.float64)
		self.rows_P2G = wp.zeros(shape=27*27*self.n_particles, dtype=wp.int32)
		self.cols_P2G = wp.zeros(shape=27*27*self.n_particles, dtype=wp.int32)
		self.vals_P2G = wp.zeros(shape=27*27*self.n_particles, dtype=wp.float64)
		self.x_P2G_warp = wp.zeros(shape=self.n_nodes, dtype=wp.float64)

		self.rows = wp.zeros(shape=4*125*self.n_matrix_size+self.n_matrix_size, dtype=wp.int32)
		self.cols = wp.zeros(shape=4*125*self.n_matrix_size+self.n_matrix_size, dtype=wp.int32)
		self.vals = wp.zeros(shape=4*125*self.n_matrix_size+self.n_matrix_size, dtype=wp.float64)

		self.bsr_matrix = wps.bsr_zeros(self.n_matrix_size, self.n_matrix_size, block_type=wp.float64)
		self.bsr_matrix_P2G = wps.bsr_zeros(self.n_nodes, self.n_nodes, block_type=wp.float64)

		# ============ Initialization ============
		wp.launch(kernel=initialization,
				  dim=self.n_particles,
				  inputs=[self.deformation_gradient_total_new, self.deformation_gradient_total_old, self.left_Cauchy_Green_new, self.left_Cauchy_Green_old])

		wp.launch(kernel=initialize_GIMP_lp_3d,
				  dim=self.n_particles,
				  inputs=[self.GIMP_lp, self.GIMP_lp_initial])

		wp.launch(kernel=initialize_youngs_modulus_diff,
				  dim=1,
				  inputs=[self.youngs_modulus_diff, self.youngs_modulus])

		# ============ Sparse differentiation setup ============
		selector = pick_grid_nodes_3d(self.n_grid_x, self.n_grid_y, self.n_grid_z, 0)
		self.max_selector_length = len(selector)

		self.selector_x_list, self.selector_y_list, self.selector_z_list, self.selector_p_list, self.e_x_list, self.e_y_list, self.e_z_list, self.e_p_list = precompute_seed_vectors_coupled_3d(self.n_grid_x, self.n_grid_y, self.n_grid_z, self.n_nodes, self.n_matrix_size, self.max_selector_length)

	# ============ Class functions ============
	def reinit_everything(self):
		particles_pos_np = init_particles_rectangle_3d(self.start_x, self.start_y, self.start_z, self.end_x, self.end_y, self.end_z, self.dx, self.n_grid_x, self.n_grid_y, self.n_grid_z, self.PPD, self.n_particles)
		self.x_particles = wp.from_numpy(particles_pos_np, dtype=wp.vec3d)

		self.E_grad = None
		self.indentation_force_list = np.empty(0)

		self.particle_pressure_array.zero_()

		wp.launch(kernel=initialization,
				  dim=self.n_particles,
				  inputs=[self.deformation_gradient_total_new, self.deformation_gradient_total_old, self.left_Cauchy_Green_new, self.left_Cauchy_Green_old])

		wp.launch(kernel=initialize_GIMP_lp_3d,
				  dim=self.n_particles,
				  inputs=[self.GIMP_lp, self.GIMP_lp_initial])

	def reset_grid(self, dx, dy, dz):
		self.dx = dx
		self.dy = dy
		self.dz = dz

		self.inv_dx = 1. / self.dx
		self.inv_dy = 1. / self.dy
		self.inv_dz = 1. / self.dz

	def reset_youngs_modulus(self, youngs_modulus):
		self.youngs_modulus = youngs_modulus
		self.lame_lambda = self.youngs_modulus*self.poisson_ratio / ((1.0+self.poisson_ratio) * (1.0-2.0*self.poisson_ratio))
		self.lame_mu = self.youngs_modulus / (2.0*(1.0+self.poisson_ratio))

		wp.launch(kernel=initialize_youngs_modulus_diff,
				  dim=1,
				  inputs=[self.youngs_modulus_diff, self.youngs_modulus])


	def reset(self):
		self.old_solution.zero_()
		self.new_solution.zero_()
		self.increment_iteration.zero_()

		self.dofStruct.boundary_flag_array.zero_()
		self.dofStruct.activate_flag_array.zero_()

		self.rhs_P2G.zero_()
		self.rows_P2G.zero_()
		self.cols_P2G.zero_()
		self.vals_P2G.zero_()
		self.x_P2G_warp.zero_()
		wps.bsr_set_zero(self.bsr_matrix_P2G)


	def newton_iter(self, current_step, total_steps):
		for iter_id in range(self.n_iter):
			self.rhs.zero_()
			self.rows.zero_()
			self.cols.zero_()
			self.vals.zero_()

			self.indentation_force.zero_()
			self.indentation_force_array.zero_()
			
			self.loss.zero_()

			tape = wp.Tape()
			with tape:
				if self.material_name=='Neo-Hookean':
					wp.launch(kernel=assemble_residual_coupled_3d_neohookean, 
							  dim=self.n_particles,
							  inputs=[self.x_particles, self.dx, self.dy, self.dz, self.inv_dx, self.inv_dy, self.inv_dz, self.dt, self.n_grid_x, self.n_grid_y, self.n_nodes, self.PPD, self.old_solution, self.new_solution, self.deformation_gradient_total_old, self.deformation_gradient_total_new, self.left_Cauchy_Green_old, self.left_Cauchy_Green_new, self.particle_external_flag_array, self.particle_traction_flag_array, self.particle_Cauchy_stress_array, self.phi_initial, self.mobility_constant, self.gravity_mag, self.traction_value_x, self.traction_value_y, self.traction_value_z, self.point_load_value_x, self.point_load_value_y, self.point_load_value_z, self.footing_region_min_x, self.footing_region_max_x, self.footing_region_min_y, self.footing_region_max_y, self.footing_initial_height, self.footing_incr_disp, self.penalty_factor, current_step, total_steps, self.youngs_modulus_diff, self.poisson_ratio, self.p_vol, self.p_rho, self.GIMP_lp, self.dofStruct.boundary_flag_array, self.dofStruct.activate_flag_array, self.rhs, self.indentation_force, self.indentation_force_array])
				else:
					raise ValueError(f"Unimplemented material model: {self.material_name}")

				wp.launch(kernel=get_indentation_loss,
							 dim=1,
							 inputs=[self.indentation_force_array, self.indentation_force, self.loss, self.footing_incr_disp, total_steps])

			# Assemble the sparse Jacobian matrix using automatic differentiation
			# Sparse differentiation
			pattern_id = 0
			for c_iter in range(5):
				for r_iter in range(5):
					for l_iter in range(5):
						# x
						tape.backward(grads={self.rhs: self.e_x_list[pattern_id]})
						jacobian_wp = tape.gradients[self.new_solution]
						wp.launch(kernel=from_jacobian_to_vector_parallel_coupled_3d,
								  dim=self.max_selector_length,
								  inputs=[jacobian_wp, self.rows, self.cols, self.vals, self.n_grid_x, self.n_grid_y, self.n_grid_z, self.n_nodes, self.n_matrix_size, self.selector_x_list[pattern_id], self.dofStruct.boundary_flag_array, self.dofStruct.activate_flag_array])
						tape.zero()

						# y
						tape.backward(grads={self.rhs: self.e_y_list[pattern_id]})
						jacobian_wp = tape.gradients[self.new_solution]
						wp.launch(kernel=from_jacobian_to_vector_parallel_coupled_3d,
								  dim=self.max_selector_length,
								  inputs=[jacobian_wp, self.rows, self.cols, self.vals, self.n_grid_x, self.n_grid_y, self.n_grid_z, self.n_nodes, self.n_matrix_size, self.selector_y_list[pattern_id], self.dofStruct.boundary_flag_array, self.dofStruct.activate_flag_array])
						tape.zero()

						# z
						tape.backward(grads={self.rhs: self.e_z_list[pattern_id]})
						jacobian_wp = tape.gradients[self.new_solution]
						wp.launch(kernel=from_jacobian_to_vector_parallel_coupled_3d,
								  dim=self.max_selector_length,
								  inputs=[jacobian_wp, self.rows, self.cols, self.vals, self.n_grid_x, self.n_grid_y, self.n_grid_z, self.n_nodes, self.n_matrix_size, self.selector_z_list[pattern_id], self.dofStruct.boundary_flag_array, self.dofStruct.activate_flag_array])
						tape.zero()

						# p  
						tape.backward(grads={self.rhs: self.e_p_list[pattern_id]})
						jacobian_wp = tape.gradients[self.new_solution]
						wp.launch(kernel=from_jacobian_to_vector_parallel_coupled_3d,
								  dim=self.max_selector_length,
								  inputs=[jacobian_wp, self.rows, self.cols, self.vals, self.n_grid_x, self.n_grid_y, self.n_grid_z, self.n_nodes, self.n_matrix_size, self.selector_p_list[pattern_id], self.dofStruct.boundary_flag_array, self.dofStruct.activate_flag_array])
						tape.zero()

						pattern_id = pattern_id + 1

			if current_step == total_steps-1:
				tape.backward(self.loss)
				self.E_grad = self.youngs_modulus_diff.grad.numpy()

			tape.reset()

			# Adjust diagonal components of the global matrix
			wp.launch(kernel=set_diagnal_component_for_boundary_and_deactivated_dofs_coupled_3d,
					  dim=self.n_matrix_size,
					  inputs=[self.dofStruct.boundary_flag_array, self.dofStruct.activate_flag_array, self.rows, self.cols, self.vals, self.n_matrix_size])

			# Solve the matrix equation
			if self.solver_name=='warp':
				wps.bsr_set_from_triplets(self.bsr_matrix, self.rows, self.cols, self.vals)
				preconditioner = wp.optim.linear.preconditioner(self.bsr_matrix, ptype='diag')
				solver_state = wp.optim.linear.bicgstab(A=self.bsr_matrix, b=self.rhs, x=self.increment_iteration, tol=1e-10, M=preconditioner)				
			elif self.solver_name=='pyamg':
				bsr_matrix_pyagm = sparse.coo_matrix((self.vals.numpy(), (self.rows.numpy(), self.cols.numpy())), shape=(self.n_matrix_size, self.n_matrix_size)).asformat('csr')
				mls = smoothed_aggregation_solver(bsr_matrix_pyagm)

				b = self.rhs.numpy()
				residuals = []
				x_pyamg = mls.solve(b, tol=1e-8, accel='bicgstab', residuals=residuals)

				self.increment_iteration = wp.from_numpy(x_pyamg, dtype=wp.float64)
			elif self.solver_name=='scipy':
				bsr_matrix_scipy = sparse.coo_matrix((self.vals.numpy(), (self.rows.numpy(), self.cols.numpy())), shape=(self.n_matrix_size, self.n_matrix_size)).asformat('csr')
				b = self.rhs.numpy()
				
				x_direct = spsolve(bsr_matrix_scipy, b)

				self.increment_iteration = wp.from_numpy(x_direct, dtype=wp.float64)
			elif self.solver_name=='pypardiso':
				bsr_matrix_pypardiso = sparse.coo_matrix((self.vals.numpy(), (self.rows.numpy(), self.cols.numpy())), shape=(self.n_matrix_size, self.n_matrix_size)).asformat('csr')
				b = self.rhs.numpy()

				x_pypardiso = pypardiso.spsolve(bsr_matrix_pypardiso, b)

				self.increment_iteration = wp.from_numpy(x_pypardiso, dtype=wp.float64)
			else:
				raise ValueError(f"Unimplemented solver: {self.solver_name}")


			# From increment to solution
			wp.launch(kernel=from_increment_to_solution,
					  dim=self.n_matrix_size,
					  inputs=[self.increment_iteration, self.new_solution])


			with np.printoptions(threshold=np.inf):
				print('residual.norm:', np.linalg.norm(self.rhs.numpy()))

			if np.linalg.norm(self.rhs.numpy())<self.tol:
				break

		# G2P
		wp.launch(kernel=G2P_coupled_3d,
				  dim=self.n_particles,
				  inputs=[self.GIMP_lp, self.x_particles, self.dx, self.dy, self.dz, self.inv_dx, self.inv_dy, self.inv_dz, self.n_grid_x, self.n_grid_y, self.n_nodes, self.old_solution, self.new_solution, self.deformation_gradient_total_old, self.deformation_gradient_total_new, self.left_Cauchy_Green_old, self.left_Cauchy_Green_new, self.particle_pressure_array])

		# Update GIMP lps
		wp.launch(kernel=update_GIMP_lp_3d,
				  dim=self.n_particles,
				  inputs=[self.deformation_gradient_total_new, self.GIMP_lp, self.GIMP_lp_initial, self.x_particles, self.domain_min_x, self.domain_max_x, self.domain_min_y, self.domain_max_y, self.domain_min_z, self.domain_max_z])

		# Print indentation force
		self.indentation_force_list = np.append(self.indentation_force_list, self.indentation_force.numpy()[0])
		with np.printoptions(threshold=np.inf):
			for force in self.indentation_force_list:
				print(force)



	def advance_one_step(self, current_step, total_steps):
		self.reset()

		# Set boundary dofs
		wp.launch(kernel=self.set_boundary_dofs,
				  dim=(self.n_grid_x+1, self.n_grid_y+1, self.n_grid_z+1),
				  inputs=[self.dofStruct.boundary_flag_array, self.n_grid_x, self.n_grid_y, self.n_nodes, self.dx, self.dy, self.end_x, self.end_y])

		# P2G
		wp.launch(kernel=P2G_coupled_3d,
				  dim=self.n_particles,
				  inputs=[self.x_particles, self.inv_dx, self.dx, self.inv_dy, self.dy, self.inv_dz, self.dz, self.n_grid_x, self.n_grid_y, self.n_nodes, self.dofStruct.boundary_flag_array, self.dofStruct.activate_flag_array, self.GIMP_lp, self.particle_pressure_boundary_flag_array, self.p_mass, self.particle_pressure_array, self.rhs_P2G, self.rows_P2G, self.cols_P2G, self.vals_P2G])

		# Solve for consistent mapping using warp
		wps.bsr_set_from_triplets(self.bsr_matrix_P2G, self.rows_P2G, self.cols_P2G, self.vals_P2G)
		preconditioner = wp.optim.linear.preconditioner(self.bsr_matrix_P2G, ptype='diag')
		solver_state = wp.optim.linear.bicgstab(A=self.bsr_matrix_P2G, b=self.rhs_P2G, x=self.x_P2G_warp, tol=1e-8, M=preconditioner)
		print('After solving consistent-mass P2G')

		wp.launch(kernel=P2G_set_nodal_solution_3d,
				  dim=self.n_nodes, 
				  inputs=[self.dofStruct.boundary_flag_array, self.dofStruct.activate_flag_array, self.n_nodes, self.old_solution, self.new_solution, self.x_P2G_warp])

		# Newton iteration
		self.newton_iter(current_step, total_steps)



