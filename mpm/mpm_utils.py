import sys
sys.path.append('..')

import warp as wp
import numpy as np


from mpm.gimp import get_GIMP_shape_function_and_gradient_2d


def init_particles_rectangle_2d(start_x, start_y, end_x, end_y, dx, n_grid_x, n_grid_y, PPD, n_particles):
	particle_id = 0
	particle_pos_np = np.zeros((n_particles, 2))

	for i in range(n_grid_x):
		for j in range(n_grid_y):
			potential_pos = np.array([(i+0.5)*dx, (j+0.5)*dx])

			if start_x < potential_pos[0] and potential_pos[0] < end_x:
				if start_y < potential_pos[1] and potential_pos[1] < end_y:
					for p_x in range(PPD):
						for p_y in range(PPD):
							particle_pos_np[particle_id] = np.array([i*dx, j*dx]) + np.array([(0.5+p_x)*dx/PPD, (0.5+p_y)*dx/PPD])

							particle_id += 1

	print(particle_id)
	return particle_pos_np


@wp.kernel
def initialization(deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
				   deformation_gradient_total_old: wp.array(dtype=wp.mat33d),
				   left_Cauchy_Green_new: wp.array(dtype=wp.mat33d),
				   left_Cauchy_Green_old: wp.array(dtype=wp.mat33d),
				   x_particles: wp.array(dtype=wp.vec2d)
				   ):
	p = wp.tid()

	float64_one = wp.float64(1.0)
	float64_zero = wp.float64(0.0)

	identity_matrix = wp.matrix(
					  float64_one+ wp.float64(1e-7), float64_zero, float64_zero,
					  float64_zero, float64_one-wp.float64(1e-6), float64_zero,
					  float64_zero, float64_zero, float64_one,
					  shape=(3,3)
		)

	deformation_gradient_total_new[p] = identity_matrix
	deformation_gradient_total_old[p] = identity_matrix
	left_Cauchy_Green_new[p] = identity_matrix
	left_Cauchy_Green_old[p] = identity_matrix


@wp.kernel
def P2G_2d(x_particles: wp.array(dtype=wp.vec2d),
		   inv_dx: wp.float64,
		   dx: wp.float64,
		   n_grid_x: wp.int32,
		   n_nodes: wp.int32,
		   activate_flag_array: wp.array(dtype=wp.bool),
		   GIMP_lp: wp.array(dtype=wp.vec2d)
		   ):
	p = wp.tid()

	float64_one = wp.float64(1.0)
	float64_zero = wp.float64(0.0)


	lpx = GIMP_lp[p][0]
	lpy = GIMP_lp[p][1]


	# GIMP
	left_bottom_corner = x_particles[p] - wp.vec2d(lpx, lpy)
	left_bottom_corner_base_x = left_bottom_corner[0]*inv_dx + wp.float64(1e-8)
	left_bottom_corner_base_y = left_bottom_corner[1]*inv_dx + wp.float64(1e-8)
	left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
	left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))

	right_up_corner = x_particles[p] + wp.vec2d(lpx, lpy)
	right_up_corner_base_x = right_up_corner[0]*inv_dx + wp.float64(1e-8)
	right_up_corner_base_y = right_up_corner[1]*inv_dx + wp.float64(1e-8)
	right_up_corner_base_int = wp.vector(wp.int(right_up_corner_base_x), wp.int(right_up_corner_base_y))
	right_up_corner_base = wp.vector(wp.float64(right_up_corner_base_int[0]), wp.float64(right_up_corner_base_int[1]))

	i_range = 0
	j_range = 0
	if left_bottom_corner_base_int[0]==right_up_corner_base_int[0]:
		i_range = 2
	else:
		i_range = 3
	if left_bottom_corner_base_int[1]==right_up_corner_base_int[1]:
		j_range = 2
	else:
		j_range = 3


	for i in range(0, i_range):
		for j in range(0, j_range):
			ix = left_bottom_corner_base_int[0] + i
			iy = left_bottom_corner_base_int[1] + j

			index_ij_x = ix + iy*(n_grid_x + wp.int(1))
			index_ij_y = index_ij_x + n_nodes

			activate_flag_array[index_ij_x] = True
			activate_flag_array[index_ij_y] = True
	
@wp.kernel
def assemble_residual_2d_hencky(x_particles: wp.array(dtype=wp.vec2d),
							    dx: wp.float64,
							    inv_dx: wp.float64,
							    n_grid_x: wp.int32,
							    n_nodes: wp.int32,
							    increment_solution: wp.array(dtype=wp.float64),
							    deformation_gradient_total_old: wp.array(dtype=wp.mat33d),
							    deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
							    left_Cauchy_Green_old: wp.array(dtype=wp.mat33d),
							    left_Cauchy_Green_new: wp.array(dtype=wp.mat33d),
							    particle_Cauchy_stress_array: wp.array(dtype=wp.mat33d),
							    gravity_mag: wp.float64,
							    traction_value_x: wp.float64,
							    traction_value_y: wp.float64,
							    point_load_value_x: wp.float64,
							    point_load_value_y: wp.float64,
							    current_step: wp.float64,
							    total_steps: wp.float64,
							    lame_lambda: wp.float64,
							    lame_mu: wp.float64,
							    p_vol: wp.float64,
							    p_rho: wp.float64,
							    GIMP_lp: wp.array(dtype=wp.vec2d),
							    boundary_flag_array: wp.array(dtype=wp.bool),
							    activate_flag_array: wp.array(dtype=wp.bool),
							    rhs: wp.array(dtype=wp.float64)
							    ):
	p = wp.tid()

	lpx = GIMP_lp[p][0]
	lpy = GIMP_lp[p][1]

	float64_one = wp.float64(1.0)
	float64_zero = wp.float64(0.0)

	standard_gravity = wp.vec2d(float64_zero, -gravity_mag*(current_step+float64_one)/total_steps)

	# Calculate shape functions
	left_bottom_corner = x_particles[p] - wp.vec2d(lpx, lpy)
	left_bottom_corner_base_x = left_bottom_corner[0]*inv_dx + wp.float64(1e-8)
	left_bottom_corner_base_y = left_bottom_corner[1]*inv_dx + wp.float64(1e-8)
	left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
	left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))

	# GIMP
	GIMP_shape_function_and_gradient_components = get_GIMP_shape_function_and_gradient_2d(x_particles[p], dx, inv_dx, n_grid_x, n_nodes, lpx, lpy)
	wx0 = GIMP_shape_function_and_gradient_components[0]
	wy0 = GIMP_shape_function_and_gradient_components[1]
	wx1 = GIMP_shape_function_and_gradient_components[2]
	wy1 = GIMP_shape_function_and_gradient_components[3]
	wx2 = GIMP_shape_function_and_gradient_components[4]
	wy2 = GIMP_shape_function_and_gradient_components[5]
	grad_wx0 = GIMP_shape_function_and_gradient_components[6]
	grad_wy0 = GIMP_shape_function_and_gradient_components[7]
	grad_wx1 = GIMP_shape_function_and_gradient_components[8]
	grad_wy1 = GIMP_shape_function_and_gradient_components[9]
	grad_wx2 = GIMP_shape_function_and_gradient_components[10]
	grad_wy2 = GIMP_shape_function_and_gradient_components[11]

	w = wp.matrix(
		wx0, wy0,
		wx1, wy1,
		wx2, wy2,
		shape=(3,2)
	)

	grad_w = wp.matrix(
			 grad_wx0, grad_wy0,
			 grad_wx1, grad_wy1,
			 grad_wx2, grad_wy2,
			 shape=(3,2)
	)

	# Loop on dofs
	delta_u_GRAD = wp.mat22d(float64_zero, float64_zero,
							 float64_zero, float64_zero)

	for i in range(0, 3):
		for j in range(0, 3):
			weight = w[i][0] * w[j][1]
			weight_GRAD = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
			ix = left_bottom_corner_base_int[0] + i
			iy = left_bottom_corner_base_int[1] + j

			index_ij_x = ix + iy*(n_grid_x + wp.int(1))
			index_ij_y = index_ij_x + n_nodes

			node_increment_solution = wp.vec2d(increment_solution[index_ij_x], increment_solution[index_ij_y])

			delta_u_GRAD += wp.outer(weight_GRAD, node_increment_solution)


	incr_F = wp.identity(n=2, dtype=wp.float64) + delta_u_GRAD
	incr_F_inv = wp.inverse(incr_F)

	incr_F_3d = wp.mat33d(incr_F[0,0], incr_F[0,1], wp.float64(0.),
						  incr_F[1,0], incr_F[1,1], wp.float64(0.),
						  wp.float64(0.), wp.float64(0.), wp.float64(1.))

	deformation_gradient_total_new[p] = incr_F_3d @ deformation_gradient_total_old[p]
	particle_J = wp.determinant(deformation_gradient_total_new[p]) 


	# Get trial elastic b
	new_elastic_b_trial = incr_F_3d * left_Cauchy_Green_old[p] * wp.transpose(incr_F_3d)
	principal_stretches_square = wp.vec3d()
	principal_directions = wp.mat33d()
	wp.eig3(new_elastic_b_trial, principal_directions, principal_stretches_square)

	principal_direction_1 = wp.vec3d(principal_directions[0,0], principal_directions[1,0], principal_directions[2,0])
	principal_direction_2 = wp.vec3d(principal_directions[0,1], principal_directions[1,1], principal_directions[2,1])
	principal_direction_3 = wp.vec3d(principal_directions[0,2], principal_directions[1,2], principal_directions[2,2])

	e_trial = wp.mat33d(
			  wp.float64(0.5) * wp.log(wp.max(principal_stretches_square[0], wp.float64(1e-8))), wp.float64(0.), wp.float64(0.),
			  wp.float64(0.), wp.float64(0.5) * wp.log(wp.max(principal_stretches_square[1], wp.float64(1e-8))), wp.float64(0.),
			  wp.float64(0.), wp.float64(0.), wp.float64(0.5) * wp.log(wp.max(principal_stretches_square[2], wp.float64(1e-8)))
			  )

	# Elasticity
	e_real = e_trial

	# Get real Kirchhoff stress, Hencky elasticity
	e_trace = wp.trace(e_real)
	Kirchhoff_principal = lame_lambda*e_trace*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*e_real
	Kirchhoff_stress = Kirchhoff_principal[0,0] * wp.outer(principal_direction_1, principal_direction_1) + Kirchhoff_principal[1,1] * wp.outer(principal_direction_2, principal_direction_2) + Kirchhoff_principal[2,2] * wp.outer(principal_direction_3, principal_direction_3)

	Kirchhoff_stress_2d = wp.mat22d(
						  Kirchhoff_stress[0,0], Kirchhoff_stress[0,1],
						  Kirchhoff_stress[1,0], Kirchhoff_stress[1,1]
						  )

	particle_Cauchy_stress = Kirchhoff_stress_2d/particle_J
	particle_Cauchy_stress_array[p] = Kirchhoff_stress/particle_J

	# Reconstruct and save left Cauchy-Green strain tensor
	exp_2_e_real = wp.mat33d(
				   wp.exp(wp.float64(2.) * e_real[0,0]), float64_zero, float64_zero,
				   float64_zero, wp.exp(wp.float64(2.) * e_real[1,1]), float64_zero,
				   float64_zero, float64_zero, wp.exp(wp.float64(2.) * e_real[2,2])
				   )

	left_Cauchy_Green_new[p] = exp_2_e_real[0,0] * wp.outer(principal_direction_1, principal_direction_1) + exp_2_e_real[1,1] * wp.outer(principal_direction_2, principal_direction_2) + exp_2_e_real[2,2] * wp.outer(principal_direction_3, principal_direction_3)


	# Get new volume and density
	new_p_vol = p_vol * particle_J
	new_p_rho = p_rho / particle_J


	# Residual term
	for i in range(0, 3):
		for j in range(0, 3):
			weight = w[i][0] * w[j][1]
			weight_GRAD = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
			weight_grad = incr_F_inv @ weight_GRAD # NOTE here is incr_F_inv


			ix = left_bottom_corner_base_int[0] + i
			iy = left_bottom_corner_base_int[1] + j

			index_ij_x = ix + iy*(n_grid_x + wp.int(1))
			index_ij_y = index_ij_x + n_nodes

			rhs_value = (-weight_grad @ particle_Cauchy_stress + weight * new_p_rho * standard_gravity) * new_p_vol # Updated Lagrangian



			if (boundary_flag_array[index_ij_x]==False and activate_flag_array[index_ij_x]==True):
				wp.atomic_add(rhs, index_ij_x, rhs_value[0])

			if (boundary_flag_array[index_ij_y]==False and activate_flag_array[index_ij_y]==True):
				wp.atomic_add(rhs, index_ij_y, rhs_value[1])

@wp.kernel
def set_diagnal_component_for_boundary_and_deactivated_dofs_2d(boundary_flag_array: wp.array(dtype=wp.bool),
															   activate_flag_array: wp.array(dtype=wp.bool),
															   rows: wp.array(dtype=wp.int32),
															   cols: wp.array(dtype=wp.int32),
															   vals: wp.array(dtype=wp.float64),
															   n_matrix_size: wp.int32
															   ):
	dof_id = wp.tid()

	if boundary_flag_array[dof_id]==True or activate_flag_array[dof_id]==False:
		rows[50*n_matrix_size + dof_id] = dof_id
		cols[50*n_matrix_size + dof_id] = dof_id
		vals[50*n_matrix_size + dof_id] = wp.float64(1.0)


@wp.kernel
def from_increment_to_solution(increment_iteration: wp.array(dtype=wp.float64),
							   increment_solution: wp.array(dtype=wp.float64)):
	i = wp.tid()

	increment_solution[i] += increment_iteration[i]


@wp.kernel
def G2P_2d(GIMP_lp: wp.array(dtype=wp.vec2d),
		   x_particles: wp.array(dtype=wp.vec2d),
		   dx: wp.float64,
		   inv_dx: wp.float64,
		   n_grid_x: wp.int32,
		   n_nodes: wp.int32,
		   increment_solution: wp.array(dtype=wp.float64),
		   deformation_gradient_total_old: wp.array(dtype=wp.mat33d),
		   deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
		   left_Cauchy_Green_old: wp.array(dtype=wp.mat33d),
		   left_Cauchy_Green_new: wp.array(dtype=wp.mat33d)
		   ):
	p = wp.tid()

	float64_one = wp.float64(1.0)
	float64_zero = wp.float64(0.0)

	lpx = GIMP_lp[p][0]
	lpy = GIMP_lp[p][1]

	# Calculate shape functions
	left_bottom_corner = x_particles[p] - wp.vec2d(lpx, lpy)
	left_bottom_corner_base_x = left_bottom_corner[0]*inv_dx + wp.float64(1e-8)
	left_bottom_corner_base_y = left_bottom_corner[1]*inv_dx + wp.float64(1e-8)
	left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
	left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))


	# GIMP
	GIMP_shape_function_and_gradient_components = get_GIMP_shape_function_and_gradient_2d(x_particles[p], dx, inv_dx, n_grid_x, n_nodes, lpx, lpy)
	wx0 = GIMP_shape_function_and_gradient_components[0]
	wy0 = GIMP_shape_function_and_gradient_components[1]
	wx1 = GIMP_shape_function_and_gradient_components[2]
	wy1 = GIMP_shape_function_and_gradient_components[3]
	wx2 = GIMP_shape_function_and_gradient_components[4]
	wy2 = GIMP_shape_function_and_gradient_components[5]

	w = wp.matrix(
		wx0, wy0,
		wx1, wy1,
		wx2, wy2,
		shape=(3,2)
	)


	# Loop on dofs
	delta_u = wp.vec2d()

	for i in range(0, 3):
		for j in range(0, 3):
			weight = w[i][0] * w[j][1]
			ix = left_bottom_corner_base_int[0] + i
			iy = left_bottom_corner_base_int[1] + j

			index_ij_x = ix + iy*(n_grid_x + wp.int(1))
			index_ij_y = index_ij_x + n_nodes

			node_increment_solution = wp.vec2d(increment_solution[index_ij_x], increment_solution[index_ij_y])

			delta_u += weight * (node_increment_solution)

	

	# Set old to new
	deformation_gradient_total_old[p] = deformation_gradient_total_new[p]
	left_Cauchy_Green_old[p] = left_Cauchy_Green_new[p]

	x_particles[p] += delta_u