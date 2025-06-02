import sys
sys.path.append('..')

import warp as wp
import numpy as np


@wp.kernel
def initialize_GIMP_lp_2d(GIMP_lp: wp.array(dtype=wp.vec2d),
					   	  GIMP_lp_initial: wp.float64
					   	  ):
	p = wp.tid()

	GIMP_lp[p] = wp.vec2d(GIMP_lp_initial, GIMP_lp_initial)


@wp.func
def get_GIMP_shape_function_and_gradient_2d(xp: wp.vec2d,
											dx: wp.float64,
											inv_dx: wp.float64,
											n_grid_x: wp.int32,
											n_nodes: wp.int32,
											lpx: wp.float64, 
											lpy: wp.float64
											):

	left_bottom_corner = xp - wp.vec2d(lpx, lpy)
	left_bottom_corner_base_x = left_bottom_corner[0]*inv_dx + wp.float64(1e-8)
	left_bottom_corner_base_y = left_bottom_corner[1]*inv_dx + wp.float64(1e-8)
	left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
	left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))

	right_up_corner = xp + wp.vec2d(lpx, lpy)
	right_up_corner_base_x = right_up_corner[0]*inv_dx + wp.float64(1e-8)
	right_up_corner_base_y = right_up_corner[1]*inv_dx + wp.float64(1e-8)
	right_up_corner_base_int = wp.vector(wp.int(right_up_corner_base_x), wp.int(right_up_corner_base_y))
	right_up_corner_base = wp.vector(wp.float64(right_up_corner_base_int[0]), wp.float64(right_up_corner_base_int[1]))


	base_x = xp[0]*inv_dx + wp.float64(1e-8)
	base_y = xp[1]*inv_dx + wp.float64(1e-8)
	base_int = wp.vector(wp.int(base_x), wp.int(base_y))
	base = wp.vector(wp.float64(base_int[0]), wp.float64(base_int[1]))

	fx = xp * inv_dx - base


	# Declare shape function components and gradient components
	wx0 = wp.float64(0.0)
	wy0 = wp.float64(0.0)
	wx1 = wp.float64(0.0)
	wy1 = wp.float64(0.0)
	wx2 = wp.float64(0.0)
	wy2 = wp.float64(0.0)
	grad_wx0 = wp.float64(0.0)
	grad_wy0 = wp.float64(0.0)
	grad_wx1 = wp.float64(0.0)
	grad_wy1 = wp.float64(0.0)
	grad_wx2 = wp.float64(0.0)
	grad_wy2 = wp.float64(0.0)
	if right_up_corner_base_int[0]==left_bottom_corner_base_int[0]:
		wx0 = wp.float64(1.0) - fx[0]
		wx1 = fx[0]
		grad_wx0 = -wp.float64(1.0) * inv_dx
		grad_wx1 = inv_dx

		index_right_bottom_x = left_bottom_corner_base_int[0] + wp.int(2) + left_bottom_corner_base_int[1]*(n_grid_x + wp.int(1))
		index_right_bottom_y = index_right_bottom_x + n_nodes
		index_right_middle_x = left_bottom_corner_base_int[0] + wp.int(2) + (left_bottom_corner_base_int[1]+wp.int(1))*(n_grid_x + wp.int(1))
		index_right_middle_y = index_right_middle_x + n_nodes
		index_right_top_x    = left_bottom_corner_base_int[0] + wp.int(2) + (left_bottom_corner_base_int[1]+wp.int(2))*(n_grid_x + wp.int(1))
		index_right_top_y    = index_right_top_x + n_nodes
	else:
		# the particle will influence 2 elements in the x direction
		# shape function & gradient
		# Refer to section 2.4.2 of [Coombs, et al.]
		x_0 = (left_bottom_corner_base[0]) * dx
		wx0 = wp.pow((dx+lpx-wp.abs(xp[0]-x_0)), wp.float64(2.0))/(wp.float64(4)*dx*lpx)
		grad_wx0 = -(dx+lpx-wp.abs(xp[0]-x_0))/(wp.float64(2)*dx*lpx) * (xp[0]-x_0)/wp.abs(xp[0]-x_0)

		x_1 = (left_bottom_corner_base[0] + wp.float64(1)) * dx
		wx1 = wp.float64(1.0) - (wp.pow((xp[0]-x_1), wp.float64(2.0)) + wp.pow(lpx, wp.float64(2.0)))/(wp.float64(2)*dx*lpx)
		grad_wx1 = -(xp[0]-x_1)/(dx*lpx)

		x_2 = (left_bottom_corner_base[0] + wp.float64(2)) * dx
		wx2 = wp.pow((dx+lpx-wp.abs(xp[0]-x_2)), wp.float64(2.0))/(wp.float64(4)*dx*lpx)
		grad_wx2 = -(dx+lpx-wp.abs(xp[0]-x_2))/(wp.float64(2)*dx*lpx) * (xp[0]-x_2)/wp.abs(xp[0]-x_2)


	# get shape function y
	if right_up_corner_base_int[1]==left_bottom_corner_base_int[1]:
		wy0 = wp.float64(1.0) - fx[1]
		wy1 = fx[1]
		grad_wy0 = -wp.float64(1.0) * inv_dx
		grad_wy1 = inv_dx

		index_top_left_x = left_bottom_corner_base_int[0] + (left_bottom_corner_base_int[1]+wp.int(2))*(n_grid_x + wp.int(1))
		index_top_left_y = index_top_left_x + n_nodes
		index_top_middle_x = left_bottom_corner_base_int[0] + wp.int(1) + (left_bottom_corner_base_int[1]+wp.int(2))*(n_grid_x + wp.int(1))
		index_top_middle_y = index_top_middle_x + n_nodes
		index_top_right_x = left_bottom_corner_base_int[0] + wp.int(2) + (left_bottom_corner_base_int[1]+wp.int(2))*(n_grid_x + wp.int(1))
		index_top_right_y = index_top_right_x + n_nodes
	else:
		# the particle will influence 2 elements in the y direction
		# shape function & gradient
		y_0 = (left_bottom_corner_base[1]) * dx
		wy0 = wp.pow((dx+lpy-wp.abs(xp[1]-y_0)), wp.float64(2.0))/(wp.float64(4)*dx*lpy)
		grad_wy0 = -(dx+lpy-wp.abs(xp[1]-y_0))/(wp.float64(2.0)*dx*lpy) * (xp[1]-y_0)/wp.abs(xp[1]-y_0)

		y_1 = (left_bottom_corner_base[1] + wp.float64(1)) * dx
		wy1 = wp.float64(1.0) - (wp.pow((xp[1]-y_1), wp.float64(2.0)) + wp.pow(lpy, wp.float64(2.0)))/(wp.float64(2)*dx*lpy)
		grad_wy1 = -(xp[1]-y_1)/(dx*lpy)

		y_2 = (left_bottom_corner_base[1] + wp.float64(2)) * dx
		wy2 = wp.pow((dx+lpy-wp.abs(xp[1]-y_2)), wp.float64(2.0))/(wp.float64(4)*dx*lpy)
		grad_wy2 = -(dx+lpy-wp.abs(xp[1]-y_2))/(wp.float64(2.0)*dx*lpy) * (xp[1]-y_2)/wp.abs(xp[1]-y_2)



	return wp.vector(wx0, wy0, wx1, wy1, wx2, wy2, grad_wx0, grad_wy0, grad_wx1, grad_wy1, grad_wx2, grad_wy2)


@wp.kernel
def update_GIMP_lp_2d(deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
					  GIMP_lp: wp.array(dtype=wp.vec2d),
					  GIMP_lp_initial: wp.float64,
					  x_particles: wp.array(dtype=wp.vec2d),
					  min_x: wp.float64,
					  max_x: wp.float64,
					  min_y: wp.float64,
					  max_y: wp.float64
					  ):
	p = wp.tid()

	# Refer to Eq. (38), (39) of Charlton et al. 2017
	this_particle_F = deformation_gradient_total_new[p]
	U00 = wp.sqrt(this_particle_F[0,0]*this_particle_F[0,0] + this_particle_F[1,0]*this_particle_F[1,0] + this_particle_F[2,0]*this_particle_F[2,0])
	U11 = wp.sqrt(this_particle_F[0,1]*this_particle_F[0,1] + this_particle_F[1,1]*this_particle_F[1,1] + this_particle_F[2,1]*this_particle_F[2,1])
	U22 = wp.sqrt(this_particle_F[0,2]*this_particle_F[0,2] + this_particle_F[1,2]*this_particle_F[1,2] + this_particle_F[2,2]*this_particle_F[2,2])

	lpx_updated = GIMP_lp_initial*U00
	lpy_updated = GIMP_lp_initial*U11

	if x_particles[p][0]-lpx_updated < min_x:
		lpx_updated = x_particles[p][0] - min_x - wp.float64(1e-6)

	if x_particles[p][0]+lpx_updated > max_x:
		lpx_updated = max_x - x_particles[p][0] - wp.float64(1e-6)

	if x_particles[p][1]-lpy_updated < min_y:
		lpy_updated = x_particles[p][1] - min_y - wp.float64(1e-6)

	if x_particles[p][1]+lpy_updated > max_y:
		lpy_updated = max_y - x_particles[p][1] - wp.float64(1e-6)


	GIMP_lp[p] = wp.vec2d(lpx_updated, lpy_updated)