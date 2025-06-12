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

@wp.kernel
def initialize_GIMP_lp_3d(GIMP_lp: wp.array(dtype=wp.vec3d),
						  GIMP_lp_initial: wp.float64
						  ):
	p = wp.tid()

	GIMP_lp[p] = wp.vec3d(GIMP_lp_initial, GIMP_lp_initial, GIMP_lp_initial)


@wp.func
def get_GIMP_shape_function_and_gradient_2d(xp: wp.vec2d,
											dx: wp.float64,
											inv_dx: wp.float64,
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


@wp.func
def get_GIMP_shape_function_and_gradient_avg_2d(xp: wp.vec2d,
												dx: wp.float64,
												inv_dx: wp.float64,
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
		wx0 = wp.float64(0.5)
		wx1 = wp.float64(0.5)
	else:
		# the particle will influence 2 elements in the x direction
		# shape function & gradient
		# Refer to section 2.4.2 of [Coombs, et al.]
		x_0 = (left_bottom_corner_base[0]) * dx
		wx0 = (dx+lpx-wp.abs(xp[0]-x_0))/(wp.float64(4.0)*lpx)
		grad_wx0 = wp.float64(-1.0)/(wp.float64(4.0)*lpx) * wp.sign(xp[0]-x_0)

		wx1 = wp.float64(0.5) 

		x_2 = (left_bottom_corner_base[0] + wp.float64(2)) * dx
		wx2 = (dx+lpx-wp.abs(xp[0]-x_2))/(wp.float64(4.0)*lpx)
		grad_wx2 = wp.float64(-1.0)/(wp.float64(4.0)*lpx) * wp.sign(xp[0]-x_2)


	# get shape function y
	if right_up_corner_base_int[1]==left_bottom_corner_base_int[1]:
		wy0 = wp.float64(0.5)
		wy1 = wp.float64(0.5)
	else:
		# the particle will influence 2 elements in the y direction
		# shape function & gradient
		y_0 = (left_bottom_corner_base[1]) * dx
		wy0 = (dx+lpy-wp.abs(xp[1]-y_0))/(wp.float64(4.0)*lpy)
		grad_wy0 = wp.float64(-1.0)/(wp.float64(4.0)*lpy) * wp.sign(xp[1]-y_0)

		wy1 = wp.float64(0.5)

		y_2 = (left_bottom_corner_base[1] + wp.float64(2)) * dx
		wy2 = (dx+lpy-wp.abs(xp[1]-y_2))/(wp.float64(4.0)*lpy)
		grad_wy2 = wp.float64(-1.0)/(wp.float64(4.0)*lpy) * wp.sign(xp[1]-y_2)



	return wp.vector(wx0, wy0, wx1, wy1, wx2, wy2, grad_wx0, grad_wy0, grad_wx1, grad_wy1, grad_wx2, grad_wy2)



@wp.func
def get_GIMP_shape_function_and_gradient_3d(xp: wp.vec3d,
											dx: wp.float64,
											dy: wp.float64,
											dz: wp.float64,
											inv_dx: wp.float64,
											inv_dy: wp.float64,
											inv_dz: wp.float64,
											lpx: wp.float64, 
											lpy: wp.float64,
											lpz: wp.float64
											):
	bottom_corner = xp - wp.vec3d(lpx, lpy, lpz)
	bottom_corner_base_x = bottom_corner[0]*inv_dx + wp.float64(1e-8)
	bottom_corner_base_y = bottom_corner[1]*inv_dy + wp.float64(1e-8)
	bottom_corner_base_z = bottom_corner[2]*inv_dz + wp.float64(1e-8)
	bottom_corner_base_int = wp.vector(wp.int(bottom_corner_base_x), wp.int(bottom_corner_base_y), wp.int(bottom_corner_base_z))
	bottom_corner_base = wp.vector(wp.float64(bottom_corner_base_int[0]), wp.float64(bottom_corner_base_int[1]), wp.float64(bottom_corner_base_int[2]))

	up_corner = xp + wp.vec3d(lpx, lpy, lpz)
	up_corner_base_x = up_corner[0]*inv_dx + wp.float64(1e-8)
	up_corner_base_y = up_corner[1]*inv_dy + wp.float64(1e-8)
	up_corner_base_z = up_corner[2]*inv_dz + wp.float64(1e-8)
	up_corner_base_int = wp.vec3i(wp.int(up_corner_base_x), wp.int(up_corner_base_y), wp.int(up_corner_base_z))
	up_corner_base = wp.vec3d(wp.float64(up_corner_base_int[0]), wp.float64(up_corner_base_int[1]), wp.float64(up_corner_base_int[2]))

	base_x = xp[0]*inv_dx + wp.float64(1e-8)
	base_y = xp[1]*inv_dy + wp.float64(1e-8)
	base_z = xp[2]*inv_dz + wp.float64(1e-8)
	base_int = wp.vec3i(wp.int(base_x), wp.int(base_y), wp.int(base_z))
	base = wp.vec3d(wp.float64(base_int[0]), wp.float64(base_int[1]), wp.float64(base_int[2]))

	fx = wp.vec3d(xp[0]*inv_dx - base[0], xp[1]*inv_dy - base[1], xp[2]*inv_dz - base[2])

	# Declare shape function components and gradient components
	wx0 = wp.float64(0.0)
	wy0 = wp.float64(0.0)
	wz0 = wp.float64(0.0)
	wx1 = wp.float64(0.0)
	wy1 = wp.float64(0.0)
	wz1 = wp.float64(0.0)
	wx2 = wp.float64(0.0)
	wy2 = wp.float64(0.0)
	wz2 = wp.float64(0.0)
	grad_wx0 = wp.float64(0.0)
	grad_wy0 = wp.float64(0.0)
	grad_wz0 = wp.float64(0.0)
	grad_wx1 = wp.float64(0.0)
	grad_wy1 = wp.float64(0.0)
	grad_wz1 = wp.float64(0.0)
	grad_wx2 = wp.float64(0.0)
	grad_wy2 = wp.float64(0.0)
	grad_wz2 = wp.float64(0.0)

	# get shape function x
	if up_corner_base_int[0]==bottom_corner_base_int[0]:
		wx0 = wp.float64(1.0) - fx[0]
		wx1 = fx[0]
		grad_wx0 = -wp.float64(1.0) * inv_dx
		grad_wx1 = inv_dx
	else:
		# the particle will influence 2 elements in the x direction
		# shape function & gradient
		# Refer to section 2.4.2 of [Coombs, et al.]
		x_0 = (bottom_corner_base[0]) * dx
		wx0 = wp.pow((dx+lpx-wp.abs(xp[0]-x_0)), wp.float64(2.0))/(wp.float64(4)*dx*lpx)
		grad_wx0 = -(dx+lpx-wp.abs(xp[0]-x_0))/(wp.float64(2)*dx*lpx) * (xp[0]-x_0)/wp.abs(xp[0]-x_0)

		x_1 = (bottom_corner_base[0] + wp.float64(1)) * dx
		wx1 = wp.float64(1.0) - (wp.pow((xp[0]-x_1), wp.float64(2.0)) + wp.pow(lpx, wp.float64(2.0)))/(wp.float64(2)*dx*lpx)
		grad_wx1 = -(xp[0]-x_1)/(dx*lpx)

		x_2 = (bottom_corner_base[0] + wp.float64(2)) * dx
		wx2 = wp.pow((dx+lpx-wp.abs(xp[0]-x_2)), wp.float64(2.0))/(wp.float64(4)*dx*lpx)
		grad_wx2 = -(dx+lpx-wp.abs(xp[0]-x_2))/(wp.float64(2)*dx*lpx) * (xp[0]-x_2)/wp.abs(xp[0]-x_2)

	# get shape function y
	if up_corner_base_int[1]==bottom_corner_base_int[1]:
		wy0 = wp.float64(1.0) - fx[1]
		wy1 = fx[1]
		grad_wy0 = -wp.float64(1.0) * inv_dy
		grad_wy1 = inv_dy
	else:
		# the particle will influence 2 elements in the y direction
		# shape function & gradient
		y_0 = (bottom_corner_base[1]) * dy
		wy0 = wp.pow((dy+lpy-wp.abs(xp[1]-y_0)), wp.float64(2.0))/(wp.float64(4)*dy*lpy)
		grad_wy0 = -(dy+lpy-wp.abs(xp[1]-y_0))/(wp.float64(2.0)*dy*lpy) * (xp[1]-y_0)/wp.abs(xp[1]-y_0)

		y_1 = (bottom_corner_base[1] + wp.float64(1)) * dy
		wy1 = wp.float64(1.0) - (wp.pow((xp[1]-y_1), wp.float64(2.0)) + wp.pow(lpy, wp.float64(2.0)))/(wp.float64(2)*dy*lpy)
		grad_wy1 = -(xp[1]-y_1)/(dy*lpy)

		y_2 = (bottom_corner_base[1] + wp.float64(2)) * dy
		wy2 = wp.pow((dy+lpy-wp.abs(xp[1]-y_2)), wp.float64(2.0))/(wp.float64(4)*dy*lpy)
		grad_wy2 = -(dy+lpy-wp.abs(xp[1]-y_2))/(wp.float64(2.0)*dy*lpy) * (xp[1]-y_2)/wp.abs(xp[1]-y_2)

	# get shape function z
	if up_corner_base_int[2]==bottom_corner_base_int[2]:
		wz0 = wp.float64(1.0) - fx[2]
		wz1 = fx[2]
		grad_wz0 = -wp.float64(1.0) * inv_dz
		grad_wz1 = inv_dz
	else:
		# the particle will influence 2 elements in the z direction
		# shape function & gradient
		z_0 = (bottom_corner_base[2]) * dz
		wz0 = wp.pow((dz+lpz-wp.abs(xp[2]-z_0)), wp.float64(2.0))/(wp.float64(4)*dz*lpz)
		grad_wz0 = -(dz+lpz-wp.abs(xp[2]-z_0))/(wp.float64(2.0)*dz*lpz) * (xp[2]-z_0)/wp.abs(xp[2]-z_0)

		z_1 = (bottom_corner_base[2] + wp.float64(1)) * dz
		wz1 = wp.float64(1.0) - (wp.pow((xp[2]-z_1), wp.float64(2.0)) + wp.pow(lpz, wp.float64(2.0)))/(wp.float64(2)*dz*lpz)
		grad_wz1 = -(xp[2]-z_1)/(dz*lpz)

		z_2 = (bottom_corner_base[2] + wp.float64(2)) * dz
		wz2 = wp.pow((dz+lpz-wp.abs(xp[2]-z_2)), wp.float64(2.0))/(wp.float64(4)*dz*lpz)
		grad_wz2 = -(dz+lpz-wp.abs(xp[2]-z_2))/(wp.float64(2.0)*dz*lpz) * (xp[2]-z_2)/wp.abs(xp[2]-z_2)

	return wp.vector(wx0, wy0, wz0, wx1, wy1, wz1, wx2, wy2, wz2, grad_wx0, grad_wy0, grad_wz0, grad_wx1, grad_wy1, grad_wz1, grad_wx2, grad_wy2, grad_wz2)


@wp.func
def get_GIMP_shape_function_and_gradient_avg_3d(xp: wp.vec3d,
												dx: wp.float64,
												dy: wp.float64,
												dz: wp.float64,
												inv_dx: wp.float64,
												inv_dy: wp.float64,
												inv_dz: wp.float64,
												n_grid_x: wp.int32,
												n_grid_y: wp.int32,
												lpx: wp.float64,
												lpy: wp.float64,
												lpz: wp.float64
												):
	bottom_corner = xp - wp.vec3d(lpx, lpy, lpz)
	bottom_corner_base_x = bottom_corner[0]*inv_dx + wp.float64(1e-8)
	bottom_corner_base_y = bottom_corner[1]*inv_dy + wp.float64(1e-8)
	bottom_corner_base_z = bottom_corner[2]*inv_dz + wp.float64(1e-8)
	bottom_corner_base_int = wp.vec3i(wp.int(bottom_corner_base_x), wp.int(bottom_corner_base_y), wp.int(bottom_corner_base_z))
	bottom_corner_base = wp.vec3d(wp.float64(bottom_corner_base_int[0]), wp.float64(bottom_corner_base_int[1]), wp.float64(bottom_corner_base_int[2]))

	up_corner = xp + wp.vec3d(lpx, lpy, lpz)
	up_corner_base_x = up_corner[0]*inv_dx + wp.float64(1e-8)
	up_corner_base_y = up_corner[1]*inv_dy + wp.float64(1e-8)
	up_corner_base_z = up_corner[2]*inv_dz + wp.float64(1e-8)
	up_corner_base_int = wp.vec3i(wp.int(up_corner_base_x), wp.int(up_corner_base_y), wp.int(up_corner_base_z))
	up_corner_base = wp.vec3d(wp.float64(up_corner_base_int[0]), wp.float64(up_corner_base_int[1]), wp.float64(up_corner_base_int[2]))

	base_x = xp[0]*inv_dx + wp.float64(1e-8)
	base_y = xp[1]*inv_dy + wp.float64(1e-8)
	base_z = xp[2]*inv_dz + wp.float64(1e-8)
	base_int = wp.vec3i(wp.int(base_x), wp.int(base_y), wp.int(base_z))
	base = wp.vec3d(wp.float64(base_int[0]), wp.float64(base_int[1]), wp.float64(base_int[2]))

	fx = wp.vec3d(xp[0]*inv_dx - base[0], xp[1]*inv_dy - base[1], xp[2]*inv_dz - base[2])

	# Declare shape function components and gradient components
	wx0 = wp.float64(0.0)
	wy0 = wp.float64(0.0)
	wz0 = wp.float64(0.0)
	wx1 = wp.float64(0.0)
	wy1 = wp.float64(0.0)
	wz1 = wp.float64(0.0)
	wx2 = wp.float64(0.0)
	wy2 = wp.float64(0.0)
	wz2 = wp.float64(0.0)
	grad_wx0 = wp.float64(0.0)
	grad_wy0 = wp.float64(0.0)
	grad_wz0 = wp.float64(0.0)
	grad_wx1 = wp.float64(0.0)
	grad_wy1 = wp.float64(0.0)
	grad_wz1 = wp.float64(0.0)
	grad_wx2 = wp.float64(0.0)
	grad_wy2 = wp.float64(0.0)
	grad_wz2 = wp.float64(0.0)

	# get shape function x
	if up_corner_base_int[0]==bottom_corner_base_int[0]:
		wx0 = wp.float64(0.5)
		wx1 = wp.float64(0.5)
	else:
		# the particle will influence 2 elements in the x direction
		# shape function & gradient
		x_0 = (bottom_corner_base[0]) * dx
		wx0 = (dx+lpx-wp.abs(xp[0]-x_0))/(wp.float64(4.0)*lpx)
		grad_wx0 = wp.float64(-1.0)/(wp.float64(4.0)*lpx) * wp.sign(xp[0]-x_0)

		wx1 = wp.float64(0.5) 

		x_2 = (bottom_corner_base[0] + wp.float64(2)) * dx
		wx2 = (dx+lpx-wp.abs(xp[0]-x_2))/(wp.float64(4.0)*lpx)
		grad_wx2 = wp.float64(-1.0)/(wp.float64(4.0)*lpx) * wp.sign(xp[0]-x_2)

	# get shape function y
	if up_corner_base_int[1]==bottom_corner_base_int[1]:
		wy0 = wp.float64(0.5)
		wy1 = wp.float64(0.5)
	else:
		# the particle will influence 2 elements in the y direction
		# shape function & gradient
		y_0 = (bottom_corner_base[1]) * dy
		wy0 = (dy+lpy-wp.abs(xp[1]-y_0))/(wp.float64(4.0)*lpy)
		grad_wy0 = wp.float64(-1.0)/(wp.float64(4.0)*lpy) * wp.sign(xp[1]-y_0)

		wy1 = wp.float64(0.5) 

		y_2 = (bottom_corner_base[1] + wp.float64(2)) * dy
		wy2 = (dy+lpy-wp.abs(xp[1]-y_2))/(wp.float64(4.0)*lpy)
		grad_wy2 = wp.float64(-1.0)/(wp.float64(4.0)*lpy) * wp.sign(xp[1]-y_2)

	# get shape function z
	if up_corner_base_int[2]==bottom_corner_base_int[2]:
		wz0 = wp.float64(0.5)
		wz1 = wp.float64(0.5)
	else:
		# the particle will influence 2 elements in the z direction
		# shape function & gradient
		z_0 = (bottom_corner_base[2]) * dz
		wz0 = (dz+lpz-wp.abs(xp[2]-z_0))/(wp.float64(4.0)*lpz)
		grad_wz0 = wp.float64(-1.0)/(wp.float64(4.0)*lpz) * wp.sign(xp[2]-z_0) 

		wz1 = wp.float64(0.5)

		z_2 = (bottom_corner_base[2] + wp.float64(2)) * dz
		wz2 = (dz+lpz-wp.abs(xp[2]-z_2))/(wp.float64(4.0)*lpz)
		grad_wz2 = wp.float64(-1.0)/(wp.float64(4.0)*lpz) * wp.sign(xp[2]-z_2)

	return wp.vector(wx0, wy0, wz0, wx1, wy1, wz1, wx2, wy2, wz2, grad_wx0, grad_wy0, grad_wz0, grad_wx1, grad_wy1, grad_wz1, grad_wx2, grad_wy2, grad_wz2)


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


@wp.kernel
def update_GIMP_lp_3d(deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
					  GIMP_lp: wp.array(dtype=wp.vec3d),
					  GIMP_lp_initial: wp.float64,
					  x_particles: wp.array(dtype=wp.vec3d),
					  min_x: wp.float64,
					  max_x: wp.float64,
					  min_y: wp.float64,
					  max_y: wp.float64,
					  min_z: wp.float64,
					  max_z: wp.float64
					  ):
	p = wp.tid()

	# Refer to Eq. (38), (39) of Charlton et al. 2017
	this_particle_F = deformation_gradient_total_new[p]
	U00 = wp.sqrt(this_particle_F[0,0]*this_particle_F[0,0] + this_particle_F[1,0]*this_particle_F[1,0] + this_particle_F[2,0]*this_particle_F[2,0])
	U11 = wp.sqrt(this_particle_F[0,1]*this_particle_F[0,1] + this_particle_F[1,1]*this_particle_F[1,1] + this_particle_F[2,1]*this_particle_F[2,1])
	U22 = wp.sqrt(this_particle_F[0,2]*this_particle_F[0,2] + this_particle_F[1,2]*this_particle_F[1,2] + this_particle_F[2,2]*this_particle_F[2,2])

	lpx_updated = GIMP_lp_initial*U00
	lpy_updated = GIMP_lp_initial*U11
	lpz_updated = GIMP_lp_initial*U22

	if x_particles[p][0]-lpx_updated < min_x:
		lpx_updated = x_particles[p][0] - min_x - wp.float64(1e-6)

	if x_particles[p][0]+lpx_updated > max_x:
		lpx_updated = max_x - x_particles[p][0] - wp.float64(1e-6)

	if x_particles[p][1]-lpy_updated < min_y:
		lpy_updated = x_particles[p][1] - min_y - wp.float64(1e-6)

	if x_particles[p][1]+lpy_updated > max_y:
		lpy_updated = max_y - x_particles[p][1] - wp.float64(1e-6)

	if x_particles[p][2]-lpz_updated < min_z:
		lpz_updated = x_particles[p][2] - min_z - wp.float64(1e-6)

	if x_particles[p][2]+lpz_updated > max_z:
		lpz_updated = max_z - x_particles[p][2] - wp.float64(1e-6)

	GIMP_lp[p] = wp.vec3d(lpx_updated, lpy_updated, lpz_updated)

