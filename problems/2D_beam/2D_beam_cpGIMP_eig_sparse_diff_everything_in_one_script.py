import warp as wp
import numpy as np
import meshio

import warp.sparse as wps

import warp.optim.linear


from scipy import sparse
from scipy.sparse.linalg import spsolve



import pyamg
from pyamg import smoothed_aggregation_solver
from pyamg.krylov import bicgstab



import time



# Implicit MPM solver for 2D beam using Warp
# Contact: Yidong Zhao (ydzhao94@gmail.com)

# Refer to an implicit FEM solver (by Xuan Li) for the usage of sparse matrix: https://github.com/xuan-li/warp_FEM/tree/main


wp.init()


n_particles = 5760 #92160 #51840 #23040 #12960 # #1440
n_grid_x = 60 #240 #180 #120 #90 # #30
n_grid_y = 60 #240 #180 #120 #90 # #30
grid_size = (n_grid_x, n_grid_y)

max_x = 15.0
dx = max_x/n_grid_x
inv_dx = float(n_grid_x/max_x)

PPD = 6 # particle per direction

youngs_modulus = 12000.0 # kPa
poisson_ratio = 0.2
lame_mu = youngs_modulus / (2.0*(1.0+poisson_ratio))
lame_lambda = youngs_modulus*poisson_ratio / ((1.0+poisson_ratio) * (1.0-2.0*poisson_ratio))

p_vol = (dx/PPD)**2 
p_rho = 1.0 # t/m^3
p_mass = p_vol * p_rho

GIMP_lp_initial = dx/PPD/2.0 - 1e-6

l0 = 10.0
d0 = 1.0





# To check whether a specific dof index is activated or at the Dirichlet boundary
@wp.struct
class DofStruct:
    activated_flag_array: wp.array(dtype=wp.bool)
    boundary_flag_array: wp.array(dtype=wp.bool)


# initialization
@wp.kernel
def initialization(deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
                   deformation_gradient_total_old: wp.array(dtype=wp.mat33d),
                   left_Cauchy_Green_new: wp.array(dtype=wp.mat33d),
                   left_Cauchy_Green_old: wp.array(dtype=wp.mat33d),
                   x_particles: wp.array(dtype=wp.vec2d),
                   v_particles: wp.array(dtype=wp.vec2d)):
    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)

    identity_matrix = wp.matrix(
                      float64_one + wp.float64(1e-7), float64_zero, float64_zero,
                      float64_zero, float64_one - wp.float64(1e-6), float64_zero,
                      float64_zero, float64_zero, float64_one,
                      shape=(3,3)
        )

    deformation_gradient_total_new[p] = identity_matrix
    deformation_gradient_total_old[p] = identity_matrix
    left_Cauchy_Green_new[p] = identity_matrix
    left_Cauchy_Green_old[p] = identity_matrix

    v_particles[p] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))

@wp.kernel
def initialize_GIMP_lp(GIMP_lp: wp.array(dtype=wp.vec2d),
                       GIMP_lp_initial: wp.float64):
    p = wp.tid()

    GIMP_lp[p] = wp.vec2d(GIMP_lp_initial, GIMP_lp_initial)



@wp.kernel
def set_external_force_flag(x_particles: wp.array(dtype=wp.vec2d),
                            particle_external_flag_array: wp.array(dtype=wp.bool),
                            start_x: wp.float64,
                            start_y: wp.float64,
                            dx: wp.float64,
                            PPD: wp.float64,
                            l0: wp.float64,
                            d0: wp.float64):
    p = wp.tid()

    this_particle = x_particles[p]

    if this_particle[0]>start_x+l0-dx/PPD:
        if this_particle[1]>l0-d0/wp.float64(2.)-dx/PPD and this_particle[1]<l0-d0/wp.float64(2.)+dx/PPD:
            particle_external_flag_array[p] = True



# Clear grid quantities
@wp.kernel
def clear_grid_quantities(grid_v: wp.array(dtype=wp.vec2d, ndim=2),
        grid_old_v: wp.array(dtype=wp.vec2d, ndim=2),
        grid_f: wp.array(dtype=wp.vec2d, ndim=2),
        grid_m: wp.array(dtype=wp.float64, ndim=2)):

    float64_zero = wp.float64(0.0)

    i, j = wp.tid()
    grid_v[i, j] = wp.vec2d(float64_zero, float64_zero)
    grid_old_v[i, j] = wp.vec2d(float64_zero, float64_zero)
    grid_f[i, j] = wp.vec2d(float64_zero, float64_zero)
    grid_m[i, j] = float64_zero




# Set boundary dofs
@wp.kernel
def set_boundary_dofs(dofStruct: DofStruct,
                      n_grid_x: wp.int32,
                      n_grid_y: wp.int32,
                      n_nodes: wp.int32,
                      end_y: wp.float64,
                      d0: wp.float64,
                      dx: wp.float64):
    
    node_idx, node_idy = wp.tid()
    dof_x = node_idx + node_idy*(n_grid_x + 1)
    dof_y = dof_x + n_nodes

    if node_idx<=1:
        dofStruct.boundary_flag_array[dof_x] = True

    if node_idx<=1 and node_idy==wp.int((end_y-d0/wp.float64(2.)+wp.float64(0.5)*dx)/dx):
        dofStruct.boundary_flag_array[dof_y] = True


# P2G
@wp.kernel
def P2G(x_particles: wp.array(dtype=wp.vec2d),
        inv_dx: wp.float64,
        dx: wp.float64,
        n_grid_x: wp.int32,
        n_nodes: wp.int32,
        dofStruct: DofStruct,
        GIMP_lp: wp.array(dtype=wp.vec2d)
        ):
    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    lpx = GIMP_lp[p][0]
    lpy = GIMP_lp[p][1]



    # GIMP
    left_bottom_corner = x_particles[p] - wp.vec2d(lpx, lpy)
    left_bottom_corner_base_x = left_bottom_corner[0]*inv_dx + wp.float64(1e-8)
    left_bottom_corner_base_y = left_bottom_corner[1]*inv_dx + wp.float64(1e-8)
    left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
    left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))

    left_up_corner = x_particles[p] + wp.vec2d(-lpx, lpy)
    left_up_corner_base_x = left_up_corner[0]*inv_dx + wp.float64(1e-8)
    left_up_corner_base_y = left_up_corner[1]*inv_dx + wp.float64(1e-8)
    left_up_corner_base_int = wp.vector(wp.int(left_up_corner_base_x), wp.int(left_up_corner_base_y))
    left_up_corner_base = wp.vector(wp.float64(left_up_corner_base_int[0]), wp.float64(left_up_corner_base_int[1]))

    right_bottom_corner = x_particles[p] + wp.vec2d(lpx, -lpy)
    right_bottom_corner_base_x = right_bottom_corner[0]*inv_dx + wp.float64(1e-8)
    right_bottom_corner_base_y = right_bottom_corner[1]*inv_dx + wp.float64(1e-8)
    right_bottom_corner_base_int = wp.vector(wp.int(right_bottom_corner_base_x), wp.int(right_bottom_corner_base_y))
    right_bottom_corner_base = wp.vector(wp.float64(right_bottom_corner_base_int[0]), wp.float64(right_bottom_corner_base_int[1]))

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

            dofStruct.activated_flag_array[index_ij_x] = True
            dofStruct.activated_flag_array[index_ij_y] = True


@wp.func
def get_GIMP_shape_function_and_gradient(xp: wp.vec2d,
                                         dx: wp.float64,
                                         inv_dx: wp.float64,
                                         dofStruct: DofStruct,
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

    if wx0<wp.float64(1e-7):
        wx0 = wp.float64(0.0)
        grad_wx0 = wp.float64(0.0)

    if wx1<wp.float64(1e-7):
        wx1 = wp.float64(0.0)
        grad_wx1 = wp.float64(0.0)

    if wx2<wp.float64(1e-7):
        wx2 = wp.float64(0.0)
        grad_wx2 = wp.float64(0.0)


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

    if wy0<wp.float64(1e-7):
        wy0 = wp.float64(0.0)
        grad_wy0 = wp.float64(0.0)

    if wy1<wp.float64(1e-7):
        wy1 = wp.float64(0.0)
        grad_wy1 = wp.float64(0.0)

    if wy2<wp.float64(1e-7):
        wy2 = wp.float64(0.0)
        grad_wy2 = wp.float64(0.0)


    return wp.vector(wx0, wy0, wx1, wy1, wx2, wy2, grad_wx0, grad_wy0, grad_wx1, grad_wy1, grad_wx2, grad_wy2)




# Assemble residual with tape
@wp.kernel
def assemble_residual(x_particles: wp.array(dtype=wp.vec2d),
                      inv_dx: wp.float64,
                      dx: wp.float64,
                      rhs: wp.array(dtype=wp.float64),
                      increment_solution: wp.array(dtype=wp.float64),
                      p_vol: wp.float64,
                      p_rho: wp.float64,
                      lame_lambda: wp.float64,
                      lame_mu: wp.float64,
                      deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
                      deformation_gradient_total_old: wp.array(dtype=wp.mat33d),
                      left_Cauchy_Green_new: wp.array(dtype=wp.mat33d),
                      left_Cauchy_Green_old: wp.array(dtype=wp.mat33d),
                      n_grid_x: wp.int32,
                      n_nodes: wp.int32,
                      dofStruct: DofStruct,
                      step: wp.float64,
                      particle_Cauchy_stress_array: wp.array(dtype=wp.mat33d),
                      particle_external_flag_array: wp.array(dtype=wp.bool),
                      GIMP_lp: wp.array(dtype=wp.vec2d)
                      ):
    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    standard_gravity = wp.vec2d(float64_zero, float64_zero)

    lpx = GIMP_lp[p][0]
    lpy = GIMP_lp[p][1]



    # Calculate shape functions
    left_bottom_corner = x_particles[p] - wp.vec2d(lpx, lpy)
    left_bottom_corner_base_x = left_bottom_corner[0]*inv_dx + wp.float64(1e-8)
    left_bottom_corner_base_y = left_bottom_corner[1]*inv_dx + wp.float64(1e-8)
    left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
    left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))


    # GIMP
    GIMP_shape_function_and_gradient_components = get_GIMP_shape_function_and_gradient(x_particles[p], dx, inv_dx, dofStruct, n_grid_x, n_nodes, lpx, lpy)
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
    delta_u_GRAD = wp.matrix(float64_zero, float64_zero,
                             float64_zero, float64_zero, shape=(2,2))
    for i in range(0, 3):
        for j in range(0, 3):
            weight = w[i][0] * w[j][1]
            weight_grad = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
            ix = left_bottom_corner_base_int[0] + i
            iy = left_bottom_corner_base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_y = index_ij_x + n_nodes

            node_increment_solution = wp.vec2d(increment_solution[index_ij_x], increment_solution[index_ij_y])

            delta_u_GRAD += wp.outer(weight_grad, node_increment_solution)



    incr_F = wp.identity(n=2, dtype=wp.float64) + delta_u_GRAD
    incr_F_inv = wp.inverse(incr_F)
    
    incr_F_3d = wp.mat33d(incr_F[0,0], incr_F[0,1], wp.float64(0.),
                          incr_F[1,0], incr_F[1,1], wp.float64(0.),
                          wp.float64(0.), wp.float64(0.), wp.float64(1.))

    deformation_gradient_total_new[p] = incr_F_3d @ deformation_gradient_total_old[p]
    particle_J = wp.determinant(deformation_gradient_total_new[p])

    # Linear elasticity
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
    e_real = e_trial

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

    left_Cauchy_Green_new[p] = new_elastic_b_trial



    # Get new volume and density
    new_p_vol = p_vol * particle_J
    new_p_rho = p_rho / particle_J


    # External force
    f_ext = wp.vec2d(wp.float64(0.), -wp.float64(100.)/wp.float64(2.)*(step+wp.float64(1.))/wp.float64(50.))
    if particle_external_flag_array[p]==False:
        f_ext = wp.vec2d()


    for i in range(0, 3):
        for j in range(0, 3):
            weight = w[i][0] * w[j][1]
            weight_GRAD = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
            weight_grad = incr_F_inv @ weight_GRAD # push forward to the current configuration

            ix = left_bottom_corner_base_int[0] + i
            iy = left_bottom_corner_base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_y = index_ij_x + n_nodes

            rhs_value = (-weight_grad @ particle_Cauchy_stress + weight * new_p_rho * standard_gravity) * new_p_vol + weight * f_ext # Updated Lagrangian



            if (dofStruct.boundary_flag_array[index_ij_x]==False and dofStruct.activated_flag_array[index_ij_x]==True):
                wp.atomic_add(rhs, index_ij_x, rhs_value[0])

            if (dofStruct.boundary_flag_array[index_ij_y]==False and dofStruct.activated_flag_array[index_ij_y]==True):
                wp.atomic_add(rhs, index_ij_y, rhs_value[1])





# from jacobian to vector
@wp.kernel
def from_jacobian_to_vector(jacobian_wp: wp.array(dtype=wp.float64),
                            rows: wp.array(dtype=wp.int32),
                            cols: wp.array(dtype=wp.int32),
                            vals: wp.array(dtype=wp.float64),
                            n_grid_x: wp.int32,
                            n_grid_y: wp.int32,
                            n_nodes: wp.int32,
                            n_matrix_size: wp.int32,
                            row_index: wp.int32,
                            dofStruct: DofStruct):
    
    column_index = wp.tid()

    # from dof to node_id
    node_idx = wp.int(0)
    node_idy = wp.int(0)

    if row_index<n_nodes:
        node_idx = wp.mod(row_index, n_grid_x+1)
        node_idy = wp.int((row_index-node_idx)/(n_grid_x+1))
    else:
        node_idx = wp.mod((row_index-n_nodes), n_grid_x+1)
        node_idy = wp.int((row_index-n_nodes)/(n_grid_x+1))


    for i in range(5):
        adj_node_idx = node_idx + (i-2)
        for j in range(5):
            adj_node_idy = node_idy + (j-2)

            adj_index_x = adj_node_idx + adj_node_idy*(n_grid_x+1)
            adj_index_y = adj_index_x + n_nodes

            if adj_node_idx>=0 and adj_node_idx<=n_grid_x and adj_node_idy>=0 and adj_node_idy<=n_grid_y: # adj_node is reasonable
                if dofStruct.boundary_flag_array[row_index]==False and dofStruct.activated_flag_array[row_index]==True:
                    if dofStruct.boundary_flag_array[adj_index_x]==False and dofStruct.activated_flag_array[adj_index_x]==True:
                        rows[row_index*25 + (i+j*5)] = row_index
                        cols[row_index*25 + (i+j*5)] = adj_index_x
                        vals[row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_x]

                    if dofStruct.boundary_flag_array[adj_index_y]==False and dofStruct.activated_flag_array[adj_index_y]==True:
                        rows[25*n_matrix_size + row_index*25 + (i+j*5)] = row_index
                        cols[25*n_matrix_size + row_index*25 + (i+j*5)] = adj_index_y
                        vals[25*n_matrix_size + row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_y]
                

@wp.kernel
def from_jacobian_to_vector_parallel(jacobian_wp: wp.array(dtype=wp.float64),
                                     rows: wp.array(dtype=wp.int32),
                                     cols: wp.array(dtype=wp.int32),
                                     vals: wp.array(dtype=wp.float64),
                                     n_grid_x: wp.int32,
                                     n_grid_y: wp.int32,
                                     n_nodes: wp.int32,
                                     n_matrix_size: wp.int32,
                                     selector_wp: wp.array(dtype=wp.int32),
                                     dofStruct: DofStruct
                                     ):
    
    selector_index = wp.tid()

    row_index = selector_wp[selector_index]

    if row_index>0 or (row_index==0 and selector_index==0):
        # from dof to node_id
        node_idx = wp.int(0)
        node_idy = wp.int(0)

        if row_index<n_nodes: # x-dof
            node_idx = wp.mod(row_index, n_grid_x+1)
            node_idy = wp.int((row_index-node_idx)/(n_grid_x+1)) 
        else:
            node_idx = wp.mod((row_index-n_nodes), n_grid_x+1)
            node_idy = wp.int((row_index-n_nodes)/(n_grid_x+1))


        for i in range(5):
            adj_node_idx = node_idx + (i-2)
            for j in range(5):
                adj_node_idy = node_idy + (j-2)
                
                adj_index_x = adj_node_idx + adj_node_idy*(n_grid_x+1)
                adj_index_y = adj_index_x + n_nodes

                if adj_node_idx>=0 and adj_node_idx<=n_grid_x and adj_node_idy>=0 and adj_node_idy<=n_grid_y: # adj_node is reasonable
                    if dofStruct.boundary_flag_array[row_index]==False and dofStruct.activated_flag_array[row_index]==True:
                        if dofStruct.boundary_flag_array[adj_index_x]==False and dofStruct.activated_flag_array[adj_index_x]==True:
                            rows[row_index*25 + (i+j*5)] = row_index
                            cols[row_index*25 + (i+j*5)] = adj_index_x
                            vals[row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_x]

                        if dofStruct.boundary_flag_array[adj_index_y]==False and dofStruct.activated_flag_array[adj_index_y]==True:
                            rows[25*n_matrix_size + row_index*25 + (i+j*5)] = row_index
                            cols[25*n_matrix_size + row_index*25 + (i+j*5)] = adj_index_y
                            vals[25*n_matrix_size + row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_y]



                

@wp.kernel
def set_diagnal_component_for_boundary_and_deactivated_dofs(dofStruct: DofStruct,
                                                            rows: wp.array(dtype=wp.int32),
                                                            cols: wp.array(dtype=wp.int32),
                                                            vals: wp.array(dtype=wp.float64),
                                                            n_matrix_size: wp.int32):
    dof_id = wp.tid()

    if dofStruct.boundary_flag_array[dof_id]==True or dofStruct.activated_flag_array[dof_id]==False:
        rows[50*n_matrix_size + dof_id] = dof_id
        cols[50*n_matrix_size + dof_id] = dof_id
        vals[50*n_matrix_size + dof_id] = wp.float64(1.0)



# From increment to solution
@wp.kernel
def from_increment_to_solution(increment_iteration: wp.array(dtype=wp.float64),
                               increment_solution: wp.array(dtype=wp.float64)):
    i = wp.tid()

    increment_solution[i] += increment_iteration[i]



# G2P
@wp.kernel
def G2P(x_particles: wp.array(dtype=wp.vec2d),
        delta_u_particles: wp.array(dtype=wp.vec2d),
        deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
        deformation_gradient_total_old: wp.array(dtype=wp.mat33d),
        left_Cauchy_Green_new: wp.array(dtype=wp.mat33d),
        left_Cauchy_Green_old: wp.array(dtype=wp.mat33d),
        inv_dx: wp.float64,
        dx: wp.float64,
        lame_lambda: wp.float64,
        lame_mu: wp.float64,
        n_grid_x: wp.int32,
        n_nodes: wp.int32,
        increment_solution: wp.array(dtype=wp.float64),
        dofStruct: DofStruct,
        GIMP_lp: wp.array(dtype=wp.vec2d)
        ):

    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    lpx = GIMP_lp[p][0]
    lpy = GIMP_lp[p][1]

    # Calculate shape functions
    left_bottom_corner = x_particles[p] - wp.vec2d(lpx, lpy)
    left_bottom_corner_base_x = left_bottom_corner[0]*inv_dx + wp.float64(1e-8)
    left_bottom_corner_base_y = left_bottom_corner[1]*inv_dx + wp.float64(1e-8)
    left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
    left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))


    # GIMP
    GIMP_shape_function_and_gradient_components = get_GIMP_shape_function_and_gradient(x_particles[p], dx, inv_dx, dofStruct, n_grid_x, n_nodes, lpx, lpy)
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
    delta_u = wp.vec2d()

    for i in range(0, 3):
        for j in range(0, 3):
            weight = w[i][0] * w[j][1]
            weight_grad = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
            ix = left_bottom_corner_base_int[0] + i
            iy = left_bottom_corner_base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_y = index_ij_x + n_nodes

            node_increment_solution = wp.vec2d(increment_solution[index_ij_x], increment_solution[index_ij_y])

            delta_u += weight * (node_increment_solution)

    

    # Set old to new
    deformation_gradient_total_old[p] = deformation_gradient_total_new[p]
    left_Cauchy_Green_old[p] = left_Cauchy_Green_new[p]

    delta_u_particles[p] = delta_u
    x_particles[p] += delta_u

    





@wp.kernel
def update_GIMP_lp(deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
                   GIMP_lp: wp.array(dtype=wp.vec2d),
                   GIMP_lp_initial: wp.float64,
                   x_particles: wp.array(dtype=wp.vec2d),
                   start_x: wp.float64
                   ):
    p = wp.tid()

    # Refer to Eq. (38), (39) of Charlton et al. 2017
    this_particle_F = deformation_gradient_total_new[p]
    U00 = wp.sqrt(this_particle_F[0,0]*this_particle_F[0,0] + this_particle_F[1,0]*this_particle_F[1,0] + this_particle_F[2,0]*this_particle_F[2,0])
    U11 = wp.sqrt(this_particle_F[0,1]*this_particle_F[0,1] + this_particle_F[1,1]*this_particle_F[1,1] + this_particle_F[2,1]*this_particle_F[2,1])
    U22 = wp.sqrt(this_particle_F[0,2]*this_particle_F[0,2] + this_particle_F[1,2]*this_particle_F[1,2] + this_particle_F[2,2]*this_particle_F[2,2])

    lpx_updated = GIMP_lp_initial*U00
    lpy_updated = GIMP_lp_initial*U11

    if x_particles[p][0]-lpx_updated < start_x:
        lpx_updated = x_particles[p][0] - start_x - wp.float64(1e-6)

    

    GIMP_lp[p] = wp.vec2d(lpx_updated, lpy_updated)




@wp.kernel
def get_midpoint_displacement(x_particles: wp.array(dtype=wp.vec2d),
                              x0_particles: wp.array(dtype=wp.vec2d),
                              particle_external_flag_array: wp.array(dtype=wp.bool),
                              horizontal_and_vertical_displacement: wp.array(dtype=wp.vec2d)):
    p = wp.tid()

    if particle_external_flag_array[p]==True:
        particle_disp = x_particles[p]-x0_particles[p]
        wp.atomic_add(horizontal_and_vertical_displacement, 0, (particle_disp)/wp.float64(2.))


# Post-processing parameters
n_iter = 6
total_step = 0
output_frame = 0






def init_particle_position(start_x, start_y, end_x, end_y, dx, n_grid_x, n_grid_y, PPD, n_particles):
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


start_x = dx
start_y = l0-d0
end_x = start_x + l0
end_y = l0

particle_pos_np = init_particle_position(start_x, start_y, end_x, end_y, dx, n_grid_x, n_grid_y, PPD, n_particles)

x_particles = wp.from_numpy(particle_pos_np, dtype=wp.vec2d)
x0_particles = wp.from_numpy(particle_pos_np, dtype=wp.vec2d)


v_particles = wp.array(np.random.rand(n_particles, 2), dtype=wp.vec2d)
deformation_gradient_total_new = wp.array(shape=n_particles, dtype=wp.mat33d)
deformation_gradient_total_old = wp.array(shape=n_particles, dtype=wp.mat33d)
left_Cauchy_Green_new = wp.array(shape=n_particles, dtype=wp.mat33d)
left_Cauchy_Green_old = wp.array(shape=n_particles, dtype=wp.mat33d)
delta_u_particles = wp.zeros(shape=n_particles, dtype=wp.vec2d)
particle_Cauchy_stress_array = wp.zeros(shape=n_particles, dtype=wp.mat33d)
particle_external_flag_array = wp.zeros(shape=n_particles, dtype=wp.bool)

GIMP_lp = wp.array(shape=n_particles, dtype=wp.vec2d)

particle_solution = wp.zeros(shape=n_particles, dtype=wp.float64)
analytical_solution = wp.zeros(shape=n_particles, dtype=wp.float64)
error_abs = wp.zeros(shape=n_particles, dtype=wp.float64)

error_numerator = wp.zeros(shape=1, dtype=wp.float64)
error_denominator = wp.zeros(shape=1, dtype=wp.float64)

horizontal_and_vertical_displacement = wp.zeros(shape=1, dtype=wp.vec2d)


grid_v = wp.zeros(shape=(n_grid_x+1, n_grid_y+1), dtype=wp.vec2d)
grid_old_v = wp.zeros(shape=(n_grid_x+1, n_grid_y+1), dtype=wp.vec2d)
grid_f = wp.zeros(shape=(n_grid_x+1, n_grid_y+1), dtype=wp.vec2d)
grid_m = wp.zeros(shape=(n_grid_x+1, n_grid_y+1), dtype=wp.float64)





# Matrix
n_nodes = (n_grid_x+1) * (n_grid_y+1)
n_matrix_size = 2 * n_nodes # 2 indicates the spatial dimension
bsr_matrix = wps.bsr_zeros(n_matrix_size, n_matrix_size, block_type=wp.float64)
rhs = wp.zeros(shape=n_matrix_size, dtype=wp.float64, requires_grad=True)
increment_iteration = wp.zeros(shape=n_matrix_size, dtype=wp.float64)
increment_solution = wp.zeros(shape=n_matrix_size, dtype=wp.float64, requires_grad=True)


rows = wp.zeros(shape=2*25*n_matrix_size+n_matrix_size, dtype=wp.int32) # 2 indicates the spatial dimension
cols = wp.zeros(shape=2*25*n_matrix_size+n_matrix_size, dtype=wp.int32)
vals = wp.zeros(shape=2*25*n_matrix_size+n_matrix_size, dtype=wp.float64)




dofStruct = DofStruct()
dofStruct.activated_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)
dofStruct.boundary_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)


# Initialization
wp.launch(kernel=initialization,
          dim=n_particles,
          inputs=[deformation_gradient_total_new, deformation_gradient_total_old, left_Cauchy_Green_new, left_Cauchy_Green_old, x_particles, v_particles])

wp.launch(kernel=initialize_GIMP_lp,
          dim=n_particles,
          inputs=[GIMP_lp, GIMP_lp_initial])

wp.launch(kernel=set_external_force_flag,
          dim=n_particles,
          inputs=[x_particles, particle_external_flag_array, start_x, start_y, dx, PPD, l0, d0])



# Post-processing
x_numpy = np.array(x_particles.numpy())
output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': particle_Cauchy_stress_array.numpy()[:,1,1], 'ext_boundary_flag': particle_external_flag_array.numpy().astype(float)})
output_particles.write("2d_beam_particles_%d.vtk" % 0)


load_displacement_array = np.array([0.0, 0.0, 0.0])





# Sparse differentiation: get the max length of the selector array
max_selector_length = 0
c_iter = 0
r_iter = 0
current_node_id = c_iter + r_iter*(n_grid_x+1)

def pick_grid_nodes(n_grid_x, n_grid_y, first_grid_node):
    # Determine the row and column of the first grid node
    row_start = first_grid_node // (n_grid_x + 1)
    col_start = first_grid_node % (n_grid_x + 1)

    # Step size for the 5x5 grid
    step = 5

    # Create arrays for rows and columns to pick nodes efficiently
    rows = np.arange(row_start, n_grid_y + 1, step)
    cols = np.arange(col_start, n_grid_x + 1, step)

    # Generate a grid of row and column indices
    row_indices, col_indices = np.meshgrid(rows, cols, indexing='ij')

    # Compute the node indices
    picked_nodes = row_indices * (n_grid_x + 1) + col_indices

    # Flatten the array and filter out indices exceeding the total number of nodes
    picked_nodes = picked_nodes.flatten()
    
    return picked_nodes

# Get the max_selector_length
selector = pick_grid_nodes(n_grid_x, n_grid_y, current_node_id)
max_selector_length = len(selector)

# Pre-compute and save the activated dofs for differentiation
selector_x_list = []
selector_y_list = []
e_x_list = []
e_y_list = []
for c_iter in range(5):
    for r_iter in range(5):
        current_node_id = c_iter + r_iter * (n_grid_x+1) 
        select_index = np.zeros(n_matrix_size)

        # x
        selector_x = pick_grid_nodes(n_grid_x, n_grid_y, current_node_id)
        selector_x_resize = selector_x + 0
        selector_x_resize.resize(max_selector_length)
        selector_x_wp = wp.from_numpy(selector_x_resize, dtype=wp.int32)
        selector_x_list.append(selector_x_wp)

        select_index[selector_x] = 1.
        e = wp.array(select_index, dtype=wp.float64)
        e_x_list.append(e)

        # y
        selector_y = selector_x + n_nodes
        selector_y_resize = selector_y + 0
        selector_y_resize.resize(max_selector_length)
        selector_y_wp = wp.from_numpy(selector_y_resize, dtype=wp.int32)
        selector_y_list.append(selector_y_wp)

        select_index = np.zeros(n_matrix_size)
        select_index[selector_y] = 1.
        e = wp.array(select_index, dtype=wp.float64)
        e_y_list.append(e)



# Timer
class reentrant_timer:
    def __init__(self):
        self.total_time = 0.0
        self._start_time = None

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start_time
        self.total_time += elapsed

    def reset(self):
        self.total_time = 0.0

    def get_total_time(self):
        return self.total_time

timer = reentrant_timer()


# tic = time.perf_counter()
for step in range(50):
    print('Load step', step)

    grid_v.zero_()
    grid_old_v.zero_()
    grid_f.zero_()
    grid_m.zero_()


    increment_solution.zero_()
    increment_iteration.zero_()

    error_numerator.zero_()
    error_denominator.zero_()

    dofStruct.boundary_flag_array.zero_()
    dofStruct.activated_flag_array.zero_()

    # P2G
    wp.launch(kernel=set_boundary_dofs,
              dim=(n_grid_x+1, n_grid_y+1),
              inputs=[dofStruct, n_grid_x, n_grid_y, n_nodes, end_y, d0, dx])

    wp.launch(kernel=P2G,
              dim=n_particles,
              inputs=[x_particles, inv_dx, dx, n_grid_x, n_nodes, dofStruct, GIMP_lp])

    boundary_flag_array_np = dofStruct.boundary_flag_array.numpy()
    activated_flag_array_np = dofStruct.activated_flag_array.numpy()


    # Newton iteration
    for iter_id in range(n_iter):

        rhs.zero_()
        rows.zero_()
        cols.zero_()
        vals.zero_()

        tape = wp.Tape()
        with tape:
            # assemble residual
            wp.launch(kernel=assemble_residual,
                      dim=n_particles,
                      inputs=[x_particles, inv_dx, dx, rhs, increment_solution, p_vol, p_rho, lame_lambda, lame_mu, deformation_gradient_total_new, deformation_gradient_total_old, left_Cauchy_Green_new, left_Cauchy_Green_old, n_grid_x, n_nodes, dofStruct, step, particle_Cauchy_stress_array, particle_external_flag_array, GIMP_lp])


        with timer:
            # Sparse differentiation
            pattern_id = 0
            for c_iter in range(5):
                for r_iter in range(5):
                    current_node_id = c_iter + r_iter * (n_grid_x+1)
                    select_index = np.zeros(n_matrix_size)

                    # x
                    tape.backward(grads={rhs: e_x_list[pattern_id]})
                    jacobian_wp = tape.gradients[increment_solution]
                    wp.launch(kernel=from_jacobian_to_vector_parallel,
                              dim=max_selector_length,
                              inputs=[jacobian_wp, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, selector_x_list[pattern_id], dofStruct])
                    tape.zero()

                    # y
                    tape.backward(grads={rhs: e_y_list[pattern_id]})
                    jacobian_wp = tape.gradients[increment_solution]
                    wp.launch(kernel=from_jacobian_to_vector_parallel,
                              dim=max_selector_length,
                              inputs=[jacobian_wp, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, selector_y_list[pattern_id], dofStruct])
                    tape.zero()


                    pattern_id = pattern_id + 1

            tape.reset()





            # # Naive differentiation
            # for output_index in range(n_matrix_size): # Loop on dofs

            #     if boundary_flag_array_np[output_index]==True or activated_flag_array_np[output_index]==False: # This is important for efficient assemblage
            #         continue

            #     select_index = np.zeros(n_matrix_size)
            #     select_index[output_index] = 1.0
            #     e = wp.array(select_index, dtype=wp.float64)

            #     tape.backward(grads={rhs: e})
            #     q_grad_i = tape.gradients[increment_solution]


            #     wp.launch(kernel=from_jacobian_to_vector,
            #           dim=n_matrix_size,
            #           inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, output_index, dofStruct])

                            
            #     tape.zero()

            # tape.reset()


        # Adjust diagonal components of the global matrix
        wp.launch(kernel=set_diagnal_component_for_boundary_and_deactivated_dofs,
                  dim=n_matrix_size,
                  inputs=[dofStruct, rows, cols, vals, n_matrix_size])

        


        # # Assemble from vectors to sparse matrix
        # wps.bsr_set_from_triplets(bsr_matrix, rows, cols, vals)

        # # Solve
        # preconditioner = wp.optim.linear.preconditioner(bsr_matrix, ptype='diag')
        # solver_state = wp.optim.linear.bicgstab(A=bsr_matrix, b=rhs, x=increment, tol=1e-10, M=preconditioner)





        # Exploring other solvers
        bsr_matrix_other = sparse.coo_matrix((vals.numpy(), (rows.numpy(), cols.numpy())), shape=(n_matrix_size, n_matrix_size)).asformat('csr')


        mls = smoothed_aggregation_solver(bsr_matrix_other)

        b = rhs.numpy()
        residuals = []
        x_pyamg = mls.solve(b, tol=1e-8, accel='bicgstab', residuals=residuals)

        increment_iteration = wp.from_numpy(x_pyamg, dtype=wp.float64)




        # # Scipy direct solver
        # bsr_matrix_other = sparse.coo_matrix((vals.numpy(), (rows.numpy(), cols.numpy())), shape=(n_matrix_size, n_matrix_size)).asformat('csr')
        # b = rhs.numpy()
        # bsr_matrix_other_array = bsr_matrix_other.toarray()
        # # for matrix_iter in range(n_matrix_size):
        # #     print(bsr_matrix_other_array[matrix_iter, matrix_iter])
        # x_direct = spsolve(bsr_matrix_other, b)

        # increment = wp.from_numpy(x_direct, dtype=wp.float64)




        # From increment to solution
        wp.launch(kernel=from_increment_to_solution,
                  dim=n_matrix_size,
                  inputs=[increment_iteration, increment_solution])



        with np.printoptions(threshold=np.inf):
            # print(bsr_matrix.values.numpy())
            # print(rows.numpy())
            # print(rhs.numpy())
            # print('solver state:', solver_state)
            print('residual.norm:', np.linalg.norm(rhs.numpy()))
            # print(solution.numpy())
            # pass


        if np.linalg.norm(rhs.numpy())<1e-8:
            break


    # G2P
    wp.launch(kernel=G2P,
              dim=n_particles,
              inputs=[x_particles, delta_u_particles, deformation_gradient_total_new, deformation_gradient_total_old, left_Cauchy_Green_new, left_Cauchy_Green_old, inv_dx, dx, lame_lambda, lame_mu, n_grid_x, n_nodes, increment_solution, dofStruct, GIMP_lp])


    wp.launch(kernel=update_GIMP_lp,
              dim=n_particles,
              inputs=[deformation_gradient_total_new, GIMP_lp, GIMP_lp_initial, x_particles, start_x])


    # Calculate horizontal and vertical displacements
    horizontal_and_vertical_displacement.zero_()

    wp.launch(kernel=get_midpoint_displacement,
              dim=n_particles,
              inputs=[x_particles, x0_particles, particle_external_flag_array, horizontal_and_vertical_displacement])

    # print(horizontal_and_vertical_displacement.numpy())

    new_data = np.array([(step+1.)/50., horizontal_and_vertical_displacement.numpy()[0,0], horizontal_and_vertical_displacement.numpy()[0,1]])
    load_displacement_array = np.r_[load_displacement_array, new_data]



    # Post-processing
    x_numpy = np.array(x_particles.numpy())
    output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_xx': particle_Cauchy_stress_array.numpy()[:,0,0],'stress_yy': particle_Cauchy_stress_array.numpy()[:,1,1], 'ext_boundary_flag': particle_external_flag_array.numpy().astype(float)})
    output_particles.write("2d_bream_particles_%d.vtk" % (step+1))

    output_frame += 1


load_displacement_array = np.reshape(load_displacement_array, (-1,3))
# print(load_displacement_array)

# toc = time.perf_counter()
# print('Total run time:', toc-tic)

print("Differentiation time:", timer.get_total_time())

