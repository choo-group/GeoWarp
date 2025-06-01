import warp as wp
import numpy as np
import meshio

import warp.sparse as wps

import warp.optim.linear

import scipy.linalg as sla
from scipy import sparse
from scipy.sparse.linalg import spsolve


import pyamg
from pyamg import smoothed_aggregation_solver
from pyamg.krylov import bicgstab


from scipy.optimize import fsolve


# Implicit MPM solver for column compaction under self weight using Warp
# Contact: Yidong Zhao (ydzhao94@gmail.com)


wp.init()


n_particles = 256 #16 #32 #64 #128 # #512 #1024 #2048 
n_grid_x = 3
n_grid_y = 66 #6 #10 #18 #34 # #130 #258 #514 
grid_size = (n_grid_x, n_grid_y)

max_x = 2.34375  #37.5 #18.75 #9.375 #4.6875 # #1.171875 #0.5859375 #0.29296875 
dx = max_x/n_grid_x 
inv_dx = float(n_grid_x/max_x)

PPD = 2 # particle per direction

youngs_modulus = 10 # kPa
poisson_ratio = 0.0
lame_mu = youngs_modulus / (2.0*(1.0+poisson_ratio))
lame_lambda = youngs_modulus*poisson_ratio / ((1.0+poisson_ratio) * (1.0-2.0*poisson_ratio))
kappa = 5. # kPa

p_vol = (dx/PPD)**2 
p_rho = 0.08 # t/m^3
p_mass = p_vol * p_rho

GIMP_lp_initial = dx/PPD/2.0 - 1e-6







# To check whether a specific dof index is activated or at the Dirichlet boundary
@wp.struct
class DofStruct:
    activate_flag_array: wp.array(dtype=wp.bool)
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
                      float64_one+ wp.float64(1e-7), float64_zero, float64_zero,
                      float64_zero, float64_one-wp.float64(1e-6), float64_zero,
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







# Set boundary dofs
@wp.kernel
def set_boundary_dofs(dofStruct: DofStruct,
                      n_grid_x: wp.int32,
                      n_nodes: wp.int32):
    
    node_idx, node_idy = wp.tid()
    dof_x = node_idx + node_idy*(n_grid_x + 1)
    dof_y = dof_x + n_nodes

    if node_idx<=1 or node_idx>=2:
        dofStruct.boundary_flag_array[dof_x] = True

    if node_idy<=1:
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

    useless_variable = float64_zero


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

            dofStruct.activate_flag_array[index_ij_x] = True
            dofStruct.activate_flag_array[index_ij_y] = True



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

    # lp = wp.float64(0.041666) #wp.float64(0.062) #wp.float64(0.25)

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






# Assemble residual with tape
@wp.kernel
def assemble_residual_GIMP(x_particles: wp.array(dtype=wp.vec2d),
                           inv_dx: wp.float64,
                           dx: wp.float64,
                           rhs: wp.array(dtype=wp.float64),
                           solution: wp.array(dtype=wp.float64),
                           old_solution: wp.array(dtype=wp.float64),
                           p_vol: wp.float64,
                           p_rho: wp.float64,
                           lame_lambda: wp.float64,
                           lame_mu: wp.float64,
                           kappa: wp.float64,
                           deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
                           deformation_gradient_total_old: wp.array(dtype=wp.mat33d),
                           left_Cauchy_Green_new: wp.array(dtype=wp.mat33d),
                           left_Cauchy_Green_old: wp.array(dtype=wp.mat33d),
                           n_grid_x: wp.int32,
                           n_nodes: wp.int32,
                           dofStruct: DofStruct,
                           step: wp.float64,
                           particle_Cauchy_stress_array: wp.array(dtype=wp.mat33d),
                           GIMP_lp: wp.array(dtype=wp.vec2d)
                           ):
    p = wp.tid()

    lpx = GIMP_lp[p][0]
    lpy = GIMP_lp[p][1]

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    standard_gravity = wp.vec2d(float64_zero, wp.float64(-10.0)*(step+float64_one)/wp.float64(40.0)) 

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

            node_solution = wp.vec2d(solution[index_ij_x], solution[index_ij_y])
            node_old_solution = wp.vec2d(old_solution[index_ij_x], old_solution[index_ij_y])

            delta_u_GRAD += wp.outer(weight_grad, node_solution-node_old_solution)


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
    e_real = e_trial

    # Get trial Kirchhoff stress
    eps_v = wp.trace(e_trial)
    tau_trial = lame_lambda*eps_v*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*e_trial

    # Get P and S
    P = wp.float64(1.)/wp.float64(3.) * wp.trace(tau_trial)
    S_trial = tau_trial - P*wp.identity(n=3, dtype=wp.float64)
    S_trial_norm = wp.sqrt(wp.pow(S_trial[0,0], wp.float64(2.)) + wp.pow(S_trial[1,1], wp.float64(2.)) + wp.pow(S_trial[2,2], wp.float64(2.)))

    if S_trial_norm>kappa: # Yield, radial return mapping for J2 plasticity
        
        n = S_trial/S_trial_norm

        delta_lambda = (S_trial_norm - kappa)/(wp.float64(2.)*lame_mu)

        e_real = e_trial - delta_lambda*n


    # Get real Kirchhoff stress
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



            if (dofStruct.boundary_flag_array[index_ij_x]==False and dofStruct.activate_flag_array[index_ij_x]==True):
                wp.atomic_add(rhs, index_ij_x, rhs_value[0])

            if (dofStruct.boundary_flag_array[index_ij_y]==False and dofStruct.activate_flag_array[index_ij_y]==True):
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
                if dofStruct.boundary_flag_array[row_index]==False and dofStruct.activate_flag_array[row_index]==True:
                    if dofStruct.boundary_flag_array[adj_index_x]==False and dofStruct.activate_flag_array[adj_index_x]==True:
                        rows[row_index*25 + (i+j*5)] = row_index
                        cols[row_index*25 + (i+j*5)] = adj_index_x
                        vals[row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_x]

                    if dofStruct.boundary_flag_array[adj_index_y]==False and dofStruct.activate_flag_array[adj_index_y]==True:
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

    if dofStruct.boundary_flag_array[dof_id]==True or dofStruct.activate_flag_array[dof_id]==False:
        rows[50*n_matrix_size + dof_id] = dof_id
        cols[50*n_matrix_size + dof_id] = dof_id
        vals[50*n_matrix_size + dof_id] = wp.float64(1.0)




# From increment to solution
@wp.kernel
def from_increment_to_solution(increment: wp.array(dtype=wp.float64),
                               solution: wp.array(dtype=wp.float64)):
    i = wp.tid()

    solution[i] += increment[i]








# G2P (GIMP)
@wp.kernel
def G2P_GIMP(x_particles: wp.array(dtype=wp.vec2d),
             delta_u_particles: wp.array(dtype=wp.vec2d),
             deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
             deformation_gradient_total_old: wp.array(dtype=wp.mat33d),
             left_Cauchy_Green_new: wp.array(dtype=wp.mat33d),
             left_Cauchy_Green_old: wp.array(dtype=wp.mat33d),
             particle_Cauchy_stress_array: wp.array(dtype=wp.mat33d),
             inv_dx: wp.float64,
             dx: wp.float64,
             lame_lambda: wp.float64,
             lame_mu: wp.float64,
             kappa: wp.float64,
             n_grid_x: wp.int32,
             n_nodes: wp.int32,
             new_solution: wp.array(dtype=wp.float64),
             old_solution: wp.array(dtype=wp.float64),
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

            node_new_solution = wp.vec2d(new_solution[index_ij_x], new_solution[index_ij_y])
            node_old_solution = wp.vec2d(old_solution[index_ij_x], old_solution[index_ij_y])

            delta_u += weight * (node_new_solution - node_old_solution)

    
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
                   start_x: wp.float64,
                   start_y: wp.float64
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

    if x_particles[p][1]-lpy_updated < start_y:
        lpy_updated = x_particles[p][1] - start_y - wp.float64(1e-6)


    GIMP_lp[p] = wp.vec2d(lpx_updated, lpy_updated)




# Calculate error
# [error_numerator, error_denominator, p_rho, p_vol]
@wp.kernel
def calculate_error(error_numerator: wp.array(dtype=wp.float64),
                    error_denominator: wp.array(dtype=wp.float64),
                    particle_Cauchy_stress_array: wp.array(dtype=wp.mat33d),
                    x0: wp.array(dtype=wp.vec2d),
                    p_rho: wp.float64,
                    p_vol: wp.float64,
                    start_y: wp.float64,
                    n_particles: wp.float64):
    p = wp.tid()

    analytical_stress_yy = p_rho * wp.float64(-10.0) * (wp.float64(50.0) - (x0[p][1]-start_y))
    particle_stress_yy = particle_Cauchy_stress_array[p][1,1]

    

    wp.atomic_add(error_numerator, 0, wp.abs(particle_stress_yy-analytical_stress_yy) * p_vol)
    wp.atomic_add(error_denominator, 0, wp.float64(10.0) * p_rho * wp.float64(50.0) * p_vol)


n_iter = 10 
total_step = 0
output_frame = 0

# Initialization
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
start_y = dx
end_x = start_x + dx
end_y = start_y + 50.0 # m

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

GIMP_lp = wp.array(shape=n_particles, dtype=wp.vec2d)



error_numerator = wp.zeros(shape=1, dtype=wp.float64)
error_denominator = wp.zeros(shape=1, dtype=wp.float64)







# Matrix
n_nodes = (n_grid_x+1) * (n_grid_y+1)
n_matrix_size = 2 * n_nodes # 2 indicates the spatial dimension
bsr_matrix = wps.bsr_zeros(n_matrix_size, n_matrix_size, block_type=wp.float64)
rhs = wp.zeros(shape=n_matrix_size, dtype=wp.float64, requires_grad=True)
increment = wp.zeros(shape=n_matrix_size, dtype=wp.float64)
old_solution = wp.zeros(shape=n_matrix_size, dtype=wp.float64)
new_solution = wp.zeros(shape=n_matrix_size, dtype=wp.float64, requires_grad=True)


rows = wp.zeros(shape=2*25*n_matrix_size+n_matrix_size, dtype=wp.int32) # 2 indicates the spatial dimension. 25 because each node will affect its surrounding 5x5 nodes
cols = wp.zeros(shape=2*25*n_matrix_size+n_matrix_size, dtype=wp.int32)
vals = wp.zeros(shape=2*25*n_matrix_size+n_matrix_size, dtype=wp.float64)




dofStruct = DofStruct()
dofStruct.activate_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)
dofStruct.boundary_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)


# Initialization
wp.launch(kernel=initialization,
          dim=n_particles,
          inputs=[deformation_gradient_total_new, deformation_gradient_total_old, left_Cauchy_Green_new, left_Cauchy_Green_old, x_particles, v_particles])
wp.launch(kernel=initialize_GIMP_lp,
          dim=n_particles,
          inputs=[GIMP_lp, GIMP_lp_initial])


# Post-processing
x_numpy = np.array(x_particles.numpy())
output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_xx': particle_Cauchy_stress_array.numpy()[:,0,0], 'stress_yy': particle_Cauchy_stress_array.numpy()[:,1,1]})
output_particles.write("1d_compaction_particles_%d.vtk" % 0)


# Analytical solution for F_xx^e
def equation_for_Fzze(Fzze, sigma_zz):
    return sigma_zz * Fzze**3 * np.exp(2.*kappa/np.sqrt(2.0/3.0)/youngs_modulus) - youngs_modulus * np.log(Fzze)


# while True:
for step in range(40):
    print('Load step', step)

    old_solution.zero_()
    new_solution.zero_()
    increment.zero_()

    error_numerator.zero_()
    error_denominator.zero_()

    # P2G
    # print('ready P2G')
    dofStruct.activate_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)
    dofStruct.boundary_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool) 
    wp.launch(kernel=set_boundary_dofs,
              dim=(n_grid_x+1, n_grid_y+1),
              inputs=[dofStruct, n_grid_x, n_nodes])

    wp.launch(kernel=P2G,
              dim=n_particles,
              inputs=[x_particles, inv_dx, dx, n_grid_x, n_nodes, dofStruct, GIMP_lp])

    boundary_flag_array_np = dofStruct.boundary_flag_array.numpy()
    activate_flag_array_np = dofStruct.activate_flag_array.numpy()


    # Newton iteration
    # print('ready newton solver')
    for iter_id in range(n_iter):

        rhs.zero_()
        rows.zero_()
        cols.zero_()
        vals.zero_()

        tape = wp.Tape()
        with tape:
            # assemble residual
            wp.launch(kernel=assemble_residual_GIMP, 
                      dim=n_particles,
                      inputs=[x_particles, inv_dx, dx, rhs, new_solution, old_solution, p_vol, p_rho, lame_lambda, lame_mu, kappa, deformation_gradient_total_new, deformation_gradient_total_old, left_Cauchy_Green_new, left_Cauchy_Green_old, n_grid_x, n_nodes, dofStruct, step, particle_Cauchy_stress_array, GIMP_lp])



        # print('ready get jacobian using auto-diff')
        for output_index in range(n_matrix_size): # Loop on dofs

            if boundary_flag_array_np[output_index]==True or activate_flag_array_np[output_index]==False: # Ignore boundary dof and non-activated dof. This is important for efficient assemblage.
                continue

            select_index = np.zeros(n_matrix_size)
            select_index[output_index] = 1.0
            e = wp.array(select_index, dtype=wp.float64)

            # print('ready call tape.backward')
            tape.backward(grads={rhs: e})
            q_grad_i = tape.gradients[new_solution]


            wp.launch(kernel=from_jacobian_to_vector,
                  dim=n_matrix_size,
                  inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, output_index, dofStruct])



                        
            tape.zero()


        tape.reset()


        # Adjust diagonal components of the global matrix
        # print('ready adjust diagonal components')
        wp.launch(kernel=set_diagnal_component_for_boundary_and_deactivated_dofs,
                  dim=n_matrix_size,
                  inputs=[dofStruct, rows, cols, vals, n_matrix_size])


        # Scipy direct solver
        # print('ready to solve')
        bsr_matrix_other = sparse.coo_matrix((vals.numpy(), (rows.numpy(), cols.numpy())), shape=(n_matrix_size, n_matrix_size)).asformat('csr')
        b = rhs.numpy()
        bsr_matrix_other_array = bsr_matrix_other.toarray()
        
        x_direct = spsolve(bsr_matrix_other, b)




        increment = wp.from_numpy(x_direct, dtype=wp.float64)



        # From increment to solution
        # print('ready from increment to solution')
        wp.launch(kernel=from_increment_to_solution,
                  dim=n_matrix_size,
                  inputs=[increment, new_solution])


        with np.printoptions(threshold=np.inf):
            print(np.linalg.norm(rhs.numpy()))


        if np.linalg.norm(rhs.numpy())<1e-8:
            break


    # G2P
    # print('ready G2P')
    wp.launch(kernel=G2P_GIMP, #G2P,
              dim=n_particles,
              inputs=[x_particles, delta_u_particles, deformation_gradient_total_new, deformation_gradient_total_old, left_Cauchy_Green_new, left_Cauchy_Green_old, particle_Cauchy_stress_array, inv_dx, dx, lame_lambda, lame_mu, kappa, n_grid_x, n_nodes, new_solution, old_solution, dofStruct, GIMP_lp])

    wp.launch(kernel=update_GIMP_lp,
              dim=n_particles,
              inputs=[deformation_gradient_total_new, GIMP_lp, GIMP_lp_initial, x_particles, start_x, start_y])


    # Calculate error
    wp.launch(kernel=calculate_error,
              dim=n_particles,
              inputs=[error_numerator, error_denominator, particle_Cauchy_stress_array, x0_particles, p_rho, p_vol, start_y, n_particles])

    print('Relative error:', error_numerator.numpy()[0]/error_denominator.numpy()[0])

   

    # Post-processing
    x_numpy = np.array(x_particles.numpy())
    output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': particle_Cauchy_stress_array.numpy()[:,1,1]})
    output_particles.write("1d_compaction_particles_%d.vtk" % (step+1))

    output_frame += 1



    # print stress
    if step==39:
        stress_xx_array = np.column_stack([particle_Cauchy_stress_array.numpy()[:,0,0], x_numpy[:,1]-start_y]).reshape(-1, 2)
        stress_yy_array = np.column_stack([particle_Cauchy_stress_array.numpy()[:,1,1], x_numpy[:,1]-start_y]).reshape(-1, 2)

        x0_numpy = np.array(x0_particles.numpy()) 
        stress_yy_analytical = p_rho * -10.0 * (50.0 - (x0_numpy[:,1]-start_y))       
        stress_yy_analytical_print = np.column_stack([stress_yy_analytical, x_numpy[:,1]-start_y]).reshape(-1, 2)
        
        np.set_printoptions(threshold=np.inf)
        print(stress_yy_analytical_print)



        # analytical solution for sigma_xx
        Fzze_solutions = np.array([fsolve(equation_for_Fzze, x0=1.0, args=(this_stress_zz_analytical,))[0] for this_stress_zz_analytical in stress_yy_analytical])
        Fzz_solutions = Fzze_solutions**3 * np.exp(2.*kappa/np.sqrt(2.0/3.0)/youngs_modulus)
        Fxxe_solutions = Fzze_solutions * np.exp(kappa/np.sqrt(2.0/3.0)/youngs_modulus)
        stress_xx_analytical = 1./Fzz_solutions * youngs_modulus * np.log(Fxxe_solutions)
        stress_xx_analytical_print = np.column_stack([stress_xx_analytical, x_numpy[:,1]-start_y]).reshape(-1, 2)
        index = np.where(stress_xx_analytical_print[:,0]>0.0)
        stress_xx_analytical_print[index,0] = 0.
        # print(stress_xx_analytical_print)


