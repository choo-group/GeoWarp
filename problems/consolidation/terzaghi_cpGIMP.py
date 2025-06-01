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



# Implicit MPM solver for coupled poromechanics (Terzaghi)
# Contact: Yidong Zhao (ydzhao94@gmail.com)

wp.init()


n_particles = 400 
n_grid_x = 3
n_grid_y = 120

max_x = 0.3
dx = max_x/n_grid_x # 0.1 m
dy = max_x/n_grid_x
inv_dx = float(n_grid_x/max_x)
inv_dy = float(n_grid_x/max_x)

PPD = 2  # particle per direction
GIMP_lp_initial = dx/PPD/2.0 - 1e-6 


youngs_modulus = 1500.0 # kPa
poisson_ratio = 0.25
lame_mu = youngs_modulus / (2.0*(1.0+poisson_ratio))
lame_lambda = youngs_modulus*poisson_ratio / ((1.0+poisson_ratio) * (1.0-2.0*poisson_ratio))

p_vol = (dx/PPD)**2 
p_rho = 1.0 # t/m^3
p_mass = p_vol * p_rho



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
                   v_particles: wp.array(dtype=wp.vec2d),
                   top_boundary_flag: wp.array(dtype=wp.bool),
                   end_x: wp.float64, 
                   end_y: wp.float64, 
                   dx: wp.float64, 
                   PPD: wp.int32):

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

    if x_particles[p][1]>end_y-dx/wp.float64(PPD):
        top_boundary_flag[p] = True
    else:
        top_boundary_flag[p] = False





@wp.kernel
def initialize_GIMP_lp(GIMP_lp: wp.array(dtype=wp.vec2d),
                       GIMP_lp_initial: wp.float64):
    p = wp.tid()

    GIMP_lp[p] = wp.vec2d(GIMP_lp_initial, GIMP_lp_initial)


# Set boundary dofs
@wp.kernel
def set_boundary_dofs(dofStruct: DofStruct,
                      n_grid_x: wp.int32,
                      n_nodes: wp.int32,
                      dx: wp.float64,
                      dy: wp.float64,
                      end_x: wp.float64
                      ):
    
    node_idx, node_idy = wp.tid()
    dof_x = node_idx + node_idy*(n_grid_x + 1)
    dof_y = dof_x + n_nodes
    dof_p = dof_x + 2*n_nodes

    # dofStruct.boundary_flag_array[dof_x] = True # Constrain all x
    if node_idx<=1 or wp.float64(node_idx)*dx+wp.float64(0.5)*dx>=end_x:
        dofStruct.boundary_flag_array[dof_x] = True

    if node_idy<=0:
        dofStruct.boundary_flag_array[dof_y] = True


    # # Dry case
    # dofStruct.boundary_flag_array[dof_p] = True




# P2G
@wp.kernel
def P2G(x_particles: wp.array(dtype=wp.vec2d),
        inv_dx: wp.float64,
        dx: wp.float64,
        inv_dy: wp.float64,
        dy: wp.float64,
        n_grid_x: wp.int32,
        n_nodes: wp.int32,
        dofStruct: DofStruct,
        grid_P2G_m: wp.array(dtype=wp.float64, ndim=2),
        grid_P2G_p: wp.array(dtype=wp.float64, ndim=2),
        p_mass: wp.float64,
        pressure_array: wp.array(dtype=wp.float64),
        top_boundary_flag: wp.array(dtype=wp.bool),
        GIMP_lp: wp.array(dtype=wp.vec2d),
        rhs_P2G: wp.array(dtype=wp.float64),
        rows_P2G: wp.array(dtype=wp.int32),
        cols_P2G: wp.array(dtype=wp.int32),
        vals_P2G: wp.array(dtype=wp.float64)):


    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    lpx = GIMP_lp[p][0]
    lpy = GIMP_lp[p][1]

    

    # GIMP
    left_bottom_corner = x_particles[p] - wp.vec2d(lpx, lpy)
    left_bottom_corner_base_x = left_bottom_corner[0]*inv_dx + wp.float64(1e-8)
    left_bottom_corner_base_y = left_bottom_corner[1]*inv_dy + wp.float64(1e-8)
    left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
    left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))

    right_up_corner = x_particles[p] + wp.vec2d(lpx, lpy)
    right_up_corner_base_x = right_up_corner[0]*inv_dx + wp.float64(1e-8)
    right_up_corner_base_y = right_up_corner[1]*inv_dy + wp.float64(1e-8)
    right_up_corner_base_int = wp.vector(wp.int(right_up_corner_base_x), wp.int(right_up_corner_base_y))
    right_up_corner_base = wp.vector(wp.float64(right_up_corner_base_int[0]), wp.float64(right_up_corner_base_int[1]))

    # shape function
    GIMP_shape_function_and_gradient_components = get_GIMP_shape_function_and_gradient(x_particles[p], dx, inv_dx, dy, inv_dy, dofStruct, n_grid_x, n_nodes, lpx, lpy)
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
            index_ij_p = index_ij_x + 2*n_nodes

            weight = w[i][0] * w[j][1]

            if weight>wp.float64(1e-6): 
                dofStruct.activate_flag_array[index_ij_x] = True
                dofStruct.activate_flag_array[index_ij_y] = True
                dofStruct.activate_flag_array[index_ij_p] = True


    # Set top pressure boundary flag
    if top_boundary_flag[p]==True:
        for i in range(0, i_range):
            for j in range(0, j_range):
                ix = left_bottom_corner_base_int[0] + i
                iy = left_bottom_corner_base_int[1] + j

                node_pos = wp.vec2d(wp.float64(ix)*dx, wp.float64(iy)*dy)
                if node_pos[1] >= x_particles[p][1]:
                    index_ij_x = ix + iy*(n_grid_x + wp.int(1))
                    index_ij_p = index_ij_x + 2*n_nodes
                    dofStruct.boundary_flag_array[index_ij_p] = True




    # pressure P2G
    for i in range(0, 3):
        for j in range(0, 3):
            weight = w[i][0] * w[j][1]
            ix = left_bottom_corner_base_int[0] + i
            iy = left_bottom_corner_base_int[1] + j
            index_ij_scalar = ix + iy*(n_grid_x + wp.int(1))
            index_ij_p = index_ij_scalar + 2*n_nodes
            for ii in range(0, 3):
                for jj in range(0, 3):
                    
                    weight_iijj = w[ii][0] * w[jj][1]
                    iix = left_bottom_corner_base_int[0] + ii
                    iiy = left_bottom_corner_base_int[1] + jj
                    index_iijj_scalar = iix + iiy*(n_grid_x + wp.int(1))
                    index_iijj_p = index_iijj_scalar + 2*n_nodes

                    wp.atomic_add(rhs_P2G, index_ij_scalar, weight * weight_iijj * p_mass * pressure_array[p])
                    rows_P2G[p*81 + (3*j+i)*9 + (3*jj+ii)] = index_ij_scalar
                    cols_P2G[p*81 + (3*j+i)*9 + (3*jj+ii)] = index_iijj_scalar
                    vals_P2G[p*81 + (3*j+i)*9 + (3*jj+ii)] = weight * weight_iijj * p_mass




@wp.kernel
def P2G_impose_p_boundary(dofStruct: DofStruct,
                          n_grid_x: wp.int32,
                          n_nodes: wp.int32,
                          rows_P2G: wp.array(dtype=wp.int32),
                          cols_P2G: wp.array(dtype=wp.int32),
                          vals_P2G: wp.array(dtype=wp.float64),
                          n_particles: wp.int32):
    
    node_idx, node_idy = wp.tid()
    dof_scalar = node_idx + node_idy*(n_grid_x + 1)
    dof_p = dof_scalar + 2*n_nodes


    if dofStruct.activate_flag_array[dof_p]==False:
        rows_P2G[n_particles*81 + dof_scalar] = dof_scalar
        cols_P2G[n_particles*81 + dof_scalar] = dof_scalar
        vals_P2G[n_particles*81 + dof_scalar] = wp.float64(1.0)




@wp.kernel
def P2G_solver_lumped(dofStruct: DofStruct,
                      n_grid_x: wp.int32,
                      n_nodes: wp.int32,
                      grid_P2G_m: wp.array(dtype=wp.float64, ndim=2),
                      grid_P2G_p: wp.array(dtype=wp.float64, ndim=2),
                      old_solution: wp.array(dtype=wp.float64),
                      new_solution: wp.array(dtype=wp.float64)):

    node_idx, node_idy = wp.tid()
    dof_x = node_idx + node_idy*(n_grid_x + 1)
    dof_y = dof_x + n_nodes
    dof_p = dof_x + 2*n_nodes

    if (dofStruct.boundary_flag_array[dof_p]==False and dofStruct.activate_flag_array[dof_p]==True):
        grid_P2G_p[node_idx, node_idy] = grid_P2G_p[node_idx, node_idy]/grid_P2G_m[node_idx, node_idy]

        old_solution[dof_p] = grid_P2G_p[node_idx, node_idy]
        new_solution[dof_p] = grid_P2G_p[node_idx, node_idy]


@wp.kernel
def P2G_solver_consistent(dofStruct: DofStruct,
                          n_nodes: wp.int32,
                          x_P2G_warp: wp.array(dtype=wp.float64),
                          old_solution: wp.array(dtype=wp.float64),
                          new_solution: wp.array(dtype=wp.float64)):
    
    dof_scalar = wp.tid()
    dof_p = dof_scalar + 2*n_nodes

    if dofStruct.boundary_flag_array[dof_p]==False:
        old_solution[dof_p] = x_P2G_warp[dof_scalar]
        new_solution[dof_p] = x_P2G_warp[dof_scalar]




@wp.func
def get_GIMP_shape_function_and_gradient(xp: wp.vec2d,
                                         dx: wp.float64,
                                         inv_dx: wp.float64,
                                         dy: wp.float64,
                                         inv_dy: wp.float64,
                                         dofStruct: DofStruct,
                                         n_grid_x: wp.int32,
                                         n_nodes: wp.int32,
                                         lpx: wp.float64,
                                         lpy: wp.float64
                                         ):


    left_bottom_corner = xp - wp.vec2d(lpx, lpy)
    left_bottom_corner_base_x = left_bottom_corner[0]*inv_dx + wp.float64(1e-8)
    left_bottom_corner_base_y = left_bottom_corner[1]*inv_dy + wp.float64(1e-8)
    left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
    left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))

    right_up_corner = xp + wp.vec2d(lpx, lpy)
    right_up_corner_base_x = right_up_corner[0]*inv_dx + wp.float64(1e-8)
    right_up_corner_base_y = right_up_corner[1]*inv_dy + wp.float64(1e-8)
    right_up_corner_base_int = wp.vector(wp.int(right_up_corner_base_x), wp.int(right_up_corner_base_y))
    right_up_corner_base = wp.vector(wp.float64(right_up_corner_base_int[0]), wp.float64(right_up_corner_base_int[1]))


    base_x = xp[0]*inv_dx + wp.float64(1e-8)
    base_y = xp[1]*inv_dy + wp.float64(1e-8)
    base_int = wp.vector(wp.int(base_x), wp.int(base_y))
    base = wp.vector(wp.float64(base_int[0]), wp.float64(base_int[1]))

    fx = wp.vec2d(xp[0]*inv_dx - base[0], xp[1]*inv_dy - base[1]) #xp * inv_dx - base


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
        grad_wy0 = -wp.float64(1.0) * inv_dy
        grad_wy1 = inv_dy

        index_top_left_x = left_bottom_corner_base_int[0] + (left_bottom_corner_base_int[1]+wp.int(2))*(n_grid_x + wp.int(1))
        index_top_left_y = index_top_left_x + n_nodes
        index_top_middle_x = left_bottom_corner_base_int[0] + wp.int(1) + (left_bottom_corner_base_int[1]+wp.int(2))*(n_grid_x + wp.int(1))
        index_top_middle_y = index_top_middle_x + n_nodes
        index_top_right_x = left_bottom_corner_base_int[0] + wp.int(2) + (left_bottom_corner_base_int[1]+wp.int(2))*(n_grid_x + wp.int(1))
        index_top_right_y = index_top_right_x + n_nodes
    else:
        # the particle will influence 2 elements in the y direction
        # shape function & gradient
        y_0 = (left_bottom_corner_base[1]) * dy
        wy0 = wp.pow((dy+lpy-wp.abs(xp[1]-y_0)), wp.float64(2.0))/(wp.float64(4)*dy*lpy)
        grad_wy0 = -(dy+lpy-wp.abs(xp[1]-y_0))/(wp.float64(2.0)*dy*lpy) * (xp[1]-y_0)/wp.abs(xp[1]-y_0)

        y_1 = (left_bottom_corner_base[1] + wp.float64(1)) * dy
        wy1 = wp.float64(1.0) - (wp.pow((xp[1]-y_1), wp.float64(2.0)) + wp.pow(lpy, wp.float64(2.0)))/(wp.float64(2)*dy*lpy)
        grad_wy1 = -(xp[1]-y_1)/(dy*lpy)

        y_2 = (left_bottom_corner_base[1] + wp.float64(2)) * dy
        wy2 = wp.pow((dy+lpy-wp.abs(xp[1]-y_2)), wp.float64(2.0))/(wp.float64(4)*dy*lpy)
        grad_wy2 = -(dy+lpy-wp.abs(xp[1]-y_2))/(wp.float64(2.0)*dy*lpy) * (xp[1]-y_2)/wp.abs(xp[1]-y_2)


    

    return wp.vector(wx0, wy0, wx1, wy1, wx2, wy2, grad_wx0, grad_wy0, grad_wx1, grad_wy1, grad_wx2, grad_wy2)




@wp.func
def get_GIMP_shape_function_and_gradient_avg(xp: wp.vec2d,
                                             dx: wp.float64,
                                             inv_dx: wp.float64,
                                             dy: wp.float64,
                                             inv_dy: wp.float64,
                                             dofStruct: DofStruct,
                                             n_grid_x: wp.int32,
                                             n_nodes: wp.int32,
                                             lpx: wp.float64,
                                             lpy: wp.float64
                                             ):
    
    # Averaged GIMP function referring to Coombs et al., CMAME 2018. Also see Zhao & Choo, CMAME 2020 for stabilized MPM
    left_bottom_corner = xp - wp.vec2d(lpx, lpy)
    left_bottom_corner_base_x = left_bottom_corner[0]*inv_dx + wp.float64(1e-8)
    left_bottom_corner_base_y = left_bottom_corner[1]*inv_dy + wp.float64(1e-8)
    left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
    left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))

    right_up_corner = xp + wp.vec2d(lpx, lpy)
    right_up_corner_base_x = right_up_corner[0]*inv_dx + wp.float64(1e-8)
    right_up_corner_base_y = right_up_corner[1]*inv_dy + wp.float64(1e-8)
    right_up_corner_base_int = wp.vector(wp.int(right_up_corner_base_x), wp.int(right_up_corner_base_y))
    right_up_corner_base = wp.vector(wp.float64(right_up_corner_base_int[0]), wp.float64(right_up_corner_base_int[1]))


    base_x = xp[0]*inv_dx + wp.float64(1e-8)
    base_y = xp[1]*inv_dy + wp.float64(1e-8)
    base_int = wp.vector(wp.int(base_x), wp.int(base_y))
    base = wp.vector(wp.float64(base_int[0]), wp.float64(base_int[1]))

    fx = wp.vec2d(xp[0]*inv_dx - base[0], xp[1]*inv_dy - base[1])


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
        y_0 = (left_bottom_corner_base[1]) * dy
        wy0 = (dy+lpy-wp.abs(xp[1]-y_0))/(wp.float64(4.0)*lpy)
        grad_wy0 = wp.float64(-1.0)/(wp.float64(4.0)*lpy) * wp.sign(xp[1]-y_0)

        wy1 = wp.float64(0.5) 

        y_2 = (left_bottom_corner_base[1] + wp.float64(2)) * dy
        wy2 = (dy+lpy-wp.abs(xp[1]-y_2))/(wp.float64(4.0)*lpy)
        grad_wy2 = wp.float64(-1.0)/(wp.float64(4.0)*lpy) * wp.sign(xp[1]-y_2)


    return wp.vector(wx0, wy0, wx1, wy1, wx2, wy2, grad_wx0, grad_wy0, grad_wx1, grad_wy1, grad_wx2, grad_wy2)






# Assemble residual with tape
@wp.kernel
def assemble_residual_GIMP(x_particles: wp.array(dtype=wp.vec2d),
                           inv_dx: wp.float64,
                           dx: wp.float64,
                           inv_dy: wp.float64,
                           dy: wp.float64,
                           rhs: wp.array(dtype=wp.float64),
                           solution: wp.array(dtype=wp.float64),
                           old_solution: wp.array(dtype=wp.float64),
                           p_vol: wp.float64,
                           p_rho: wp.float64,
                           lame_lambda: wp.float64,
                           lame_mu: wp.float64,
                           poisson_ratio: wp.float64,
                           deformation_gradient_total_new: wp.array(dtype=wp.mat33d),
                           deformation_gradient_total_old: wp.array(dtype=wp.mat33d),
                           left_Cauchy_Green_new: wp.array(dtype=wp.mat33d),
                           left_Cauchy_Green_old: wp.array(dtype=wp.mat33d),
                           n_grid_x: wp.int32,
                           n_nodes: wp.int32,
                           dofStruct: DofStruct,
                           step: wp.float64,
                           particle_Cauchy_stress_array: wp.array(dtype=wp.mat33d),
                           GIMP_lp: wp.array(dtype=wp.vec2d),
                           dt: wp.float64,
                           PPD: wp.float64,
                           n_particles: wp.int32,
                           top_boundary_flag: wp.array(dtype=wp.bool)
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
    left_bottom_corner_base_y = left_bottom_corner[1]*inv_dy + wp.float64(1e-8)
    left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
    left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))



    # GIMP
    GIMP_shape_function_and_gradient_components = get_GIMP_shape_function_and_gradient(x_particles[p], dx, inv_dx, dy, inv_dy, dofStruct, n_grid_x, n_nodes, lpx, lpy)
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



    GIMP_shape_function_and_gradient_components_avg = get_GIMP_shape_function_and_gradient_avg(x_particles[p], dx, inv_dx, dy, inv_dy, dofStruct, n_grid_x, n_nodes, lpx, lpy)
    wx0_avg = GIMP_shape_function_and_gradient_components_avg[0]
    wy0_avg = GIMP_shape_function_and_gradient_components_avg[1]
    wx1_avg = GIMP_shape_function_and_gradient_components_avg[2]
    wy1_avg = GIMP_shape_function_and_gradient_components_avg[3]
    wx2_avg = GIMP_shape_function_and_gradient_components_avg[4]
    wy2_avg = GIMP_shape_function_and_gradient_components_avg[5]
    grad_wx0_avg = GIMP_shape_function_and_gradient_components_avg[6]
    grad_wy0_avg = GIMP_shape_function_and_gradient_components_avg[7]
    grad_wx1_avg = GIMP_shape_function_and_gradient_components_avg[8]
    grad_wy1_avg = GIMP_shape_function_and_gradient_components_avg[9]
    grad_wx2_avg = GIMP_shape_function_and_gradient_components_avg[10]
    grad_wy2_avg = GIMP_shape_function_and_gradient_components_avg[11]

    w_avg = wp.matrix(
        wx0_avg, wy0_avg,
        wx1_avg, wy1_avg,
        wx2_avg, wy2_avg,
        shape=(3,2)
    )





    # Loop on dofs
    delta_u_GRAD = wp.matrix(float64_zero, float64_zero,
                             float64_zero, float64_zero, shape=(2,2))
    div_delta_u = wp.float64(0.0)
    delta_u = wp.vec2d()
    for i in range(0, 3):
        for j in range(0, 3):
            weight = w[i][0] * w[j][1]
            weight_grad = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
            ix = left_bottom_corner_base_int[0] + i
            iy = left_bottom_corner_base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_y = index_ij_x + n_nodes
            index_ij_p = index_ij_x + 2*n_nodes

            node_solution = wp.vec2d(solution[index_ij_x], solution[index_ij_y])
            node_old_solution = wp.vec2d(old_solution[index_ij_x], old_solution[index_ij_y])
            node_p_solution = solution[index_ij_p]

            delta_u_GRAD += wp.outer(weight_grad, node_solution-node_old_solution)

            div_delta_u += wp.dot(weight_grad, node_solution-node_old_solution)

            delta_u += weight * (node_solution - node_old_solution)


    old_F_3d = deformation_gradient_total_old[p]
    particle_old_J = wp.determinant(old_F_3d)

    incr_F = wp.identity(n=2, dtype=wp.float64) + delta_u_GRAD
    incr_F_inv = wp.inverse(incr_F)

    incr_F_3d = wp.mat33d(incr_F[0,0], incr_F[0,1], wp.float64(0.),
                          incr_F[1,0], incr_F[1,1], wp.float64(0.),
                          wp.float64(0.), wp.float64(0.), wp.float64(1.))

    new_F_3d = incr_F_3d @ deformation_gradient_total_old[p]
    new_F_3d_inv = wp.inverse(new_F_3d)
    deformation_gradient_total_new[p] = new_F_3d
    particle_J = wp.determinant(new_F_3d)

    left_Cauchy_Green_new[p] = incr_F_3d * left_Cauchy_Green_old[p] * wp.transpose(incr_F_3d)

    # Neo-Hookean
    particle_Cauchy_stress = lame_mu / particle_J * (new_F_3d*wp.transpose(new_F_3d) - wp.identity(n=3, dtype=wp.float64)) + lame_lambda/particle_J * wp.log(particle_J) * wp.identity(n=3, dtype=wp.float64)
    particle_Cauchy_stress_array[p] = particle_Cauchy_stress

    particle_Cauchy_stress_2d = wp.mat22d(
                                particle_Cauchy_stress[0,0], particle_Cauchy_stress[0,1],
                                particle_Cauchy_stress[1,0], particle_Cauchy_stress[1,1]
                                )


   
    


    particle_pressure = wp.float64(0.0)
    particle_old_pressure = wp.float64(0.0)
    particle_pressure_avg = wp.float64(0.0)
    particle_old_pressure_avg = wp.float64(0.0)
    particle_grad_new_p = wp.vec2d()

    for i in range(0, 3):
        for j in range(0, 3):
            weight = w[i][0] * w[j][1]
            weight_GRAD = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
            weight_grad = incr_F_inv @ weight_GRAD

            weight_avg = w_avg[i][0] * w_avg[j][1]

            ix = left_bottom_corner_base_int[0] + i
            iy = left_bottom_corner_base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_p = index_ij_x + 2*n_nodes

            node_p_solution = solution[index_ij_p]
            node_p_old_solution = old_solution[index_ij_p]

            particle_pressure += weight * node_p_solution
            particle_old_pressure += weight * node_p_old_solution
            particle_pressure_avg += weight_avg * node_p_solution
            particle_old_pressure_avg += weight_avg * node_p_old_solution
            particle_grad_new_p += weight_grad * node_p_solution



    new_p_vol = p_vol * particle_J
    phi_s_initial = wp.float64(1.0) - wp.float64(0.5) # magic number 0.5 indicates the initial porosity
    new_p_porosity = wp.float64(1.0) - phi_s_initial/particle_J
    new_p_mixture_rho = new_p_porosity*wp.float64(1.0) + (wp.float64(1.0)-new_p_porosity)*p_rho 

    # Get mobility
    kappa_initial = wp.float64(1e-8) * wp.pow(wp.float64(1.0)-wp.float64(0.5), wp.float64(2.0)) / wp.pow(wp.float64(0.5), wp.float64(3.0))
    mobility = wp.float64(1e-6) # Constant


    # PPP stabilization parameter
    tau = wp.float64(0.5)/lame_mu


    # Traction
    traction_y = wp.float64(0.0)
    if top_boundary_flag[p]==True:
        traction_y = wp.float64(-1.0) * dx/PPD 



    # Momentum balance & Mass balance
    for i in range(0, 3):
        for j in range(0, 3):
            weight = w[i][0] * w[j][1]
            weight_GRAD = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
            weight_grad = incr_F_inv @ weight_GRAD # NOTE here is incr_F_inv

            weight_avg = w_avg[i][0] * w_avg[j][1]


            ix = left_bottom_corner_base_int[0] + i
            iy = left_bottom_corner_base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_y = index_ij_x + n_nodes

            # Momentum balance
            rhs_value = (-weight_grad @ (particle_Cauchy_stress_2d - particle_pressure*wp.identity(n=2, dtype=wp.float64)) + weight * new_p_mixture_rho * standard_gravity) * new_p_vol # Updated Lagrangian



            if (dofStruct.boundary_flag_array[index_ij_x]==False and dofStruct.activate_flag_array[index_ij_x]==True):
                wp.atomic_add(rhs, index_ij_x, rhs_value[0])

            if (dofStruct.boundary_flag_array[index_ij_y]==False and dofStruct.activate_flag_array[index_ij_y]==True):
                wp.atomic_add(rhs, index_ij_y, rhs_value[1])
                # wp.atomic_add(rhs, index_ij_y, penalty_residual_y * weight)
                wp.atomic_add(rhs, index_ij_y, weight*traction_y)


            # Mass balance
            index_ij_p = index_ij_x + 2*n_nodes

            rhs_mass_value = (weight * (wp.log(particle_J)-wp.log(particle_old_J))
                              + wp.dot(weight_grad, (mobility * particle_grad_new_p)) * dt
                              ) * new_p_vol

            if (dofStruct.boundary_flag_array[index_ij_p]==False and dofStruct.activate_flag_array[index_ij_p]==True):
                wp.atomic_add(rhs, index_ij_p, rhs_mass_value)


            # PPP stabilization
            rhs_ppp = (tau * (weight-weight_avg) * (particle_pressure - particle_pressure_avg - (particle_old_pressure - particle_old_pressure_avg))) * new_p_vol
            if (dofStruct.boundary_flag_array[index_ij_p]==False and dofStruct.activate_flag_array[index_ij_p]==True):
                wp.atomic_add(rhs, index_ij_p, rhs_ppp)








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

    if row_index<n_nodes: # x dof
        node_idx = wp.mod(row_index, n_grid_x+1)
        node_idy = wp.int((row_index-node_idx)/(n_grid_x+1))
    elif row_index>=n_nodes and row_index<2*n_nodes: # y dof
        node_idx = wp.mod((row_index-n_nodes), n_grid_x+1)
        node_idy = wp.int((row_index-n_nodes)/(n_grid_x+1))
    else: # p dof
        node_idx = wp.mod((row_index-2*n_nodes), n_grid_x+1)
        node_idy = wp.int((row_index-2*n_nodes)/(n_grid_x+1))




    for i in range(5):
        adj_node_idx = node_idx + (i-2)
        for j in range(5):
            adj_node_idy = node_idy + (j-2)

            adj_index_x = adj_node_idx + adj_node_idy*(n_grid_x+1)
            adj_index_y = adj_index_x + n_nodes
            adj_index_p = adj_index_x + 2*n_nodes



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

                    if dofStruct.boundary_flag_array[adj_index_p]==False and dofStruct.activate_flag_array[adj_index_p]==True:
                        rows[50*n_matrix_size + row_index*25 + (i+j*5)] = row_index
                        cols[50*n_matrix_size + row_index*25 + (i+j*5)] = adj_index_p
                        vals[50*n_matrix_size + row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_p]



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
        elif row_index>=n_nodes and row_index<2*n_nodes: # y-dof
            node_idx = wp.mod((row_index-n_nodes), n_grid_x+1)
            node_idy = wp.int((row_index-n_nodes)/(n_grid_x+1))
        else:
            node_idx = wp.mod((row_index-2*n_nodes), n_grid_x+1)
            node_idy = wp.int((row_index-2*n_nodes)/(n_grid_x+1))


        for i in range(5):
            adj_node_idx = node_idx + (i-2)
            for j in range(5):
                adj_node_idy = node_idy + (j-2)
                
                adj_index_x = adj_node_idx + adj_node_idy*(n_grid_x+1)
                adj_index_y = adj_index_x + n_nodes
                adj_index_p = adj_index_x + 2*n_nodes

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

                        if dofStruct.boundary_flag_array[adj_index_p]==False and dofStruct.activate_flag_array[adj_index_p]==True:
                            rows[50*n_matrix_size + row_index*25 + (i+j*5)] = row_index
                            cols[50*n_matrix_size + row_index*25 + (i+j*5)] = adj_index_p
                            vals[50*n_matrix_size + row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_p]
  
                

@wp.kernel
def set_diagnal_component_for_boundary_and_deactivated_dofs(dofStruct: DofStruct,
                                                            rows: wp.array(dtype=wp.int32),
                                                            cols: wp.array(dtype=wp.int32),
                                                            vals: wp.array(dtype=wp.float64),
                                                            n_matrix_size: wp.int32):
    dof_id = wp.tid()

    if dofStruct.boundary_flag_array[dof_id]==True or dofStruct.activate_flag_array[dof_id]==False:
        rows[75*n_matrix_size + dof_id] = dof_id
        cols[75*n_matrix_size + dof_id] = dof_id
        vals[75*n_matrix_size + dof_id] = wp.float64(1.0)




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
             pressure_array: wp.array(dtype=wp.float64),
             inv_dx: wp.float64,
             dx: wp.float64,
             inv_dy: wp.float64,
             dy: wp.float64,
             lame_lambda: wp.float64,
             lame_mu: wp.float64,
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
    left_bottom_corner_base_y = left_bottom_corner[1]*inv_dy + wp.float64(1e-8)
    left_bottom_corner_base_int = wp.vector(wp.int(left_bottom_corner_base_x), wp.int(left_bottom_corner_base_y))
    left_bottom_corner_base = wp.vector(wp.float64(left_bottom_corner_base_int[0]), wp.float64(left_bottom_corner_base_int[1]))



    # GIMP
    GIMP_shape_function_and_gradient_components = get_GIMP_shape_function_and_gradient(x_particles[p], dx, inv_dx, dy, inv_dy, dofStruct, n_grid_x, n_nodes, lpx, lpy)
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
    delta_u_GRAD = wp.matrix(float64_zero, float64_zero,
                             float64_zero, float64_zero, shape=(2,2))
    particle_new_p = wp.float64(0.0)
    for i in range(0, 3):
        for j in range(0, 3):
            weight = w[i][0] * w[j][1]
            weight_grad = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
            ix = left_bottom_corner_base_int[0] + i
            iy = left_bottom_corner_base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_y = index_ij_x + n_nodes
            index_ij_p = index_ij_x + 2*n_nodes

            node_new_solution = wp.vec2d(new_solution[index_ij_x], new_solution[index_ij_y])
            node_old_solution = wp.vec2d(old_solution[index_ij_x], old_solution[index_ij_y])

            delta_u += weight * (node_new_solution - node_old_solution)
            delta_u_GRAD += wp.outer(weight_grad, node_new_solution-node_old_solution)

            particle_new_p += weight * new_solution[index_ij_p]


    # Set old to new
    deformation_gradient_total_old[p] = deformation_gradient_total_new[p]
    left_Cauchy_Green_old[p] = left_Cauchy_Green_new[p]

    delta_u_particles[p] = delta_u
    x_particles[p] += delta_u



    # Post-processing for pressure
    pressure_array[p] = particle_new_p


@wp.kernel
def update_GIMP_lp(deformation_gradient: wp.array(dtype=wp.mat33d),
                   GIMP_lp: wp.array(dtype=wp.vec2d),
                   GIMP_lp_initial: wp.float64,
                   x_particles: wp.array(dtype=wp.vec2d),
                   start_x: wp.float64,
                   start_y: wp.float64,
                   end_x: wp.float64
                   ):
    p = wp.tid()

    # Refer to Eq. (38), (39) of Charlton et al. 2017
    this_particle_F = deformation_gradient[p]
    U00 = wp.sqrt(this_particle_F[0,0]*this_particle_F[0,0] + this_particle_F[1,0]*this_particle_F[1,0] + this_particle_F[2,0]*this_particle_F[2,0])
    U11 = wp.sqrt(this_particle_F[0,1]*this_particle_F[0,1] + this_particle_F[1,1]*this_particle_F[1,1] + this_particle_F[2,1]*this_particle_F[2,1])
    U22 = wp.sqrt(this_particle_F[0,2]*this_particle_F[0,2] + this_particle_F[1,2]*this_particle_F[1,2] + this_particle_F[2,2]*this_particle_F[2,2])

    lpx_updated = GIMP_lp_initial*U00
    lpy_updated = GIMP_lp_initial*U11

    if x_particles[p][0]-lpx_updated < start_x:
        lpx_updated = x_particles[p][0] - start_x - wp.float64(1e-6)
    if x_particles[p][1]-lpy_updated < start_y:
        lpy_updated = x_particles[p][1] - start_y - wp.float64(1e-6)
    

    if x_particles[p][0]+lpx_updated > end_x:
        lpx_updated = end_x - x_particles[p][0] - wp.float64(1e-6)
    
    GIMP_lp[p] = wp.vec2d(lpx_updated, lpy_updated)






# Post-processing parameters
n_iter = 20 


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
start_y = 0.0 
end_x = start_x + dx
end_y = start_y + 10.0 # m





v_particles = wp.array(np.random.rand(n_particles, 2), dtype=wp.vec2d)
deformation_gradient_total_new = wp.array(shape=n_particles, dtype=wp.mat33d)
deformation_gradient_total_old = wp.array(shape=n_particles, dtype=wp.mat33d)
left_Cauchy_Green_new = wp.array(shape=n_particles, dtype=wp.mat33d)
left_Cauchy_Green_old = wp.array(shape=n_particles, dtype=wp.mat33d)
delta_u_particles = wp.zeros(shape=n_particles, dtype=wp.vec2d)
particle_Cauchy_stress_array = wp.zeros(shape=n_particles, dtype=wp.mat33d)
pressure_array = wp.zeros(shape=n_particles, dtype=wp.float64)

top_boundary_flag = wp.zeros(shape=n_particles, dtype=wp.bool)


grid_P2G_m = wp.zeros(shape=(n_grid_x+1, n_grid_y+1), dtype=wp.float64)
grid_P2G_p = wp.zeros(shape=(n_grid_x+1, n_grid_y+1), dtype=wp.float64)


GIMP_lp = wp.zeros(shape=n_particles, dtype=wp.vec2d)


# Matrix
n_nodes = (n_grid_x+1) * (n_grid_y+1)
n_matrix_size = 3 * n_nodes # 3 includes the spatial dimension (2) plus pore water pressure
bsr_matrix = wps.bsr_zeros(n_matrix_size, n_matrix_size, block_type=wp.float64)
rhs = wp.zeros(shape=n_matrix_size, dtype=wp.float64, requires_grad=True)
increment = wp.zeros(shape=n_matrix_size, dtype=wp.float64)
old_solution = wp.zeros(shape=n_matrix_size, dtype=wp.float64)
new_solution = wp.zeros(shape=n_matrix_size, dtype=wp.float64, requires_grad=True)

rows = wp.zeros(shape=3*25*n_matrix_size+n_matrix_size, dtype=wp.int32) # 3 includes the spatial dimension (2) and pore water pressure
cols = wp.zeros(shape=3*25*n_matrix_size+n_matrix_size, dtype=wp.int32)
vals = wp.zeros(shape=3*25*n_matrix_size+n_matrix_size, dtype=wp.float64)

# Consistent P2G
rhs_P2G = wp.zeros(shape=n_nodes, dtype=wp.float64)
rows_P2G = wp.zeros(shape=81*n_particles+n_nodes, dtype=wp.int32)
cols_P2G = wp.zeros(shape=81*n_particles+n_nodes, dtype=wp.int32)
vals_P2G = wp.zeros(shape=81*n_particles+n_nodes, dtype=wp.float64)



dofStruct = DofStruct()
dofStruct.activate_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)
dofStruct.boundary_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)



# Generate particles
particle_pos_np = init_particle_position(start_x, start_y, end_x, end_y, dx, n_grid_x, n_grid_y, PPD, n_particles)



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
selector_p_list = []
e_x_list = []
e_y_list = []
e_p_list = []
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

        # p 
        selector_p = selector_x + 2*n_nodes
        selector_p_resize = selector_p + 0
        selector_p_resize.resize(max_selector_length)
        selector_p_wp = wp.from_numpy(selector_p_resize, dtype=wp.int32)
        selector_p_list.append(selector_p_wp)

        select_index = np.zeros(n_matrix_size)
        select_index[selector_p] = 1.
        e = wp.array(select_index, dtype=wp.float64)
        e_p_list.append(e)




# Init particles
x_particles = wp.from_numpy(particle_pos_np, dtype=wp.vec2d)


# Initialization
wp.launch(kernel=initialization,
          dim=n_particles,
          inputs=[deformation_gradient_total_new, deformation_gradient_total_old, left_Cauchy_Green_new, left_Cauchy_Green_old, x_particles, v_particles, top_boundary_flag, end_x, end_y, dx, PPD])

wp.launch(kernel=initialize_GIMP_lp,
              dim=n_particles,
              inputs=[GIMP_lp, GIMP_lp_initial])


pressure_array.zero_()


# Post-processing
x_numpy = np.array(x_particles.numpy())
output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': particle_Cauchy_stress_array.numpy()[:,1,1], 'pressure': pressure_array.numpy()})
output_particles.write("terzaghi_particles_%d.vtk" % 0)




dt = 100.0 # s
total_time = 0.0

top_disp_time_array = np.empty(0)

for step in range(600):
    print('Load step', step)

    x_beginning_numpy = x_particles.numpy()


    grid_P2G_m.zero_()
    grid_P2G_p.zero_()


    old_solution.zero_()
    new_solution.zero_()
    increment.zero_()


    # P2G
    dofStruct.activate_flag_array.zero_()
    dofStruct.boundary_flag_array.zero_()

    rhs_P2G.zero_()
    rows_P2G.zero_()
    cols_P2G.zero_()
    vals_P2G.zero_()
    wp.launch(kernel=set_boundary_dofs,
              dim=(n_grid_x+1, n_grid_y+1),
              inputs=[dofStruct, n_grid_x, n_nodes, dx, dy, end_x])

    wp.launch(kernel=P2G,
              dim=n_particles,
              inputs=[x_particles, inv_dx, dx, inv_dy, dy, n_grid_x, n_nodes, dofStruct, grid_P2G_m, grid_P2G_p, p_mass, pressure_array, top_boundary_flag, GIMP_lp, rhs_P2G, rows_P2G, cols_P2G, vals_P2G])

    wp.launch(kernel=P2G_impose_p_boundary,
              dim=(n_grid_x+1, n_grid_y+1),
              inputs=[dofStruct, n_grid_x, n_nodes, rows_P2G, cols_P2G, vals_P2G, n_particles])

    # wp.launch(kernel=P2G_solver_lumped,
    #           dim=(n_grid_x+1, n_grid_y+1),
    #           inputs=[dofStruct, n_grid_x, n_nodes, grid_P2G_m, grid_P2G_p, old_solution, new_solution])




    # Solve for consistent mass matrix using pyamg
    bsr_matrix_P2G = sparse.coo_matrix((vals_P2G.numpy(), (rows_P2G.numpy(), cols_P2G.numpy())), shape=(n_nodes, n_nodes)).asformat('csr')
    mls = smoothed_aggregation_solver(bsr_matrix_P2G)

    b = rhs_P2G.numpy()
    residuals = []
    x_P2G_pyamg = mls.solve(b, tol=1e-8, accel='bicgstab', residuals=residuals)

    x_P2G_warp = wp.from_numpy(x_P2G_pyamg, dtype=wp.float64)
    print('After solving consistent-mass P2G')


    wp.launch(kernel=P2G_solver_consistent,
              dim=n_nodes,
              inputs=[dofStruct, n_nodes, x_P2G_warp, old_solution, new_solution])



    boundary_flag_array_np = dofStruct.boundary_flag_array.numpy()
    activate_flag_array_np = dofStruct.activate_flag_array.numpy()



    # Newton iteration
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
                      inputs=[x_particles, inv_dx, dx, inv_dy, dy, rhs, new_solution, old_solution, p_vol, p_rho, lame_lambda, lame_mu, poisson_ratio, deformation_gradient_total_new, deformation_gradient_total_old, left_Cauchy_Green_new, left_Cauchy_Green_old, n_grid_x, n_nodes, dofStruct, step, particle_Cauchy_stress_array, GIMP_lp, dt, PPD, n_particles, top_boundary_flag])
            


        # Sparse differentiation
        pattern_id = 0
        for c_iter in range(5):
            for r_iter in range(5):
                current_node_id = c_iter + r_iter * (n_grid_x+1)
                select_index = np.zeros(n_matrix_size)

                # x
                tape.backward(grads={rhs: e_x_list[pattern_id]})
                jacobian_wp = tape.gradients[new_solution]
                wp.launch(kernel=from_jacobian_to_vector_parallel,
                          dim=max_selector_length,
                          inputs=[jacobian_wp, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, selector_x_list[pattern_id], dofStruct])
                tape.zero()

                # y
                tape.backward(grads={rhs: e_y_list[pattern_id]})
                jacobian_wp = tape.gradients[new_solution]
                wp.launch(kernel=from_jacobian_to_vector_parallel,
                          dim=max_selector_length,
                          inputs=[jacobian_wp, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, selector_y_list[pattern_id], dofStruct])
                tape.zero()

                # p  
                tape.backward(grads={rhs: e_p_list[pattern_id]})
                jacobian_wp = tape.gradients[new_solution]
                wp.launch(kernel=from_jacobian_to_vector_parallel,
                          dim=max_selector_length,
                          inputs=[jacobian_wp, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, selector_p_list[pattern_id], dofStruct])
                tape.zero()

                pattern_id = pattern_id + 1

        
        tape.reset()




        # Adjust diagonal components of the global matrix
        wp.launch(kernel=set_diagnal_component_for_boundary_and_deactivated_dofs,
                  dim=n_matrix_size,
                  inputs=[dofStruct, rows, cols, vals, n_matrix_size])


        
        

        # Scipy direct solver
        bsr_matrix_other = sparse.coo_matrix((vals.numpy(), (rows.numpy(), cols.numpy())), shape=(n_matrix_size, n_matrix_size)).asformat('csr')
        b = rhs.numpy()
        bsr_matrix_other_array = bsr_matrix_other.toarray()
        x_direct = spsolve(bsr_matrix_other, b)

        increment = wp.from_numpy(x_direct, dtype=wp.float64)



        # From increment to solution
        wp.launch(kernel=from_increment_to_solution,
                  dim=n_matrix_size,
                  inputs=[increment, new_solution])


        with np.printoptions(threshold=np.inf):
            print('residual.norm:', np.linalg.norm(rhs.numpy()))



        if np.linalg.norm(rhs.numpy())<1e-8:
            break


    # G2P
    wp.launch(kernel=G2P_GIMP, #G2P,
              dim=n_particles,
              inputs=[x_particles, delta_u_particles, deformation_gradient_total_new, deformation_gradient_total_old, left_Cauchy_Green_new, left_Cauchy_Green_old, particle_Cauchy_stress_array, pressure_array, inv_dx, dx, inv_dy, dy, lame_lambda, lame_mu, n_grid_x, n_nodes, new_solution, old_solution, dofStruct, GIMP_lp])


    wp.launch(kernel=update_GIMP_lp,
              dim=n_particles,
              inputs=[deformation_gradient_total_new, GIMP_lp, GIMP_lp_initial, x_particles, start_x, start_y, end_x])

   

    # Post-processing
    x_numpy = np.array(x_particles.numpy())
    
    
    # Plot out the pressure
    if step==54 or step==110 or step==221 or step==332 or step==554:
        pressure_array_np = pressure_array.numpy()
        for p in range(len(x_numpy)):
            print(x_numpy[p][1]-start_y, pressure_array_np[p], ";")




    total_time += dt

    print('Time:', total_time)





