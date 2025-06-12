
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

import time

import pypardiso

# Implicit MPM solver for 3D indentation or a rigid footing into porous ground
# Contact: Yidong Zhao (ydzhao94@gmail.com)


wp.init()
device = "cuda:0"
wp.set_device(device)


n_particles = 512000 
n_grid_x = 24 
n_grid_y = 24 
n_grid_z = 24 
grid_size = (n_grid_x, n_grid_y, n_grid_z)

max_x = 6.0 # m
dx = max_x/n_grid_x # dx = 0.25 m
dy = max_x/n_grid_x
dz = max_x/n_grid_x
inv_dx = float(n_grid_x/max_x)
inv_dy = float(n_grid_x/max_x)
inv_dz = float(n_grid_x/max_x)

dt = 0.1 # s

PPD = 4 # particle per direction
GIMP_lp_initial = dx/PPD/2.0 


youngs_modulus = 1e4 #1e3 # kPa
poisson_ratio = 0.3
lame_mu = youngs_modulus / (2.0*(1.0+poisson_ratio))
lame_lambda = youngs_modulus*poisson_ratio / ((1.0+poisson_ratio) * (1.0-2.0*poisson_ratio))

p_vol = (dx/PPD)**3
p_rho = 2.0 # t/m^3
p_mass = p_vol * p_rho





# To check whether a specific dof index is activated or at the Dirichlet boundary
@wp.struct
class DofStruct:
    activate_flag_array: wp.array(dtype=wp.bool)
    boundary_flag_array: wp.array(dtype=wp.bool)


# initialization
@wp.kernel
def initialization(deformation_gradient: wp.array(dtype=wp.mat33d),
                   x_particles: wp.array(dtype=wp.vec3d),
                   particle_boundary_flag_array: wp.array(dtype=wp.bool),
                   end_y: wp.float64,
                   end_z: wp.float64,
                   dx: wp.float64,
                   PPD: wp.float64,
                   start_x: wp.float64,
                   start_z: wp.float64
                   ):
    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)

    identity_matrix = wp.matrix(
                      float64_one, float64_zero, float64_zero,
                      float64_zero, float64_one, float64_zero,
                      float64_zero, float64_zero, float64_one,
                      shape=(3,3)
        )

    deformation_gradient[p] = identity_matrix

    # Setup boundary particle flag
    if x_particles[p][2]>end_z-dx/PPD:
        particle_boundary_flag_array[p] = True
    else:
        particle_boundary_flag_array[p] = False



@wp.kernel
def initialize_GIMP_lp(GIMP_lp: wp.array(dtype=wp.vec3d),
                       GIMP_lp_initial: wp.float64):
    p = wp.tid()

    GIMP_lp[p] = wp.vec3d(GIMP_lp_initial, GIMP_lp_initial, GIMP_lp_initial)
    


@wp.kernel
def initialize_youngs_modulus_diff(youngs_modulus_diff: wp.array(dtype=wp.float64),
                                   youngs_modulus: wp.float64
                                   ):
    youngs_modulus_diff[0] = youngs_modulus



# Set boundary dofs
@wp.kernel
def set_boundary_dofs(dofStruct: DofStruct,
                      n_grid_x: wp.int32,
                      n_grid_y: wp.int32,
                      n_nodes: wp.int32,
                      dx: wp.float64,
                      dy: wp.float64,
                      end_x: wp.float64,
                      end_y: wp.float64
                      ):
    
    node_idx, node_idy, node_idz = wp.tid()
    dof_x = node_idx + node_idy*(n_grid_x+1) + node_idz*((n_grid_x+1)*(n_grid_y+1))
    dof_y = dof_x + n_nodes
    dof_z = dof_x + 2*n_nodes
    dof_p = dof_x + 3*n_nodes

    if node_idx<=1 or (wp.float64(node_idx)+wp.float64(0.5))*dx>end_x:
        dofStruct.boundary_flag_array[dof_x] = True
        
    if node_idy<=1 or (wp.float64(node_idy)+wp.float64(0.5))*dy>end_y:
        dofStruct.boundary_flag_array[dof_y] = True

    if node_idz<=0:
        dofStruct.boundary_flag_array[dof_z] = True

    # # Dry case
    # dofStruct.boundary_flag_array[dof_p] = True



@wp.func
def get_GIMP_shape_function_and_gradient(xp: wp.vec3d,
                                         dx: wp.float64,
                                         dy: wp.float64,
                                         dz: wp.float64,
                                         inv_dx: wp.float64,
                                         inv_dy: wp.float64,
                                         inv_dz: wp.float64,
                                         dofStruct: DofStruct,
                                         n_grid_x: wp.int32,
                                         n_grid_y: wp.int32,
                                         n_nodes: wp.int32,
                                         GIMP_lpx: wp.float64,
                                         GIMP_lpy: wp.float64,
                                         GIMP_lpz: wp.float64):
    
    bottom_corner = xp - wp.vec3d(GIMP_lpx, GIMP_lpy, GIMP_lpz)
    bottom_corner_base_x = bottom_corner[0]*inv_dx + wp.float64(1e-8)
    bottom_corner_base_y = bottom_corner[1]*inv_dy + wp.float64(1e-8)
    bottom_corner_base_z = bottom_corner[2]*inv_dz + wp.float64(1e-8)
    bottom_corner_base_int = wp.vec3i(wp.int(bottom_corner_base_x), wp.int(bottom_corner_base_y), wp.int(bottom_corner_base_z))
    bottom_corner_base = wp.vec3d(wp.float64(bottom_corner_base_int[0]), wp.float64(bottom_corner_base_int[1]), wp.float64(bottom_corner_base_int[2]))

    up_corner = xp + wp.vec3d(GIMP_lpx, GIMP_lpy, GIMP_lpz)
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
        wx0 = wp.pow((dx+GIMP_lpx-wp.abs(xp[0]-x_0)), wp.float64(2.0))/(wp.float64(4)*dx*GIMP_lpx)
        grad_wx0 = -(dx+GIMP_lpx-wp.abs(xp[0]-x_0))/(wp.float64(2)*dx*GIMP_lpx) * (xp[0]-x_0)/wp.abs(xp[0]-x_0)

        x_1 = (bottom_corner_base[0] + wp.float64(1)) * dx
        wx1 = wp.float64(1.0) - (wp.pow((xp[0]-x_1), wp.float64(2.0)) + wp.pow(GIMP_lpx, wp.float64(2.0)))/(wp.float64(2)*dx*GIMP_lpx)
        grad_wx1 = -(xp[0]-x_1)/(dx*GIMP_lpx)

        x_2 = (bottom_corner_base[0] + wp.float64(2)) * dx
        wx2 = wp.pow((dx+GIMP_lpx-wp.abs(xp[0]-x_2)), wp.float64(2.0))/(wp.float64(4)*dx*GIMP_lpx)
        grad_wx2 = -(dx+GIMP_lpx-wp.abs(xp[0]-x_2))/(wp.float64(2)*dx*GIMP_lpx) * (xp[0]-x_2)/wp.abs(xp[0]-x_2)


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
        wy0 = wp.pow((dy+GIMP_lpy-wp.abs(xp[1]-y_0)), wp.float64(2.0))/(wp.float64(4)*dy*GIMP_lpy)
        grad_wy0 = -(dy+GIMP_lpy-wp.abs(xp[1]-y_0))/(wp.float64(2.0)*dy*GIMP_lpy) * (xp[1]-y_0)/wp.abs(xp[1]-y_0)

        y_1 = (bottom_corner_base[1] + wp.float64(1)) * dy
        wy1 = wp.float64(1.0) - (wp.pow((xp[1]-y_1), wp.float64(2.0)) + wp.pow(GIMP_lpy, wp.float64(2.0)))/(wp.float64(2)*dy*GIMP_lpy)
        grad_wy1 = -(xp[1]-y_1)/(dy*GIMP_lpy)

        y_2 = (bottom_corner_base[1] + wp.float64(2)) * dy
        wy2 = wp.pow((dy+GIMP_lpy-wp.abs(xp[1]-y_2)), wp.float64(2.0))/(wp.float64(4)*dy*GIMP_lpy)
        grad_wy2 = -(dy+GIMP_lpy-wp.abs(xp[1]-y_2))/(wp.float64(2.0)*dy*GIMP_lpy) * (xp[1]-y_2)/wp.abs(xp[1]-y_2)


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
        wz0 = wp.pow((dz+GIMP_lpz-wp.abs(xp[2]-z_0)), wp.float64(2.0))/(wp.float64(4)*dz*GIMP_lpz)
        grad_wz0 = -(dz+GIMP_lpz-wp.abs(xp[2]-z_0))/(wp.float64(2.0)*dz*GIMP_lpz) * (xp[2]-z_0)/wp.abs(xp[2]-z_0)

        z_1 = (bottom_corner_base[2] + wp.float64(1)) * dz
        wz1 = wp.float64(1.0) - (wp.pow((xp[2]-z_1), wp.float64(2.0)) + wp.pow(GIMP_lpz, wp.float64(2.0)))/(wp.float64(2)*dz*GIMP_lpz)
        grad_wz1 = -(xp[2]-z_1)/(dz*GIMP_lpz)

        z_2 = (bottom_corner_base[2] + wp.float64(2)) * dz
        wz2 = wp.pow((dz+GIMP_lpz-wp.abs(xp[2]-z_2)), wp.float64(2.0))/(wp.float64(4)*dz*GIMP_lpz)
        grad_wz2 = -(dz+GIMP_lpz-wp.abs(xp[2]-z_2))/(wp.float64(2.0)*dz*GIMP_lpz) * (xp[2]-z_2)/wp.abs(xp[2]-z_2)



    return wp.vector(wx0, wy0, wz0, wx1, wy1, wz1, wx2, wy2, wz2, grad_wx0, grad_wy0, grad_wz0, grad_wx1, grad_wy1, grad_wz1, grad_wx2, grad_wy2, grad_wz2)



@wp.func
def get_GIMP_shape_function_and_gradient_avg(xp: wp.vec3d,
                                             dx: wp.float64,
                                             dy: wp.float64,
                                             dz: wp.float64,
                                             inv_dx: wp.float64,
                                             inv_dy: wp.float64,
                                             inv_dz: wp.float64,
                                             dofStruct: DofStruct,
                                             n_grid_x: wp.int32,
                                             n_grid_y: wp.int32,
                                             n_nodes: wp.int32,
                                             GIMP_lpx: wp.float64,
                                             GIMP_lpy: wp.float64,
                                             GIMP_lpz: wp.float64
                                             ):
    bottom_corner = xp - wp.vec3d(GIMP_lpx, GIMP_lpy, GIMP_lpz)
    bottom_corner_base_x = bottom_corner[0]*inv_dx + wp.float64(1e-8)
    bottom_corner_base_y = bottom_corner[1]*inv_dy + wp.float64(1e-8)
    bottom_corner_base_z = bottom_corner[2]*inv_dz + wp.float64(1e-8)
    bottom_corner_base_int = wp.vec3i(wp.int(bottom_corner_base_x), wp.int(bottom_corner_base_y), wp.int(bottom_corner_base_z))
    bottom_corner_base = wp.vec3d(wp.float64(bottom_corner_base_int[0]), wp.float64(bottom_corner_base_int[1]), wp.float64(bottom_corner_base_int[2]))

    up_corner = xp + wp.vec3d(GIMP_lpx, GIMP_lpy, GIMP_lpz)
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
        wx0 = (dx+GIMP_lpx-wp.abs(xp[0]-x_0))/(wp.float64(4.0)*GIMP_lpx)
        grad_wx0 = wp.float64(-1.0)/(wp.float64(4.0)*GIMP_lpx) * wp.sign(xp[0]-x_0)

        wx1 = wp.float64(0.5) 

        x_2 = (bottom_corner_base[0] + wp.float64(2)) * dx
        wx2 = (dx+GIMP_lpx-wp.abs(xp[0]-x_2))/(wp.float64(4.0)*GIMP_lpx)
        grad_wx2 = wp.float64(-1.0)/(wp.float64(4.0)*GIMP_lpx) * wp.sign(xp[0]-x_2)

    # get shape function y
    if up_corner_base_int[1]==bottom_corner_base_int[1]:
        wy0 = wp.float64(0.5)
        wy1 = wp.float64(0.5)
    else:
        # the particle will influence 2 elements in the y direction
        # shape function & gradient
        y_0 = (bottom_corner_base[1]) * dy
        wy0 = (dy+GIMP_lpy-wp.abs(xp[1]-y_0))/(wp.float64(4.0)*GIMP_lpy)
        grad_wy0 = wp.float64(-1.0)/(wp.float64(4.0)*GIMP_lpy) * wp.sign(xp[1]-y_0)

        wy1 = wp.float64(0.5) 

        y_2 = (bottom_corner_base[1] + wp.float64(2)) * dy
        wy2 = (dy+GIMP_lpy-wp.abs(xp[1]-y_2))/(wp.float64(4.0)*GIMP_lpy)
        grad_wy2 = wp.float64(-1.0)/(wp.float64(4.0)*GIMP_lpy) * wp.sign(xp[1]-y_2)

    # get shape function z
    if up_corner_base_int[2]==bottom_corner_base_int[2]:
        wz0 = wp.float64(0.5)
        wz1 = wp.float64(0.5)
    else:
        # the particle will influence 2 elements in the z direction
        # shape function & gradient
        z_0 = (bottom_corner_base[2]) * dz
        wz0 = (dz+GIMP_lpz-wp.abs(xp[2]-z_0))/(wp.float64(4.0)*GIMP_lpz)
        grad_wz0 = wp.float64(-1.0)/(wp.float64(4.0)*GIMP_lpz) * wp.sign(xp[2]-z_0) 

        wz1 = wp.float64(0.5)

        z_2 = (bottom_corner_base[2] + wp.float64(2)) * dz
        wz2 = (dz+GIMP_lpz-wp.abs(xp[2]-z_2))/(wp.float64(4.0)*GIMP_lpz)
        grad_wz2 = wp.float64(-1.0)/(wp.float64(4.0)*GIMP_lpz) * wp.sign(xp[2]-z_2)


    return wp.vector(wx0, wy0, wz0, wx1, wy1, wz1, wx2, wy2, wz2, grad_wx0, grad_wy0, grad_wz0, grad_wx1, grad_wy1, grad_wz1, grad_wx2, grad_wy2, grad_wz2)


# P2G
@wp.kernel
def P2G(x_particles: wp.array(dtype=wp.vec3d),
        inv_dx: wp.float64,
        inv_dy: wp.float64,
        inv_dz: wp.float64,
        dx: wp.float64,
        dy: wp.float64,
        dz: wp.float64,
        n_grid_x: wp.int32,
        n_grid_y: wp.int32,
        n_nodes: wp.int32,
        dofStruct: DofStruct,
        GIMP_lp: wp.array(dtype=wp.vec3d),
        particle_boundary_flag_array: wp.array(dtype=wp.bool),
        rhs_P2G: wp.array(dtype=wp.float64),
        rows_P2G: wp.array(dtype=wp.int32),
        cols_P2G: wp.array(dtype=wp.int32),
        vals_P2G: wp.array(dtype=wp.float64),
        p_mass: wp.float64,
        particle_pressure_array: wp.array(dtype=wp.float64)
        ):
    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    GIMP_lpx = GIMP_lp[p][0]
    GIMP_lpy = GIMP_lp[p][1]
    GIMP_lpz = GIMP_lp[p][2]

    # Calculate shape functions
    # GIMP
    xp = x_particles[p]
    bottom_corner = xp - wp.vec3d(GIMP_lpx, GIMP_lpy, GIMP_lpz)
    bottom_corner_base_x = bottom_corner[0]*inv_dx + wp.float64(1e-8)
    bottom_corner_base_y = bottom_corner[1]*inv_dy + wp.float64(1e-8)
    bottom_corner_base_z = bottom_corner[2]*inv_dz + wp.float64(1e-8)
    bottom_corner_base_int = wp.vec3i(wp.int(bottom_corner_base_x), wp.int(bottom_corner_base_y), wp.int(bottom_corner_base_z))
    bottom_corner_base = wp.vec3d(wp.float64(bottom_corner_base_int[0]), wp.float64(bottom_corner_base_int[1]), wp.float64(bottom_corner_base_int[2]))

    up_corner = xp + wp.vec3d(GIMP_lpx, GIMP_lpy, GIMP_lpz)
    up_corner_base_x = up_corner[0]*inv_dx + wp.float64(1e-8)
    up_corner_base_y = up_corner[1]*inv_dy + wp.float64(1e-8)
    up_corner_base_z = up_corner[2]*inv_dz + wp.float64(1e-8)
    up_corner_base_int = wp.vec3i(wp.int(up_corner_base_x), wp.int(up_corner_base_y), wp.int(up_corner_base_z))
    up_corner_base = wp.vec3d(wp.float64(up_corner_base_int[0]), wp.float64(up_corner_base_int[1]), wp.float64(up_corner_base_int[2]))

    GIMP_f_gradf = get_GIMP_shape_function_and_gradient(xp, dx, dy, dz, inv_dx, inv_dy, inv_dz, dofStruct, n_grid_x, n_grid_y, n_nodes, GIMP_lpx, GIMP_lpy, GIMP_lpz)
    wx0 = GIMP_f_gradf[0]
    wy0 = GIMP_f_gradf[1]
    wz0 = GIMP_f_gradf[2]
    wx1 = GIMP_f_gradf[3]
    wy1 = GIMP_f_gradf[4]
    wz1 = GIMP_f_gradf[5]
    wx2 = GIMP_f_gradf[6]
    wy2 = GIMP_f_gradf[7]
    wz2 = GIMP_f_gradf[8]

    w = wp.matrix(
        wx0, wy0, wz0,
        wx1, wy1, wz1,
        wx2, wy2, wz2,
        shape=(3,3)
        )


    # Loop on grid nodes
    # Flattened loop
    for flattened_id in range(27):
        i = wp.int(flattened_id/9)
        j = wp.mod(wp.int(flattened_id/3), 3)
        k = wp.mod(flattened_id, 3)
    # for i in range(0, 3):
    #     for j in range(0, 3):
    #         for k in range(0, 3):

        ix = bottom_corner_base_int[0] + i
        iy = bottom_corner_base_int[1] + j
        iz = bottom_corner_base_int[2] + k

        index_ij_x = ix + iy*(n_grid_x+1) + iz*((n_grid_x+1)*(n_grid_y+1))
        index_ij_y = index_ij_x + n_nodes
        index_ij_z = index_ij_x + 2*n_nodes
        index_ij_p = index_ij_x + 3*n_nodes

        weight = w[i][0] * w[j][1] * w[k][2]

        if weight>wp.float64(1e-7):
            dofStruct.activate_flag_array[index_ij_x] = True
            dofStruct.activate_flag_array[index_ij_y] = True
            dofStruct.activate_flag_array[index_ij_z] = True
            dofStruct.activate_flag_array[index_ij_p] = True


        # Set top pressure boundary flag
        if particle_boundary_flag_array[p]==True:
            node_pos = wp.vec3d(wp.float64(ix)*dx, wp.float64(iy)*dy, wp.float64(iz)*dz)
            if node_pos[2]>=x_particles[p][2]:
                dofStruct.boundary_flag_array[index_ij_p] = True



    # Pressure P2G
    # Flattened
    # for i in range(0, 3):
    #     for j in range(0, 3):
    #         for k in range(0, 3):
    #             for ii in range(0, 3):
    #                 for jj in range(0, 3):
    #                     for kk in range(0, 3):
    #                         todo

    for flattened_id in range(729):
        i = flattened_id / 243
        j = wp.mod(flattened_id/81, 3)
        k = wp.mod(flattened_id/27, 3)
        ii = wp.mod(flattened_id/9, 3)
        jj = wp.mod(flattened_id/3, 3)
        kk = wp.mod(flattened_id, 3)

        weight = w[i][0] * w[j][1] * w[k][2]
        ix = bottom_corner_base_int[0] + i
        jy = bottom_corner_base_int[1] + j
        kz = bottom_corner_base_int[2] + k

        index_ij_scalar = ix + jy*(n_grid_x+1) + kz*((n_grid_x+1)*(n_grid_y+1))
        index_ij_p = index_ij_scalar + 3*n_nodes

        weight_iijj = w[ii][0] * w[jj][1] * w[kk][2]
        iix = bottom_corner_base_int[0] + ii
        jjy = bottom_corner_base_int[1] + jj
        kkz = bottom_corner_base_int[2] + kk

        index_iijj_scalar = iix + jjy*(n_grid_x+1) + kkz*((n_grid_x+1)*(n_grid_y+1))
        index_iijj_p = index_iijj_scalar + 3*n_nodes

        wp.atomic_add(rhs_P2G, index_ij_scalar, weight * weight_iijj * p_mass * particle_pressure_array[p])

        rows_P2G[p*27*27 + (i+j*3+k*9)*27 + (ii+jj*3+kk*9)] = index_ij_scalar
        cols_P2G[p*27*27 + (i+j*3+k*9)*27 + (ii+jj*3+kk*9)] = index_iijj_scalar
        vals_P2G[p*27*27 + (i+j*3+k*9)*27 + (ii+jj*3+kk*9)] = weight * weight_iijj * p_mass


        # # Lumped
        # if i==ii and j==jj and k==kk:
        #     wp.atomic_add(rhs_P2G, index_ij_scalar, weight * p_mass * particle_pressure_array[p])
        #     rows_P2G[p*27*27 + (i+j*3+k*9)*27 + (ii+jj*3+kk*9)] = index_ij_scalar
        #     cols_P2G[p*27*27 + (i+j*3+k*9)*27 + (ii+jj*3+kk*9)] = index_iijj_scalar
        #     vals_P2G[p*27*27 + (i+j*3+k*9)*27 + (ii+jj*3+kk*9)] = weight  * p_mass




@wp.kernel
def P2G_solution_to_solution(dofStruct: DofStruct,
                             n_nodes: wp.int32,
                             P2G_solution: wp.array(dtype=wp.float64),
                             old_solution: wp.array(dtype=wp.float64),
                             new_solution: wp.array(dtype=wp.float64)
                             ):
    dof_scalar = wp.tid()
    dof_p = dof_scalar + 3*n_nodes

    if dofStruct.boundary_flag_array[dof_p]==False and dofStruct.activate_flag_array[dof_p]==True:
        old_solution[dof_p] = P2G_solution[dof_scalar]
        new_solution[dof_p] = P2G_solution[dof_scalar]




# Assemble residual with tape
@wp.kernel
def assemble_residual(x_particles: wp.array(dtype=wp.vec3d),
                      inv_dx: wp.float64,
                      inv_dy: wp.float64,
                      inv_dz: wp.float64,
                      dx: wp.float64,
                      dy: wp.float64,
                      dz: wp.float64,
                      rhs: wp.array(dtype=wp.float64),
                      solution: wp.array(dtype=wp.float64),
                      old_solution: wp.array(dtype=wp.float64),
                      p_vol: wp.float64,
                      p_rho: wp.float64,
                      youngs_modulus_diff: wp.array(dtype=wp.float64),
                      poisson_ratio: wp.float64,
                      deformation_gradient: wp.array(dtype=wp.mat33d),
                      particle_boundary_flag_array: wp.array(dtype=wp.bool),
                      n_grid_x: wp.int32,
                      n_grid_y: wp.int32,
                      n_nodes: wp.int32,
                      dofStruct: DofStruct,
                      step: wp.float64,
                      PPD: wp.float64,
                      end_y: wp.float64,
                      end_z: wp.float64,
                      GIMP_lp: wp.array(dtype=wp.vec3d),
                      dt: wp.float64,
                      indentation_force_array: wp.array(dtype=wp.float64),
                      indentation_force: wp.array(dtype=wp.float64),
                      n_particles: wp.int32
                      ):
    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    lame_lambda = youngs_modulus_diff[0]*poisson_ratio / ((wp.float64(1.0)+poisson_ratio) * (wp.float64(1.0)-wp.float64(2.0)*poisson_ratio))
    lame_mu = youngs_modulus_diff[0] / (wp.float64(2.0)*(wp.float64(1.0)+poisson_ratio))

    standard_gravity = wp.vec3d(float64_zero, float64_zero, float64_zero) #wp.vec3d(float64_zero, wp.float64(-10.0)*(step+float64_one)/wp.float64(40.0), float64_zero)

    GIMP_lpx = GIMP_lp[p][0]
    GIMP_lpy = GIMP_lp[p][1]
    GIMP_lpz = GIMP_lp[p][2]

    # Calculate shape functions
    # GIMP
    xp = x_particles[p]
    bottom_corner = xp - wp.vec3d(GIMP_lpx, GIMP_lpy, GIMP_lpz)
    bottom_corner_base_x = bottom_corner[0]*inv_dx + wp.float64(1e-8)
    bottom_corner_base_y = bottom_corner[1]*inv_dy + wp.float64(1e-8)
    bottom_corner_base_z = bottom_corner[2]*inv_dz + wp.float64(1e-8)
    bottom_corner_base_int = wp.vec3i(wp.int(bottom_corner_base_x), wp.int(bottom_corner_base_y), wp.int(bottom_corner_base_z))
    bottom_corner_base = wp.vec3d(wp.float64(bottom_corner_base_int[0]), wp.float64(bottom_corner_base_int[1]), wp.float64(bottom_corner_base_int[2]))

    up_corner = xp + wp.vec3d(GIMP_lpx, GIMP_lpy, GIMP_lpz)
    up_corner_base_x = up_corner[0]*inv_dx + wp.float64(1e-8)
    up_corner_base_y = up_corner[1]*inv_dy + wp.float64(1e-8)
    up_corner_base_z = up_corner[2]*inv_dz + wp.float64(1e-8)
    up_corner_base_int = wp.vec3i(wp.int(up_corner_base_x), wp.int(up_corner_base_y), wp.int(up_corner_base_z))
    up_corner_base = wp.vec3d(wp.float64(up_corner_base_int[0]), wp.float64(up_corner_base_int[1]), wp.float64(up_corner_base_int[2]))


    GIMP_f_gradf = get_GIMP_shape_function_and_gradient(xp, dx, dy, dz, inv_dx, inv_dy, inv_dz, dofStruct, n_grid_x, n_grid_y, n_nodes, GIMP_lpx, GIMP_lpy, GIMP_lpz)
    wx0 = GIMP_f_gradf[0]
    wy0 = GIMP_f_gradf[1]
    wz0 = GIMP_f_gradf[2]
    wx1 = GIMP_f_gradf[3]
    wy1 = GIMP_f_gradf[4]
    wz1 = GIMP_f_gradf[5]
    wx2 = GIMP_f_gradf[6]
    wy2 = GIMP_f_gradf[7]
    wz2 = GIMP_f_gradf[8]
    grad_wx0 = GIMP_f_gradf[9]
    grad_wy0 = GIMP_f_gradf[10]
    grad_wz0 = GIMP_f_gradf[11]
    grad_wx1 = GIMP_f_gradf[12]
    grad_wy1 = GIMP_f_gradf[13]
    grad_wz1 = GIMP_f_gradf[14]
    grad_wx2 = GIMP_f_gradf[15]
    grad_wy2 = GIMP_f_gradf[16]
    grad_wz2 = GIMP_f_gradf[17]

    
    w = wp.matrix(
        wx0, wy0, wz0,
        wx1, wy1, wz1,
        wx2, wy2, wz2,
        shape=(3,3)
        )

    grad_w = wp.matrix(
        grad_wx0, grad_wy0, grad_wz0,
        grad_wx1, grad_wy1, grad_wz1,
        grad_wx2, grad_wy2, grad_wz2,
        shape=(3,3)
        )


    GIMP_f_gradf_avg = get_GIMP_shape_function_and_gradient_avg(xp, dx, dy, dz, inv_dx, inv_dy, inv_dz, dofStruct, n_grid_x, n_grid_y, n_nodes, GIMP_lpx, GIMP_lpy, GIMP_lpz)
    wx0_avg = GIMP_f_gradf_avg[0]
    wy0_avg = GIMP_f_gradf_avg[1]
    wz0_avg = GIMP_f_gradf_avg[2]
    wx1_avg = GIMP_f_gradf_avg[3]
    wy1_avg = GIMP_f_gradf_avg[4]
    wz1_avg = GIMP_f_gradf_avg[5]
    wx2_avg = GIMP_f_gradf_avg[6]
    wy2_avg = GIMP_f_gradf_avg[7]
    wz2_avg = GIMP_f_gradf_avg[8]
    grad_wx0_avg = GIMP_f_gradf_avg[9]
    grad_wy0_avg = GIMP_f_gradf_avg[10]
    grad_wz0_avg = GIMP_f_gradf_avg[11]
    grad_wx1_avg = GIMP_f_gradf_avg[12]
    grad_wy1_avg = GIMP_f_gradf_avg[13]
    grad_wz1_avg = GIMP_f_gradf_avg[14]
    grad_wx2_avg = GIMP_f_gradf_avg[15]
    grad_wy2_avg = GIMP_f_gradf_avg[16]
    grad_wz2_avg = GIMP_f_gradf_avg[17]

    w_avg = wp.matrix(
        wx0_avg, wy0_avg, wz0_avg,
        wx1_avg, wy1_avg, wz1_avg,
        wx2_avg, wy2_avg, wz2_avg,
        shape=(3,3)
    )

    

    delta_u_GRAD = wp.mat33d()
    delta_u = wp.vec3d()
    div_delta_u = wp.float64(0.)

    # Pressure term
    particle_pressure = wp.float64(0.)
    particle_old_pressure = wp.float64(0.)
    particle_pressure_avg = wp.float64(0.)
    particle_old_pressure_avg = wp.float64(0.)
    particle_grad_new_p = wp.vec3d()

    # Loop on dofs
    # Note the indices are not flattened
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):

                weight = w[i][0] * w[j][1] * w[k][2]
                weight_grad = wp.vec3d(grad_w[i][0]*w[j][1]*w[k][2], w[i][0]*grad_w[j][1]*w[k][2], w[i][0]*w[j][1]*grad_w[k][2])
                weight_avg = w_avg[i][0] * w_avg[j][1] * w_avg[k][2]

                ix = bottom_corner_base_int[0] + i
                iy = bottom_corner_base_int[1] + j
                iz = bottom_corner_base_int[2] + k

                if weight<wp.float64(1e-7):
                    weight = wp.float64(0.0)
                    weight_grad = wp.vec3d()

                index_ij_x = ix + iy*(n_grid_x+1) + iz*((n_grid_x+1)*(n_grid_y+1))
                index_ij_y = index_ij_x + n_nodes
                index_ij_z = index_ij_x + 2*n_nodes
                index_ij_p = index_ij_x + 3*n_nodes

                node_solution = wp.vec3d(solution[index_ij_x], solution[index_ij_y], solution[index_ij_z])
                node_old_solution = wp.vec3d(old_solution[index_ij_x], old_solution[index_ij_y], old_solution[index_ij_z])
                node_p_solution = solution[index_ij_p]
                node_p_old_solution = old_solution[index_ij_p]

                delta_u_GRAD += wp.outer(weight_grad, node_solution-node_old_solution)
                delta_u += weight * (node_solution - node_old_solution)
                div_delta_u += wp.dot(weight_grad, node_solution-node_old_solution)

                particle_pressure += weight * node_p_solution
                particle_old_pressure += weight * node_p_old_solution
                particle_pressure_avg += weight_avg * node_p_solution
                particle_old_pressure_avg += weight_avg * node_p_old_solution
                particle_grad_new_p += weight_grad * node_p_solution


    old_F = deformation_gradient[p]
    particle_old_J = wp.determinant(old_F)

    incr_F = wp.identity(n=3, dtype=wp.float64) + delta_u_GRAD
    incr_F_inv = wp.inverse(incr_F)
    new_F = incr_F @ old_F

    # Neo-Hookean: From deformation gradient to stress
    new_F_inv = wp.inverse(new_F)
    particle_J = wp.determinant(new_F)
    particle_PK1_stress = lame_mu * (new_F - wp.transpose(new_F_inv)) + lame_lambda * wp.log(particle_J) * wp.transpose(new_F_inv)
    particle_Cauchy_stress = float64_one/particle_J * particle_PK1_stress @ wp.transpose(new_F)



    new_p_vol = p_vol * particle_J
    new_p_rho = p_rho / particle_J
    phi_s_initial = wp.float64(1.) - wp.float64(0.5) # magic number 0.5 indicates the initial porosity
    new_p_porosity = wp.float64(1.0) - phi_s_initial/particle_J
    new_p_mixture_rho = new_p_porosity*wp.float64(1.0) + (wp.float64(1.0)-new_p_porosity)*p_rho 

    # Get mobility
    mobility = wp.float64(1e-8) # assume constant

    # PPP stabilization parameter
    tau = wp.float64(0.5)/lame_mu


    # Penalty term
    penalty_parameter = youngs_modulus_diff[0] * wp.float64(1000.0) * (dy/PPD) * (dy/PPD)
    penalty_residual_x = wp.float64(0.0)
    penalty_residual_y = wp.float64(0.0)
    penalty_residual_z = wp.float64(0.0)


    original_height = end_z-dz/PPD
    current_height = original_height + (step+wp.float64(1.0)) * wp.float64(-0.02)
    if (x_particles[p][2] >= current_height) and (x_particles[p][0]<dx+wp.float64(1.)) and (x_particles[p][1]<dy+wp.float64(1.)): #(wp.norm_l2(corner_point-x_particles[p]) <= wp.float64(1.0)):
        new_p = x_particles[p] + delta_u
        penetration = current_height - new_p[2]
        penalty_residual_z = -penetration * penalty_parameter

        wp.atomic_add(indentation_force_array, int(step), -penalty_residual_z)
        wp.atomic_add(indentation_force, 0, -penalty_residual_z)



    # Momentum balance & Mass balance
    # Note the indices are not flattened
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):

                weight = w[i][0] * w[j][1] * w[k][2]
                weight_GRAD = wp.vec3d(grad_w[i][0]*w[j][1]*w[k][2], w[i][0]*grad_w[j][1]*w[k][2], w[i][0]*w[j][1]*grad_w[k][2])
                weight_grad = incr_F_inv @ weight_GRAD # NOTE here is incr_F_inv

                weight_avg = w_avg[i][0] * w_avg[j][1] * w_avg[k][2]

                ix = bottom_corner_base_int[0] + i
                iy = bottom_corner_base_int[1] + j
                iz = bottom_corner_base_int[2] + k

                index_ij_x = ix + iy*(n_grid_x+1) + iz*((n_grid_x+1)*(n_grid_y+1))
                index_ij_y = index_ij_x + n_nodes
                index_ij_z = index_ij_x + 2*n_nodes
                index_ij_p = index_ij_x + 3*n_nodes

                # Momentum balance
                rhs_value = (-weight_grad @ (particle_Cauchy_stress - particle_pressure*wp.identity(n=3, dtype=wp.float64)) + weight * new_p_mixture_rho * standard_gravity) * new_p_vol # Updated Lagrangian


                if (dofStruct.boundary_flag_array[index_ij_x]==False and dofStruct.activate_flag_array[index_ij_x]==True):
                    wp.atomic_add(rhs, index_ij_x, rhs_value[0])

                if (dofStruct.boundary_flag_array[index_ij_y]==False and dofStruct.activate_flag_array[index_ij_y]==True):
                    wp.atomic_add(rhs, index_ij_y, rhs_value[1])

                if (dofStruct.boundary_flag_array[index_ij_z]==False and dofStruct.activate_flag_array[index_ij_z]==True):
                    wp.atomic_add(rhs, index_ij_z, rhs_value[2])
                    wp.atomic_add(rhs, index_ij_z, penalty_residual_z * weight)


                # Mass balance
                rhs_mass_value = (weight * (wp.log(particle_J)-wp.log(particle_old_J))
                              + wp.dot(weight_grad, (mobility * particle_grad_new_p)) * dt
                              ) * new_p_vol

                if (dofStruct.boundary_flag_array[index_ij_p]==False and dofStruct.activate_flag_array[index_ij_p]==True):
                    wp.atomic_add(rhs, index_ij_p, rhs_mass_value)

                # PPP stabilization
                rhs_ppp = (tau * (weight-weight_avg) * (particle_pressure - particle_pressure_avg - (particle_old_pressure - particle_old_pressure_avg))) * new_p_vol
                if (dofStruct.boundary_flag_array[index_ij_p]==False and dofStruct.activate_flag_array[index_ij_p]==True):
                    wp.atomic_add(rhs, index_ij_p, rhs_ppp)                
               

@wp.kernel
def from_jacobian_to_vector_parallel(jacobian_wp: wp.array(dtype=wp.float64),
                                     rows: wp.array(dtype=wp.int32),
                                     cols: wp.array(dtype=wp.int32),
                                     vals: wp.array(dtype=wp.float64),
                                     n_grid_x: wp.int32,
                                     n_grid_y: wp.int32,
                                     n_grid_z: wp.int32,
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
        node_idz = wp.int(0)

        if row_index<n_nodes: # x-dof
            node_idx = wp.mod(row_index, n_grid_x+1)
            node_idy = wp.int(wp.mod(row_index, ((n_grid_x+1)*(n_grid_y+1))) / (n_grid_x+1)) 
            node_idz = wp.int(row_index / ((n_grid_x+1)*(n_grid_y+1)))
        elif row_index>=n_nodes and row_index<2*n_nodes: # y-dof
            node_idx = wp.mod(row_index-n_nodes, n_grid_x+1)
            node_idy = wp.int(wp.mod(row_index-n_nodes, ((n_grid_x+1)*(n_grid_y+1))) / (n_grid_x+1)) 
            node_idz = wp.int((row_index-n_nodes) / ((n_grid_x+1)*(n_grid_y+1)))
        elif row_index>=2*n_nodes and row_index<3*n_nodes: # z-dof
            node_idx = wp.mod(row_index-2*n_nodes, n_grid_x+1)
            node_idy = wp.int(wp.mod(row_index-2*n_nodes, ((n_grid_x+1)*(n_grid_y+1))) / (n_grid_x+1)) 
            node_idz = wp.int((row_index-2*n_nodes) / ((n_grid_x+1)*(n_grid_y+1)))
        else: # pressure dof
            node_idx = wp.mod(row_index-3*n_nodes, n_grid_x+1)
            node_idy = wp.int(wp.mod(row_index-3*n_nodes, ((n_grid_x+1)*(n_grid_y+1))) / (n_grid_x+1)) 
            node_idz = wp.int((row_index-3*n_nodes) / ((n_grid_x+1)*(n_grid_y+1)))


        # Flattened loop
        for flattened_id in range(125):
            i = wp.int(flattened_id/25)
            j = wp.mod(wp.int(flattened_id/5), 5)
            k = wp.mod(flattened_id, 5)

            adj_node_idx = node_idx + (i-2)
            adj_node_idy = node_idy + (j-2)
            adj_node_idz = node_idz + (k-2)
        
            adj_index_x = adj_node_idx + adj_node_idy*(n_grid_x+1) + adj_node_idz*((n_grid_x+1)*(n_grid_y+1))
            adj_index_y = adj_index_x + n_nodes
            adj_index_z = adj_index_x + 2*n_nodes
            adj_index_p = adj_index_x + 3*n_nodes

            if adj_node_idx>=0 and adj_node_idx<=n_grid_x and adj_node_idy>=0 and adj_node_idy<=n_grid_y and adj_node_idz>=0 and adj_node_idz<=n_grid_z: # adj_node is reasonable
                if dofStruct.boundary_flag_array[row_index]==False and dofStruct.activate_flag_array[row_index]==True:
                    if dofStruct.boundary_flag_array[adj_index_x]==False and dofStruct.activate_flag_array[adj_index_x]==True and wp.isnan(jacobian_wp[adj_index_x])==False:
                        rows[row_index*125 + (i+j*5+k*25)] = row_index
                        cols[row_index*125 + (i+j*5+k*25)] = adj_index_x
                        vals[row_index*125 + (i+j*5+k*25)] = -jacobian_wp[adj_index_x]

                    if dofStruct.boundary_flag_array[adj_index_y]==False and dofStruct.activate_flag_array[adj_index_y]==True and wp.isnan(jacobian_wp[adj_index_y])==False:
                        rows[125*n_matrix_size + row_index*125 + (i+j*5+k*25)] = row_index
                        cols[125*n_matrix_size + row_index*125 + (i+j*5+k*25)] = adj_index_y
                        vals[125*n_matrix_size + row_index*125 + (i+j*5+k*25)] = -jacobian_wp[adj_index_y]

                    if dofStruct.boundary_flag_array[adj_index_z]==False and dofStruct.activate_flag_array[adj_index_z]==True and wp.isnan(jacobian_wp[adj_index_z])==False:
                        rows[125*2*n_matrix_size + row_index*125 + (i+j*5+k*25)] = row_index
                        cols[125*2*n_matrix_size + row_index*125 + (i+j*5+k*25)] = adj_index_z
                        vals[125*2*n_matrix_size + row_index*125 + (i+j*5+k*25)] = -jacobian_wp[adj_index_z]

                    if dofStruct.boundary_flag_array[adj_index_p]==False and dofStruct.activate_flag_array[adj_index_p]==True and wp.isnan(jacobian_wp[adj_index_p])==False:
                        rows[125*3*n_matrix_size + row_index*125 + (i+j*5+k*25)] = row_index
                        cols[125*3*n_matrix_size + row_index*125 + (i+j*5+k*25)] = adj_index_p
                        vals[125*3*n_matrix_size + row_index*125 + (i+j*5+k*25)] = -jacobian_wp[adj_index_p]




@wp.kernel
def set_diagnal_component_for_boundary_and_deactivated_dofs(dofStruct: DofStruct,
                                                            rows: wp.array(dtype=wp.int32),
                                                            cols: wp.array(dtype=wp.int32),
                                                            vals: wp.array(dtype=wp.float64),
                                                            n_matrix_size: wp.int32):
    dof_id = wp.tid()

    if dofStruct.boundary_flag_array[dof_id]==True or dofStruct.activate_flag_array[dof_id]==False:
        rows[125*4*n_matrix_size + dof_id] = dof_id
        cols[125*4*n_matrix_size + dof_id] = dof_id
        vals[125*4*n_matrix_size + dof_id] = wp.float64(1.)


# From increment to solution
@wp.kernel
def from_increment_to_solution(increment: wp.array(dtype=wp.float64),
                               solution: wp.array(dtype=wp.float64)):
    i = wp.tid()

    solution[i] += increment[i]



# G2P
@wp.kernel
def G2P(x_particles: wp.array(dtype=wp.vec3d),
        deformation_gradient: wp.array(dtype=wp.mat33d),
        particle_Cauchy_stress_array: wp.array(dtype=wp.mat33d),
        inv_dx: wp.float64,
        inv_dy: wp.float64,
        inv_dz: wp.float64,
        dx: wp.float64,
        dy: wp.float64,
        dz: wp.float64,
        youngs_modulus_diff: wp.array(dtype=wp.float64),
        poisson_ratio: wp.float64,
        n_grid_x: wp.int32,
        n_grid_y: wp.int32,
        n_nodes: wp.int32,
        new_solution: wp.array(dtype=wp.float64),
        old_solution: wp.array(dtype=wp.float64),
        dofStruct: DofStruct,
        particle_shear_stress_array: wp.array(dtype=wp.float64),
        particle_mean_normal_stress_array: wp.array(dtype=wp.float64),
        particle_pressure_array: wp.array(dtype=wp.float64),
        GIMP_lp: wp.array(dtype=wp.vec3d)
        ):

    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    lame_lambda = youngs_modulus_diff[0]*poisson_ratio / ((wp.float64(1.0)+poisson_ratio) * (wp.float64(1.0)-wp.float64(2.0)*poisson_ratio))
    lame_mu = youngs_modulus_diff[0] / (wp.float64(2.0)*(wp.float64(1.0)+poisson_ratio))


    GIMP_lpx = GIMP_lp[p][0]
    GIMP_lpy = GIMP_lp[p][1]
    GIMP_lpz = GIMP_lp[p][2]

    # GIMP
    xp = x_particles[p]
    bottom_corner = xp - wp.vec3d(GIMP_lpx, GIMP_lpy, GIMP_lpz)
    bottom_corner_base_x = bottom_corner[0]*inv_dx + wp.float64(1e-8)
    bottom_corner_base_y = bottom_corner[1]*inv_dy + wp.float64(1e-8)
    bottom_corner_base_z = bottom_corner[2]*inv_dz + wp.float64(1e-8)
    bottom_corner_base_int = wp.vec3i(wp.int(bottom_corner_base_x), wp.int(bottom_corner_base_y), wp.int(bottom_corner_base_z))
    bottom_corner_base = wp.vec3d(wp.float64(bottom_corner_base_int[0]), wp.float64(bottom_corner_base_int[1]), wp.float64(bottom_corner_base_int[2]))

    up_corner = xp + wp.vec3d(GIMP_lpx, GIMP_lpy, GIMP_lpz)
    up_corner_base_x = up_corner[0]*inv_dx + wp.float64(1e-8)
    up_corner_base_y = up_corner[1]*inv_dy + wp.float64(1e-8)
    up_corner_base_z = up_corner[2]*inv_dz + wp.float64(1e-8)
    up_corner_base_int = wp.vec3i(wp.int(up_corner_base_x), wp.int(up_corner_base_y), wp.int(up_corner_base_z))
    up_corner_base = wp.vec3d(wp.float64(up_corner_base_int[0]), wp.float64(up_corner_base_int[1]), wp.float64(up_corner_base_int[2]))


    GIMP_f_gradf = get_GIMP_shape_function_and_gradient(xp, dx, dy, dz, inv_dx, inv_dy, inv_dz, dofStruct, n_grid_x, n_grid_y, n_nodes, GIMP_lpx, GIMP_lpy, GIMP_lpz)
    wx0 = GIMP_f_gradf[0]
    wy0 = GIMP_f_gradf[1]
    wz0 = GIMP_f_gradf[2]
    wx1 = GIMP_f_gradf[3]
    wy1 = GIMP_f_gradf[4]
    wz1 = GIMP_f_gradf[5]
    wx2 = GIMP_f_gradf[6]
    wy2 = GIMP_f_gradf[7]
    wz2 = GIMP_f_gradf[8]
    grad_wx0 = GIMP_f_gradf[9]
    grad_wy0 = GIMP_f_gradf[10]
    grad_wz0 = GIMP_f_gradf[11]
    grad_wx1 = GIMP_f_gradf[12]
    grad_wy1 = GIMP_f_gradf[13]
    grad_wz1 = GIMP_f_gradf[14]
    grad_wx2 = GIMP_f_gradf[15]
    grad_wy2 = GIMP_f_gradf[16]
    grad_wz2 = GIMP_f_gradf[17]

    w = wp.matrix(
        wx0, wy0, wz0,
        wx1, wy1, wz1,
        wx2, wy2, wz2,
        shape=(3,3)
        )

    grad_w = wp.matrix(
        grad_wx0, grad_wy0, grad_wz0,
        grad_wx1, grad_wy1, grad_wz1,
        grad_wx2, grad_wy2, grad_wz2,
        shape=(3,3)
        )



    # Loop on dofs
    delta_u = wp.vec3d()
    delta_u_GRAD = wp.mat33d()
    particle_new_p = wp.float64(0.)
    #Flattened loop
    for flattened_id in range(27):
        i = wp.int(flattened_id/9)
        j = wp.mod(wp.int(flattened_id/3), 3)
        k = wp.mod(flattened_id, 3)

        weight = w[i][0] * w[j][1] * w[k][2]
        weight_grad = wp.vec3d(grad_w[i][0]*w[j][1]*w[k][2], w[i][0]*grad_w[j][1]*w[k][2], w[i][0]*w[j][1]*grad_w[k][2])
        ix = bottom_corner_base_int[0] + i
        iy = bottom_corner_base_int[1] + j
        iz = bottom_corner_base_int[2] + k


        index_ij_x = ix + iy*(n_grid_x+1) + iz*((n_grid_x+1)*(n_grid_y+1))
        index_ij_y = index_ij_x + n_nodes
        index_ij_z = index_ij_x + 2*n_nodes
        index_ij_p = index_ij_x + 3*n_nodes

        node_new_solution = wp.vec3d(new_solution[index_ij_x], new_solution[index_ij_y], new_solution[index_ij_z])
        node_old_solution = wp.vec3d(old_solution[index_ij_x], old_solution[index_ij_y], old_solution[index_ij_z])

        delta_u += weight * (node_new_solution - node_old_solution)
        delta_u_GRAD += wp.outer(weight_grad, node_new_solution-node_old_solution)

        particle_new_p += weight * new_solution[index_ij_p]

    old_F_3d = deformation_gradient[p]
    old_F = old_F_3d

    incr_F = wp.identity(n=3, dtype=wp.float64) + delta_u_GRAD
    new_F = incr_F @ old_F


    # Save results
    deformation_gradient[p] = new_F

    x_particles[p] += delta_u

    new_F_3d_inv = wp.inverse(deformation_gradient[p])
    particle_J = wp.determinant(deformation_gradient[p])
    particle_PK1_stress_3d = lame_mu * (deformation_gradient[p] - wp.transpose(new_F_3d_inv)) + lame_lambda * wp.log(particle_J) * wp.transpose(new_F_3d_inv)
    particle_Cauchy_stress_3d = float64_one/particle_J * particle_PK1_stress_3d @ wp.transpose(deformation_gradient[p])
    particle_Cauchy_stress_array[p] = particle_Cauchy_stress_3d

    # get shear stress
    Q = wp.mat33d()
    d = wp.vec3d()
    wp.eig3(particle_Cauchy_stress_3d, Q, d)
    stress0 = wp.min(wp.min(d[0], d[1]), d[2])
    stress2 = wp.max(wp.max(d[0], d[1]), d[2])
    particle_shear_stress_array[p] = stress2 - stress0

    particle_mean_normal_stress_array[p] = wp.float64(1.)/wp.float64(3.) * (particle_Cauchy_stress_3d[0,0] + particle_Cauchy_stress_3d[1,1] + particle_Cauchy_stress_3d[2,2])
    particle_pressure_array[p] = particle_new_p



@wp.kernel
def update_GIMP_lp(deformation_gradient: wp.array(dtype=wp.mat33d),
                   GIMP_lp: wp.array(dtype=wp.vec3d),
                   GIMP_lp_initial: wp.float64,
                   x_particles: wp.array(dtype=wp.vec3d),
                   start_x: wp.float64,
                   start_y: wp.float64,
                   start_z: wp.float64,
                   end_x: wp.float64,
                   end_y: wp.float64
                   ):
    p = wp.tid()

    # Refer to Eq. (38), (39) of Charlton et al. 2017
    this_particle_F = deformation_gradient[p]
    U00 = wp.sqrt(this_particle_F[0,0]*this_particle_F[0,0] + this_particle_F[1,0]*this_particle_F[1,0] + this_particle_F[2,0]*this_particle_F[2,0])
    U11 = wp.sqrt(this_particle_F[0,1]*this_particle_F[0,1] + this_particle_F[1,1]*this_particle_F[1,1] + this_particle_F[2,1]*this_particle_F[2,1])
    U22 = wp.sqrt(this_particle_F[0,2]*this_particle_F[0,2] + this_particle_F[1,2]*this_particle_F[1,2] + this_particle_F[2,2]*this_particle_F[2,2])

    lpx_updated = GIMP_lp_initial*U00
    lpy_updated = GIMP_lp_initial*U11
    lpz_updated = GIMP_lp_initial*U22

    if x_particles[p][0]-lpx_updated < start_x:
        lpx_updated = x_particles[p][0] - start_x - wp.float64(1e-6)
    if x_particles[p][1]-lpy_updated < start_y:
        lpy_updated = x_particles[p][1] - start_y - wp.float64(1e-6)
    if x_particles[p][2]-lpz_updated < start_z:
        lpz_updated = x_particles[p][2] - start_z - wp.float64(1e-6)

    if x_particles[p][0]+lpx_updated > end_x:
        lpx_updated = end_x - x_particles[p][0] - wp.float64(1e-6)
    if x_particles[p][1]+lpy_updated > end_y:
        lpy_updated = end_y - x_particles[p][1] - wp.float64(1e-6)

    GIMP_lp[p] = wp.vec3d(lpx_updated, lpy_updated, lpz_updated)




@wp.kernel
def get_loss(indentation_force_array: wp.array(dtype=wp.float64),
             indentation_force: wp.array(dtype=wp.float64),
             loss: wp.array(dtype=wp.float64)):

    loss[0] = (indentation_force[0]/(wp.float64(0.02)*wp.float64(25.)) - wp.float64(13379.54276765))**wp.float64(2.) # The magic number 13379.54276765 is from E = 1e4 kPa



# Post-processing parameters
n_iter = 15


def init_particle_position(start_x, start_y, start_z, end_x, end_y, end_z, dx, n_grid_x, n_grid_y, n_grid_z, PPD, n_particles):
    particle_id = 0
    particle_pos_np = np.zeros((n_particles, 3))

    for i in range(n_grid_x):
        for j in range(n_grid_y):
            for k in range(n_grid_z):
                cell_center_pos = np.array([(i+0.5)*dx, (j+0.5)*dx, (k+0.5)*dx])

                if start_x < cell_center_pos[0] and cell_center_pos[0] < end_x:
                    if start_y < cell_center_pos[1] and cell_center_pos[1] < end_y:
                        if start_z < cell_center_pos[2] and cell_center_pos[2] < end_z:
                            for p_x in range(PPD):
                                for p_y in range(PPD):
                                    for p_z in range(PPD):
                                        particle_pos_np[particle_id] = np.array([i*dx, j*dx, k*dx]) + np.array([(0.5+p_x)*dx/PPD, (0.5+p_y)*dx/PPD, (0.5+p_z)*dx/PPD])

                                        particle_id += 1


    print(particle_id)
    return particle_pos_np


start_x = dx
start_y = dx
start_z = 0.0 #dx
end_x = start_x + 5.0
end_y = start_y + 5.0
end_z = start_z + 5.0


particle_pos_np = init_particle_position(start_x, start_y, start_z, end_x, end_y, end_z, dx, n_grid_x, n_grid_y, n_grid_z, PPD, n_particles)



deformation_gradient = wp.array(shape=n_particles, dtype=wp.mat33d)
particle_Cauchy_stress_array = wp.zeros(shape=n_particles, dtype=wp.mat33d)
particle_shear_stress_array = wp.zeros(shape=n_particles, dtype=wp.float64)
particle_mean_normal_stress_array = wp.zeros(shape=n_particles, dtype=wp.float64)
particle_pressure_array = wp.zeros(shape=n_particles, dtype=wp.float64)

GIMP_lp = wp.array(shape=n_particles, dtype=wp.vec3d)

particle_boundary_flag_array = wp.zeros(shape=n_particles, dtype=wp.bool)




# Matrix
n_nodes = (n_grid_x+1) * (n_grid_y+1) * (n_grid_z+1)
n_matrix_size = 4 * n_nodes # 4 = 3+1, where 3 indicates the spatial dimension and 1 indicates the pressure dof
bsr_matrix = wps.bsr_zeros(n_matrix_size, n_matrix_size, block_type=wp.float64)
bsr_matrix_P2G = wps.bsr_zeros(n_nodes, n_nodes, block_type=wp.float64)
rhs = wp.zeros(shape=n_matrix_size, dtype=wp.float64, requires_grad=True)
increment = wp.zeros(shape=n_matrix_size, dtype=wp.float64)
P2G_solution = wp.zeros(shape=n_nodes, dtype=wp.float64)
old_solution = wp.zeros(shape=n_matrix_size, dtype=wp.float64)
new_solution = wp.zeros(shape=n_matrix_size, dtype=wp.float64, requires_grad=True)


rows = wp.zeros(shape=4*125*n_matrix_size+n_matrix_size, dtype=wp.int32) # 2 indicates the spatial dimension
cols = wp.zeros(shape=4*125*n_matrix_size+n_matrix_size, dtype=wp.int32)
vals = wp.zeros(shape=4*125*n_matrix_size+n_matrix_size, dtype=wp.float64)

rhs_P2G = wp.zeros(shape=n_nodes, dtype=wp.float64)
rows_P2G = wp.zeros(shape=27*27*n_particles, dtype=wp.int32)
cols_P2G = wp.zeros(shape=27*27*n_particles, dtype=wp.int32)
vals_P2G = wp.zeros(shape=27*27*n_particles, dtype=wp.float64)



dofStruct = DofStruct()
dofStruct.activate_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)
dofStruct.boundary_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)



# Inverse
youngs_modulus_diff = wp.zeros(shape=1, dtype=wp.float64, requires_grad=True)
indentation_force_array = wp.zeros(shape=25, dtype=wp.float64, requires_grad=True)
indentation_force = wp.zeros(shape=1, dtype=wp.float64, requires_grad=True)
loss = wp.zeros(shape=1, dtype=wp.float64, requires_grad=True)



# Get the max length of the selector array
max_selector_length = 0
c_iter = 0 # column
r_iter = 0 # row
l_iter = 0 # layer
current_node_id = c_iter + r_iter*(n_grid_x+1) + l_iter*((n_grid_x+1)*(n_grid_y+1))

def pick_grid_nodes(n_grid_x, n_grid_y, n_grid_z, first_grid_node):
    # Determine the row, column, and layer of the first grid node
    layer_start = first_grid_node // ((n_grid_x + 1) * (n_grid_y + 1))
    row_start = (first_grid_node % ((n_grid_x + 1) * (n_grid_y + 1))) // (n_grid_x + 1)
    col_start = first_grid_node % (n_grid_x + 1)

    # Step size for the 3x3x3 grid
    step = 5

    # Create arrays for layers, rows, and columns to pick nodes efficiently
    layers = np.arange(layer_start, n_grid_z + 1, step)
    rows = np.arange(row_start, n_grid_y + 1, step)
    cols = np.arange(col_start, n_grid_x + 1, step)

    # Generate a grid of layer, row, and column indices
    layer_indices, row_indices, col_indices = np.meshgrid(layers, rows, cols, indexing='ij')

    # Compute the node indices
    picked_nodes = (layer_indices * (n_grid_x + 1) * (n_grid_y + 1) +
                    row_indices * (n_grid_x + 1) +
                    col_indices)

    # Flatten the array and filter out indices exceeding the total number of nodes
    # total_nodes = (n_grid_x + 1) * (n_grid_y + 1) * (n_grid_z + 1)
    picked_nodes = picked_nodes.flatten()
    # picked_nodes = picked_nodes[picked_nodes < total_nodes]

    return picked_nodes

tic = time.perf_counter()
selector = pick_grid_nodes(n_grid_x, n_grid_y, n_grid_z, current_node_id)
max_selector_length = len(selector)
toc = time.perf_counter()
print('Time for pick_grid_nodes:', toc-tic) 



e_list_x = []
e_list_y = []
e_list_z = []
e_list_p = []
for c_iter in range(5):
    for r_iter in range(5):
        for l_iter in range(5):
            current_node_id = c_iter + r_iter * (n_grid_x+1) + l_iter*((n_grid_x+1)*(n_grid_y+1))
            select_index = np.zeros(n_matrix_size)

            # x
            selector = pick_grid_nodes(n_grid_x, n_grid_y, n_grid_z, current_node_id)

            select_index[selector] = 1.0
            e = wp.array(select_index, dtype=wp.float64)
            e_list_x.append(e)

            # y
            selector_y = selector + n_nodes

            select_index = np.zeros(n_matrix_size)
            select_index[selector_y] = 1.0
            e = wp.array(select_index, dtype=wp.float64)
            e_list_y.append(e)

            # z
            selector_z = selector + 2*n_nodes

            select_index = np.zeros(n_matrix_size)
            select_index[selector_z] = 1.0
            e = wp.array(select_index, dtype=wp.float64)
            e_list_z.append(e)

            # p
            selector_p = selector + 3*n_nodes

            select_index = np.zeros(n_matrix_size)
            select_index[selector_p] = 1.0
            e = wp.array(select_index, dtype=wp.float64)
            e_list_p.append(e)




for training_iter in range(10):
    indentation_force_list = np.empty(0)
    E_grad = None

    dx = max_x/n_grid_x
    dy = max_x/n_grid_x
    dz = max_x/n_grid_x
    inv_dx = float(n_grid_x/max_x)
    inv_dy = float(n_grid_x/max_x)
    inv_dz = float(n_grid_x/max_x)


    # Init particles
    x_particles = wp.from_numpy(particle_pos_np, dtype=wp.vec3d)

    print('ready initialization')
    # Initialization
    wp.launch(kernel=initialization,
              dim=n_particles,
              inputs=[deformation_gradient, x_particles, particle_boundary_flag_array, end_y, end_z, dx, PPD, start_x, start_z])

    wp.launch(kernel=initialize_GIMP_lp,
              dim=n_particles,
              inputs=[GIMP_lp, GIMP_lp_initial])

    wp.launch(kernel=initialize_youngs_modulus_diff,
              dim=1,
              inputs=[youngs_modulus_diff, youngs_modulus])

    particle_pressure_array.zero_()

    selector_wp = wp.zeros(shape=max_selector_length, dtype=wp.int32)

    # e = wp.zeros(shape=n_matrix_size, dtype=wp.float64)


    # Post-processing
    x_numpy = np.array(x_particles.numpy())
    output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': particle_Cauchy_stress_array.numpy()[:,1,1]})
    output_particles.write("3d_compression_coupled_particles_%d.vtk" % 0)



    # while True:
    for step in range(25):

        old_solution.zero_()
        new_solution.zero_()
        increment.zero_()


        dofStruct.activate_flag_array.zero_()
        dofStruct.boundary_flag_array.zero_()

        print('Step ', step)

        # P2G
        rhs_P2G.zero_()
        rows_P2G.zero_()
        cols_P2G.zero_()
        vals_P2G.zero_()
        P2G_solution.zero_()
        wps.bsr_set_zero(bsr_matrix_P2G)
        wp.launch(kernel=set_boundary_dofs,
                  dim=(n_grid_x+1, n_grid_y+1, n_grid_z+1),
                  inputs=[dofStruct, n_grid_x, n_grid_y, n_nodes, dx, dy, end_x, end_y])


        wp.launch(kernel=P2G,
                  dim=n_particles,
                  inputs=[x_particles, inv_dx, inv_dy, inv_dz, dx, dy, dz, n_grid_x, n_grid_y, n_nodes, dofStruct, GIMP_lp, particle_boundary_flag_array, rhs_P2G, rows_P2G, cols_P2G, vals_P2G, p_mass, particle_pressure_array])


        # Solve for consistent mass matrix 
        wps.bsr_set_from_triplets(bsr_matrix_P2G, rows_P2G, cols_P2G, vals_P2G)
        preconditioner = wp.optim.linear.preconditioner(bsr_matrix_P2G, ptype='diag')
        solver_state = wp.optim.linear.bicgstab(A=bsr_matrix_P2G, b=rhs_P2G, x=P2G_solution, tol=1e-8, M=preconditioner)

        print(solver_state)

        wp.launch(kernel=P2G_solution_to_solution,
                  dim=n_nodes,
                  inputs=[dofStruct, n_nodes, P2G_solution, old_solution, new_solution])


        # Newton iteration
        # tic = time.perf_counter()
        for iter_id in range(n_iter):

            rhs.zero_()
            rows.zero_()
            cols.zero_()
            vals.zero_()
            indentation_force_array.zero_()
            indentation_force.zero_()

            loss.zero_()

            tape = wp.Tape()
            with tape:
                # tic = time.perf_counter()
                # assemble residual
                wp.launch(kernel=assemble_residual,
                          dim=n_particles,
                          inputs=[x_particles, inv_dx, inv_dy, inv_dz, dx, dy, dz, rhs, new_solution, old_solution, p_vol, p_rho, youngs_modulus_diff, poisson_ratio, deformation_gradient, particle_boundary_flag_array, n_grid_x, n_grid_y, n_nodes, dofStruct, step, PPD, end_y, end_z, GIMP_lp, dt, indentation_force_array, indentation_force, n_particles])
                # toc = time.perf_counter()
                # print('Time for assemble_residual:', toc-tic) 

                wp.launch(kernel=get_loss,
                          dim=1,
                          inputs=[indentation_force_array, indentation_force, loss])


            tic = time.perf_counter()
            # Auto-diff for sparse matrix
            # x dof
            pattern_id = 0
            for c_iter in range(5):
                for r_iter in range(5):
                    for l_iter in range(5):
                        current_node_id = c_iter + r_iter * (n_grid_x+1) + l_iter*((n_grid_x+1)*(n_grid_y+1))
                        select_index = np.zeros(n_matrix_size)

                        # tic = time.perf_counter()
                        selector = pick_grid_nodes(n_grid_x, n_grid_y, n_grid_z, current_node_id)
                        # toc = time.perf_counter()
                        # print('Time for pick_grid_nodes:', toc-tic) # 



                        # tic = time.perf_counter()
                        select_index[selector] = 1.0
                        # toc = time.perf_counter()
                        # print('Time for e = wp.array:', toc-tic) # 0.01246162600000389 * 3 * 125 = 4.67 s, very slow!

                        # tic = time.perf_counter()
                        tape.backward(grads={rhs: e_list_x[pattern_id]})
                        q_grad_i = tape.gradients[new_solution]
                        # toc = time.perf_counter()
                        # print('Time for auto-diff:', toc-tic) # 

                        # tic = time.perf_counter()

                        
                        # fill the selector to the max length
                        selector_resize = selector + 0
                        selector_resize.resize(max_selector_length)
                        selector_wp = wp.from_numpy(selector_resize, dtype=wp.int32)


                        wp.launch(kernel=from_jacobian_to_vector_parallel,
                                  dim=max_selector_length,
                                  inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_grid_z, n_nodes, n_matrix_size, selector_wp, dofStruct])

                        
                        # toc = time.perf_counter()
                        # print('Time for from_jacobian_to_vector_parallel:', toc-tic)

                        # tic = time.perf_counter()
                        tape.zero()
                        # toc = time.perf_counter()
                        # print('Time for tape.zero():', toc-tic)

                        # toc = time.perf_counter()
                        # print('Time for x_dof:', toc-tic)



                        # y dof
                        selector_y = selector + n_nodes

                        select_index = np.zeros(n_matrix_size)
                        select_index[selector_y] = 1.0

                        # tic = time.perf_counter()
                        tape.backward(grads={rhs: e_list_y[pattern_id]})
                        q_grad_i = tape.gradients[new_solution]
                        # toc = time.perf_counter()
                        # print('Time for auto-diff:', toc-tic)   

                        selector_resize = selector_y + 0
                        selector_resize.resize(max_selector_length)
                        selector_wp = wp.from_numpy(selector_resize, dtype=wp.int32)

                        wp.launch(kernel=from_jacobian_to_vector_parallel,
                                  dim=max_selector_length,
                                  inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_grid_z, n_nodes, n_matrix_size, selector_wp, dofStruct])

                        tape.zero()


                        # z dof
                        selector_z = selector + 2*n_nodes

                        select_index = np.zeros(n_matrix_size)
                        select_index[selector_z] = 1.0
                        
                        tape.backward(grads={rhs: e_list_z[pattern_id]})
                        q_grad_i = tape.gradients[new_solution]

                        selector_resize = selector_z + 0
                        selector_resize.resize(max_selector_length)
                        selector_wp = wp.from_numpy(selector_resize, dtype=wp.int32)

                        wp.launch(kernel=from_jacobian_to_vector_parallel,
                                  dim=max_selector_length,
                                  inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_grid_z, n_nodes, n_matrix_size, selector_wp, dofStruct])

                        tape.zero()



                        # p dof
                        selector_p = selector + 3*n_nodes

                        select_index = np.zeros(n_matrix_size)
                        select_index[selector_p] = 1.0

                        tape.backward(grads={rhs: e_list_p[pattern_id]})
                        q_grad_i = tape.gradients[new_solution]

                        selector_resize = selector_p + 0
                        selector_resize.resize(max_selector_length)
                        selector_wp = wp.from_numpy(selector_resize, dtype=wp.int32)

                        wp.launch(kernel=from_jacobian_to_vector_parallel,
                                  dim=max_selector_length,
                                  inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_grid_z, n_nodes, n_matrix_size, selector_wp, dofStruct]
                                  )
                        tape.zero()


                        pattern_id = pattern_id + 1

            if step == 24:
                tape.backward(loss)
                E_grad = youngs_modulus_diff.grad.numpy()
                # print('Loss:', loss.numpy(), 'Force:', indentation_force.numpy()[0], 'E_grad:', E_grad)

            
            tape.reset()


            # toc = time.perf_counter()
            # print('Time for auto-diff and from_jacobian_to_vector_parallel:', toc-tic) # 



            
            # pypardiso direct solver
            wp.launch(kernel=set_diagnal_component_for_boundary_and_deactivated_dofs,
                      dim=n_matrix_size,
                      inputs=[dofStruct, rows, cols, vals, n_matrix_size])

            bsr_matrix_pypardiso = sparse.coo_matrix((vals.numpy(), (rows.numpy(), cols.numpy())), shape=(n_matrix_size, n_matrix_size)).asformat('csr')
            b = rhs.numpy()
            x_pypardiso = pypardiso.spsolve(bsr_matrix_pypardiso, b)

            increment = wp.from_numpy(x_pypardiso, dtype=wp.float64)



            # From increment to solution
            wp.launch(kernel=from_increment_to_solution,
                      dim=n_matrix_size,
                      inputs=[increment, new_solution])






            with np.printoptions(threshold=np.inf):
                print('residual.norm:', np.linalg.norm(rhs.numpy()))


        # toc = time.perf_counter()
        # print('Time for newton iteration:', toc-tic)

            if np.linalg.norm(rhs.numpy())<1e-9:
                break


        # G2P
        wp.launch(kernel=G2P,
                  dim=n_particles,
                  inputs=[x_particles, deformation_gradient, particle_Cauchy_stress_array, inv_dx, inv_dy, inv_dz, dx, dy, dz, youngs_modulus_diff, poisson_ratio, n_grid_x, n_grid_y, n_nodes, new_solution, old_solution, dofStruct, particle_shear_stress_array, particle_mean_normal_stress_array, particle_pressure_array, GIMP_lp])


        wp.launch(kernel=update_GIMP_lp,
                  dim=n_particles,
                  inputs=[deformation_gradient, GIMP_lp, GIMP_lp_initial, x_particles, start_x, start_y, start_z, end_x, end_y])

        # Print indentation force
        indentation_force_list = np.append(indentation_force_list, indentation_force.numpy()[0])
        with np.printoptions(threshold=np.inf):
            for force in indentation_force_list:
                print(force)

        

        # Post-processing
        x_numpy = np.array(x_particles.numpy())
        
        output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'shear_stress': particle_shear_stress_array.numpy(), 'pressure': particle_pressure_array.numpy()})
        output_particles.write("3d_compression_coupled_particles_%d.vtk" % (step+1))


        # Compress the background grid
        dz -= 0.02/20
        inv_dz = 1.0/dz




    # Inverse
    print('Loss after finishing load steps:', loss.numpy(), 'Force:', indentation_force.numpy()[0], 'E_grad:', E_grad)

    # Gradient descent
    learning_rate = 0.2
    updated_E = youngs_modulus - learning_rate * E_grad[0]
    print('Updated E:', updated_E)
    youngs_modulus = updated_E

    







