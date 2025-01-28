import warp as wp
import numpy as np
import meshio

import warp.sparse as wps

import warp.optim.linear

import time


# Implicit MPM solver for 2D compaction under self weight using Warp
# Contact: Yidong Zhao (ydzhao@kaist.ac.kr; ydzhao94@gmail.com)

# Refer to an implicit FEM solver (by Xuan Li) for the usage of sparse matrix: https://github.com/xuan-li/warp_FEM/tree/main


wp.init()


n_particles = 36864 #9216 #1024 #256 
n_grid_x = 80 #40 #20
n_grid_y = 80 #40 #20
grid_size = (n_grid_x, n_grid_y)

max_x = 20.0
dx = max_x/n_grid_x
inv_dx = float(n_grid_x/max_x)

PPD = 6 #4 #2 # particle per direction

youngs_modulus = 100.0 # kPa
poisson_ratio = 0.3
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
def initialization(deformation_gradient: wp.array(dtype=wp.mat33d),
                   x_particles: wp.array(dtype=wp.vec2d),
                   v_particles: wp.array(dtype=wp.vec2d)):
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

    v_particles[p] = wp.vec2d(wp.float64(0.0), wp.float64(0.0))




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
                      n_nodes: wp.int32):
    
    node_idx, node_idy = wp.tid()
    dof_x = node_idx + node_idy*(n_grid_x + 1)
    dof_y = dof_x + n_nodes

    if node_idx<=1:
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
        dofStruct: DofStruct):
    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    # Calculate shape functions
    base_x = x_particles[p][0] * inv_dx - wp.float64(0.5)
    base_y = x_particles[p][1] * inv_dx - wp.float64(0.5)
    base_int = wp.vector(wp.int(base_x), wp.int(base_y))
    base = wp.vector(wp.float64(base_int[0]), wp.float64(base_int[1]))

    # Loop on grid nodes
    for i in range(0, 3):
        for j in range(0, 3):
            ix = base_int[0] + i
            iy = base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_y = index_ij_x + n_nodes

            dofStruct.activate_flag_array[index_ij_x] = True
            dofStruct.activate_flag_array[index_ij_y] = True





# Assemble residual with tape
@wp.kernel
def assemble_residual(x_particles: wp.array(dtype=wp.vec2d),
                      inv_dx: wp.float64,
                      dx: wp.float64,
                      rhs: wp.array(dtype=wp.float64),
                      solution: wp.array(dtype=wp.float64),
                      old_solution: wp.array(dtype=wp.float64),
                      p_vol: wp.float64,
                      p_rho: wp.float64,
                      lame_lambda: wp.float64,
                      lame_mu: wp.float64,
                      deformation_gradient: wp.array(dtype=wp.mat33d),
                      n_grid_x: wp.int32,
                      n_nodes: wp.int32,
                      dofStruct: DofStruct,
                      step: wp.float64
                      ):
    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    standard_gravity = wp.vec2d(float64_zero, wp.float64(-10.0)*(step+float64_one)/wp.float64(40.0))

    # Calculate shape functions
    base_x = x_particles[p][0] * inv_dx - wp.float64(0.5)
    base_y = x_particles[p][1] * inv_dx - wp.float64(0.5)
    base_int = wp.vector(wp.int(base_x), wp.int(base_y))
    base = wp.vector(wp.float64(base_int[0]), wp.float64(base_int[1]))

    fx = x_particles[p] * inv_dx - base

    w = wp.matrix(
        wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[0], wp.float64(2.0)), wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[1], wp.float64(2.0)), # 0.5 * (1.5 - fx) ** 2
        wp.float64(0.75)-wp.pow(fx[0]-wp.float64(1.0), wp.float64(2.0)), wp.float64(0.75)-wp.pow(fx[1]-wp.float64(1.0), wp.float64(2.0)), # 0.75 - (fx - 1) ** 2
        wp.float64(0.5)*wp.pow(fx[0]-wp.float64(0.5), wp.float64(2.0)), wp.float64(0.5)*wp.pow(fx[1]-wp.float64(0.5), wp.float64(2.0)), # 0.5 * (fx - 0.5) ** 2
        shape=(3, 2)
    )

    grad_w = wp.matrix(
        (fx[0]-wp.float64(1.5))*inv_dx, (fx[1]-wp.float64(1.5))*inv_dx, # (fx - 1.5) * inv_dx
        (wp.float64(2.0)-wp.float64(2.0)*fx[0])*inv_dx, (wp.float64(2.0)-wp.float64(2.0)*fx[1])*inv_dx, # (2 - 2 * fx) * inv_dx
        (fx[0]-wp.float64(0.5))*inv_dx, (fx[1]-wp.float64(0.5))*inv_dx, # (fx - 0.5) * inv_dx
        shape=(3, 2)
    )


    # # Modification for boundary elements
    # if base_int[1]==0:
    #     w = wp.matrix(
    #         wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[0], wp.float64(2.0)), float64_zero,
    #         wp.float64(0.75)-wp.pow(fx[0]-wp.float64(1.0), wp.float64(2.0)), (wp.float64(3.0)-wp.float64(4.0)*wp.pow(fx[1]-float64_one, wp.float64(2.0)))/wp.float64(3.0), # (3 - 4 * (fx[0]-1)**2) / (3.0)
    #         wp.float64(0.5)*wp.pow(fx[0]-wp.float64(0.5), wp.float64(2.0)), wp.float64(4.0)/wp.float64(3.0) * wp.pow(fx[1]-float64_one, wp.float64(2.0)), # 4.0/3.0 * (fx[0]-1)**2
    #         shape=(3,2)
    #     )

    #     grad_w = wp.matrix(
    #         (fx[0]-wp.float64(1.5))*inv_dx, float64_zero,
    #         (wp.float64(2.0)-wp.float64(2.0)*fx[0])*inv_dx, wp.float64(-8.0)/wp.float64(3.0) * (fx[1]-float64_one) * inv_dx, # -8.0/3.0 * (fx[0]-1) * (inv_dx)
    #         (fx[0]-wp.float64(0.5))*inv_dx, wp.float64(8.0)/wp.float64(3.0) * (fx[1]-float64_one) * inv_dx, # 8.0/3.0 * (fx[0]-1) * (inv_dx)
    #         shape=(3,2)
    #     )

    #     # print(w[0][1]+w[1][1]+w[2][1])
    #     # if wp.abs(grad_w[0][1]+grad_w[1][1]+grad_w[2][1])>1e-5:
    #     # print(grad_w[0][1]+grad_w[1][1]+grad_w[2][1])

    # if base_int[1]==1:
    #     w = wp.matrix(
    #         wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[0], wp.float64(2.0)), wp.float64(2.0)/wp.float64(3.0) * wp.pow(wp.float64(3.0)/wp.float64(2.0) - fx[1], wp.float64(2.0)), # 2.0/3.0 * (3.0/2.0 - (fx[0]))**2
    #         wp.float64(0.75)-wp.pow(fx[0]-wp.float64(1.0), wp.float64(2.0)), -(wp.float64(15.0) - wp.float64(60.0)*fx[1] + wp.float64(28)*wp.pow(fx[1], wp.float64(2.0))) / wp.float64(24.0), # - (15 - 60*(fx[0]) + 28*(fx[0])**2) / (24.0)
    #         wp.float64(0.5)*wp.pow(fx[0]-wp.float64(0.5), wp.float64(2.0)), wp.float64(0.5)*wp.pow(fx[1]-wp.float64(0.5), wp.float64(2.0)),
    #         shape=(3,2)
    #     )

    #     grad_w = wp.matrix(
    #         (fx[0]-wp.float64(1.5))*inv_dx, wp.float64(4.0)/wp.float64(3.0) * (wp.float64(3.0)/wp.float64(2.0) - fx[1]) * (-inv_dx), # 4.0/3.0 * (3.0/2.0 - fx[0]) * (-inv_dx)
    #         (wp.float64(2.0)-wp.float64(2.0)*fx[0])*inv_dx, -(-wp.float64(60.0)*inv_dx + wp.float64(56.0)*fx[1]*inv_dx) / wp.float64(24.0), # - (-60*(inv_dx) + 56*(fx[0])*(inv_dx)) / 24.0
    #         (fx[0]-wp.float64(0.5))*inv_dx, (fx[1]-wp.float64(0.5))*inv_dx,
    #         shape=(3,2)
    #     )

    #     # if wp.abs(w[0][1]+w[1][1]+w[2][1]-float64_one)>1e-5:
    #     #     print((w[0][1]+w[1][1]+w[2][1]))

    #     # if wp.abs(grad_w[0][1]+grad_w[1][1]+grad_w[2][1])>1e-5:
    #     #     print(grad_w[0][1]+grad_w[1][1]+grad_w[2][1])



    # MLS modification
    if base_int[0]==0 and base_int[1]==0:
        w = wp.matrix(
            float64_zero, float64_zero,
            float64_one - (fx[0]-float64_one), float64_one - (fx[1]-float64_one),
            fx[0]-float64_one, fx[1]-float64_one,
            shape=(3,2)
        )

        grad_w = wp.matrix(
            float64_zero, float64_zero,
            -inv_dx, -inv_dx,
            inv_dx, inv_dx,
            shape=(3,2)
        )
    elif base_int[0]==0:
        w = wp.matrix(
            float64_zero,     wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[1], wp.float64(2.0)), 
            float64_one - (fx[0]-float64_one),     wp.float64(0.75)-wp.pow(fx[1]-wp.float64(1.0), wp.float64(2.0)), 
            fx[0]-float64_one,     wp.float64(0.5)*wp.pow(fx[1]-wp.float64(0.5), wp.float64(2.0)), 
            shape=(3,2)
        )

        grad_w = wp.matrix(
            float64_zero,     (fx[1]-wp.float64(1.5))*inv_dx, 
            -inv_dx,     (wp.float64(2.0)-wp.float64(2.0)*fx[1])*inv_dx, 
            inv_dx,     (fx[1]-wp.float64(0.5))*inv_dx, 
            shape=(3,2)
        )
    elif base_int[1]==0:
        w = wp.matrix(
            wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[0], wp.float64(2.0)), float64_zero,
            wp.float64(0.75)-wp.pow(fx[0]-wp.float64(1.0), wp.float64(2.0)), float64_one - (fx[1]-float64_one),
            wp.float64(0.5)*wp.pow(fx[0]-wp.float64(0.5), wp.float64(2.0)), fx[1]-float64_one,
            shape=(3,2)
        )

        grad_w = wp.matrix(
            (fx[0]-wp.float64(1.5))*inv_dx, float64_zero,
            (wp.float64(2.0)-wp.float64(2.0)*fx[0])*inv_dx, -inv_dx,
            (fx[0]-wp.float64(0.5))*inv_dx, inv_dx,
            shape=(3,2)
        )


        




    # Loop on dofs
    delta_u_GRAD = wp.matrix(float64_zero, float64_zero,
                                            float64_zero, float64_zero, shape=(2,2))
    for i in range(0, 3):
        for j in range(0, 3):
            dpos = ( wp.vec2d(wp.float64(i), wp.float64(j)) - fx) * dx
            weight = w[i][0] * w[j][1]
            weight_grad = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
            ix = base_int[0] + i
            iy = base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_y = index_ij_x + n_nodes

            node_solution = wp.vec2d(solution[index_ij_x], solution[index_ij_y])
            node_old_solution = wp.vec2d(old_solution[index_ij_x], old_solution[index_ij_y])

            delta_u_GRAD += wp.outer(weight_grad, node_solution-node_old_solution)

    old_F_3d = deformation_gradient[p]
    old_F = wp.mat22d(old_F_3d[0,0], old_F_3d[0,1],
                      old_F_3d[1,0], old_F_3d[1,1])

    incr_F = wp.identity(n=2, dtype=wp.float64) + delta_u_GRAD
    incr_F_inv = wp.inverse(incr_F)
    new_F = incr_F @ old_F

    # Neo-Hookean: From deformation gradient to stress
    # TODO: handle more complex cases
    new_F_inv = wp.inverse(new_F)
    particle_J = wp.determinant(new_F)
    particle_PK1_stress = lame_mu * (new_F - wp.transpose(new_F_inv)) + lame_lambda * wp.log(particle_J) * wp.transpose(new_F_inv)
    particle_Cauchy_stress = float64_one/particle_J * particle_PK1_stress @ wp.transpose(new_F)



    new_p_vol = p_vol * particle_J
    new_p_rho = p_rho / particle_J


    for i in range(0, 3):
        for j in range(0, 3):
            dpos = ( wp.vec2d(wp.float64(i), wp.float64(j)) - fx) * dx
            weight = w[i][0] * w[j][1]
            weight_GRAD = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
            weight_grad = incr_F_inv @ weight_GRAD # NOTE here is incr_F_inv

            ix = base_int[0] + i
            iy = base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_y = index_ij_x + n_nodes

            rhs_value = (-weight_grad @ particle_Cauchy_stress + weight * new_p_rho * standard_gravity) * new_p_vol # Updated Lagrangian



            if (dofStruct.boundary_flag_array[index_ij_x]==False):
                wp.atomic_add(rhs, index_ij_x, rhs_value[0])

            if (dofStruct.boundary_flag_array[index_ij_y]==False):
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
                if dofStruct.boundary_flag_array[row_index]==False:
                    if dofStruct.boundary_flag_array[adj_index_x]==False:
                        rows[row_index*25 + (i+j*5)] = row_index
                        cols[row_index*25 + (i+j*5)] = adj_index_x
                        vals[row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_x]

                    if dofStruct.boundary_flag_array[adj_index_y]==False:
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
                    if dofStruct.boundary_flag_array[row_index]==False:
                        if dofStruct.boundary_flag_array[adj_index_x]==False:
                            rows[row_index*25 + (i+j*5)] = row_index
                            cols[row_index*25 + (i+j*5)] = adj_index_x
                            vals[row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_x]

                        if dofStruct.boundary_flag_array[adj_index_y]==False:
                            rows[25*n_matrix_size + row_index*25 + (i+j*5)] = row_index
                            cols[25*n_matrix_size + row_index*25 + (i+j*5)] = adj_index_y
                            vals[25*n_matrix_size + row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_y]





# From increment to solution
@wp.kernel
def from_increment_to_solution(increment: wp.array(dtype=wp.float64),
                               solution: wp.array(dtype=wp.float64)):
    i = wp.tid()

    solution[i] += increment[i]



# G2P
@wp.kernel
def G2P(x_particles: wp.array(dtype=wp.vec2d),
        delta_u_particles: wp.array(dtype=wp.vec2d),
        deformation_gradient: wp.array(dtype=wp.mat33d),
        particle_Cauchy_stress_array: wp.array(dtype=wp.mat33d),
        inv_dx: wp.float64,
        dx: wp.float64,
        lame_lambda: wp.float64,
        lame_mu: wp.float64,
        n_grid_x: wp.int32,
        n_nodes: wp.int32,
        new_solution: wp.array(dtype=wp.float64),
        old_solution: wp.array(dtype=wp.float64),
        dofStruct: DofStruct):

    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    # Calculate shape functions
    base_x = x_particles[p][0] * inv_dx - wp.float64(0.5)
    base_y = x_particles[p][1] * inv_dx - wp.float64(0.5)
    base_int = wp.vector(wp.int(base_x), wp.int(base_y))
    base = wp.vector(wp.float64(base_int[0]), wp.float64(base_int[1]))

    fx = x_particles[p] * inv_dx - base

    w = wp.matrix(
        wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[0], wp.float64(2.0)), wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[1], wp.float64(2.0)), # 0.5 * (1.5 - fx) ** 2
        wp.float64(0.75)-wp.pow(fx[0]-wp.float64(1.0), wp.float64(2.0)), wp.float64(0.75)-wp.pow(fx[1]-wp.float64(1.0), wp.float64(2.0)), # 0.75 - (fx - 1) ** 2
        wp.float64(0.5)*wp.pow(fx[0]-wp.float64(0.5), wp.float64(2.0)), wp.float64(0.5)*wp.pow(fx[1]-wp.float64(0.5), wp.float64(2.0)), # 0.5 * (fx - 0.5) ** 2
        shape=(3, 2)
    )

    grad_w = wp.matrix(
        (fx[0]-wp.float64(1.5))*inv_dx, (fx[1]-wp.float64(1.5))*inv_dx, # (fx - 1.5) * inv_dx
        (wp.float64(2.0)-wp.float64(2.0)*fx[0])*inv_dx, (wp.float64(2.0)-wp.float64(2.0)*fx[1])*inv_dx, # (2 - 2 * fx) * inv_dx
        (fx[0]-wp.float64(0.5))*inv_dx, (fx[1]-wp.float64(0.5))*inv_dx, # (fx - 0.5) * inv_dx
        shape=(3, 2)
    )


    # # Modification for boundary elements
    # if base_int[1]==0:
    #     w = wp.matrix(
    #         wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[0], wp.float64(2.0)), float64_zero,
    #         wp.float64(0.75)-wp.pow(fx[0]-wp.float64(1.0), wp.float64(2.0)), (wp.float64(3.0)-wp.float64(4.0)*wp.pow(fx[1]-float64_one, wp.float64(2.0)))/wp.float64(3.0), # (3 - 4 * (fx[0]-1)**2) / (3.0)
    #         wp.float64(0.5)*wp.pow(fx[0]-wp.float64(0.5), wp.float64(2.0)), wp.float64(4.0)/wp.float64(3.0) * wp.pow(fx[1]-float64_one, wp.float64(2.0)), # 4.0/3.0 * (fx[0]-1)**2
    #         shape=(3,2)
    #     )

    #     grad_w = wp.matrix(
    #         (fx[0]-wp.float64(1.5))*inv_dx, float64_zero,
    #         (wp.float64(2.0)-wp.float64(2.0)*fx[0])*inv_dx, wp.float64(-8.0)/wp.float64(3.0) * (fx[1]-float64_one) * inv_dx, # -8.0/3.0 * (fx[0]-1) * (inv_dx)
    #         (fx[0]-wp.float64(0.5))*inv_dx, wp.float64(8.0)/wp.float64(3.0) * (fx[1]-float64_one) * inv_dx, # 8.0/3.0 * (fx[0]-1) * (inv_dx)
    #         shape=(3,2)
    #     )

    # if base_int[1]==1:
    #     w = wp.matrix(
    #         wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[0], wp.float64(2.0)), wp.float64(2.0)/wp.float64(3.0) * wp.pow(wp.float64(3.0)/wp.float64(2.0) - fx[1], wp.float64(2.0)), # 2.0/3.0 * (3.0/2.0 - (fx[0]))**2
    #         wp.float64(0.75)-wp.pow(fx[0]-wp.float64(1.0), wp.float64(2.0)), -(wp.float64(15.0) - wp.float64(60.0)*fx[1] + wp.float64(28)*wp.pow(fx[1], wp.float64(2.0))) / wp.float64(24.0), # - (15 - 60*(fx[0]) + 28*(fx[0])**2) / (24.0)
    #         wp.float64(0.5)*wp.pow(fx[0]-wp.float64(0.5), wp.float64(2.0)), wp.float64(0.5)*wp.pow(fx[1]-wp.float64(0.5), wp.float64(2.0)),
    #         shape=(3,2)
    #     )

    #     grad_w = wp.matrix(
    #         (fx[0]-wp.float64(1.5))*inv_dx, wp.float64(4.0)/wp.float64(3.0) * (wp.float64(3.0)/wp.float64(2.0) - fx[1]) * (-inv_dx), # 4.0/3.0 * (3.0/2.0 - fx[0]) * (-inv_dx)
    #         (wp.float64(2.0)-wp.float64(2.0)*fx[0])*inv_dx, -(-wp.float64(60.0)*inv_dx + wp.float64(56.0)*fx[1]*inv_dx) / wp.float64(24.0), # - (-60*(inv_dx) + 56*(fx[0])*(inv_dx)) / 24.0
    #         (fx[0]-wp.float64(0.5))*inv_dx, (fx[1]-wp.float64(0.5))*inv_dx,
    #         shape=(3,2)
    #     )


    # MLS modification
    if base_int[0]==0 and base_int[1]==0:
        w = wp.matrix(
            float64_zero, float64_zero,
            float64_one - (fx[0]-float64_one), float64_one - (fx[1]-float64_one),
            fx[0]-float64_one, fx[1]-float64_one,
            shape=(3,2)
        )

        grad_w = wp.matrix(
            float64_zero, float64_zero,
            -inv_dx, -inv_dx,
            inv_dx, inv_dx,
            shape=(3,2)
        )
    elif base_int[0]==0:
        w = wp.matrix(
            float64_zero,     wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[1], wp.float64(2.0)), 
            float64_one - (fx[0]-float64_one),     wp.float64(0.75)-wp.pow(fx[1]-wp.float64(1.0), wp.float64(2.0)), 
            fx[0]-float64_one,     wp.float64(0.5)*wp.pow(fx[1]-wp.float64(0.5), wp.float64(2.0)), 
            shape=(3,2)
        )

        grad_w = wp.matrix(
            float64_zero,     (fx[1]-wp.float64(1.5))*inv_dx, 
            -inv_dx,     (wp.float64(2.0)-wp.float64(2.0)*fx[1])*inv_dx, 
            inv_dx,     (fx[1]-wp.float64(0.5))*inv_dx, 
            shape=(3,2)
        )
    elif base_int[1]==0:
        w = wp.matrix(
            wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[0], wp.float64(2.0)), float64_zero,
            wp.float64(0.75)-wp.pow(fx[0]-wp.float64(1.0), wp.float64(2.0)), float64_one - (fx[1]-float64_one),
            wp.float64(0.5)*wp.pow(fx[0]-wp.float64(0.5), wp.float64(2.0)), fx[1]-float64_one,
            shape=(3,2)
        )

        grad_w = wp.matrix(
            (fx[0]-wp.float64(1.5))*inv_dx, float64_zero,
            (wp.float64(2.0)-wp.float64(2.0)*fx[0])*inv_dx, -inv_dx,
            (fx[0]-wp.float64(0.5))*inv_dx, inv_dx,
            shape=(3,2)
        )


    # Loop on dofs
    delta_u = wp.vec2d()
    delta_u_GRAD = wp.matrix(float64_zero, float64_zero,
                             float64_zero, float64_zero, shape=(2,2))
    for i in range(0, 3):
        for j in range(0, 3):
            dpos = ( wp.vec2d(wp.float64(i), wp.float64(j)) - fx) * dx
            weight = w[i][0] * w[j][1]
            weight_grad = wp.vector(grad_w[i][0]*w[j][1], w[i][0]*grad_w[j][1])
            ix = base_int[0] + i
            iy = base_int[1] + j

            index_ij_x = ix + iy*(n_grid_x + wp.int(1))
            index_ij_y = index_ij_x + n_nodes

            node_new_solution = wp.vec2d(new_solution[index_ij_x], new_solution[index_ij_y])
            node_old_solution = wp.vec2d(old_solution[index_ij_x], old_solution[index_ij_y])

            delta_u += weight * (node_new_solution - node_old_solution)
            delta_u_GRAD += wp.outer(weight_grad, node_new_solution-node_old_solution)

    old_F_3d = deformation_gradient[p]
    old_F = wp.mat22d(old_F_3d[0,0], old_F_3d[0,1],
                      old_F_3d[1,0], old_F_3d[1,1])

    incr_F = wp.identity(n=2, dtype=wp.float64) + delta_u_GRAD
    new_F = incr_F @ old_F


    # Save results
    deformation_gradient[p] = wp.mat33d(new_F[0,0], new_F[0,1], float64_zero,
                                        new_F[1,0], new_F[1,1], float64_zero,
                                        float64_zero, float64_zero, float64_one)

    delta_u_particles[p] = delta_u
    x_particles[p] += delta_u

    new_F_3d_inv = wp.inverse(deformation_gradient[p])
    particle_J = wp.determinant(deformation_gradient[p])
    particle_PK1_stress_3d = lame_mu * (deformation_gradient[p] - wp.transpose(new_F_3d_inv)) + lame_lambda * wp.log(particle_J) * wp.transpose(new_F_3d_inv)
    particle_Cauchy_stress_3d = float64_one/particle_J * particle_PK1_stress_3d @ wp.transpose(deformation_gradient[p])
    particle_Cauchy_stress_array[p] = particle_Cauchy_stress_3d





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
start_y = dx
end_x = start_x + 8.0
end_y = start_y + 8.0

particle_pos_np = init_particle_position(start_x, start_y, end_x, end_y, dx, n_grid_x, n_grid_y, PPD, n_particles)

x_particles = wp.from_numpy(particle_pos_np, dtype=wp.vec2d)
x0_particles = wp.from_numpy(particle_pos_np, dtype=wp.vec2d)


v_particles = wp.array(np.random.rand(n_particles, 2), dtype=wp.vec2d)
deformation_gradient = wp.array(shape=n_particles, dtype=wp.mat33d)
delta_u_particles = wp.zeros(shape=n_particles, dtype=wp.vec2d)
particle_Cauchy_stress_array = wp.zeros(shape=n_particles, dtype=wp.mat33d)

particle_solution = wp.zeros(shape=n_particles, dtype=wp.float64)
analytical_solution = wp.zeros(shape=n_particles, dtype=wp.float64)
error_abs = wp.zeros(shape=n_particles, dtype=wp.float64)

error_numerator = wp.zeros(shape=1, dtype=wp.float64)
error_denominator = wp.zeros(shape=1, dtype=wp.float64)


grid_v = wp.zeros(shape=(n_grid_x+1, n_grid_y+1), dtype=wp.vec2d)
grid_old_v = wp.zeros(shape=(n_grid_x+1, n_grid_y+1), dtype=wp.vec2d)
grid_f = wp.zeros(shape=(n_grid_x+1, n_grid_y+1), dtype=wp.vec2d)
grid_m = wp.zeros(shape=(n_grid_x+1, n_grid_y+1), dtype=wp.float64)





# Matrix
n_nodes = (n_grid_x+1) * (n_grid_y+1)
n_matrix_size = 2 * n_nodes # 2 indicates the spatial dimension
bsr_matrix = wps.bsr_zeros(n_matrix_size, n_matrix_size, block_type=wp.float64)
rhs = wp.zeros(shape=n_matrix_size, dtype=wp.float64, requires_grad=True)
increment = wp.zeros(shape=n_matrix_size, dtype=wp.float64)
old_solution = wp.zeros(shape=n_matrix_size, dtype=wp.float64)
new_solution = wp.zeros(shape=n_matrix_size, dtype=wp.float64, requires_grad=True)


rows = wp.zeros(shape=2*25*n_matrix_size, dtype=wp.int32) # 2 indicates the spatial dimension
cols = wp.zeros(shape=2*25*n_matrix_size, dtype=wp.int32)
vals = wp.zeros(shape=2*25*n_matrix_size, dtype=wp.float64)




dofStruct = DofStruct()
dofStruct.activate_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)
dofStruct.boundary_flag_array = wp.zeros(shape=n_matrix_size, dtype=wp.bool)


# Initialization
wp.launch(kernel=initialization,
          dim=n_particles,
          inputs=[deformation_gradient, x_particles, v_particles])



# Post-processing
x_numpy = np.array(x_particles.numpy())
output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': particle_Cauchy_stress_array.numpy()[:,1,1]})
output_particles.write("2d_compaction_particles_%d.vtk" % 0)


# Get the max length of the selector array
max_selector_length = 0
c_iter = 0
r_iter = 0
current_node_id = c_iter + r_iter * (n_grid_x + 1)

selector_right = np.arange(current_node_id, r_iter*(n_grid_x+1)+n_grid_x+1, 5)

selector = np.empty(0, int)
selector = np.append(selector, selector_right)

for selector_iter in range(len(selector_right)):
    this_selector = selector_right[selector_iter]
    this_selector_c = this_selector % (n_grid_x+1)
    selector_up = np.arange(this_selector+5*(n_grid_x+1), this_selector_c+n_grid_y*(n_grid_x+1)+1, 5*(n_grid_x+1))

    selector = np.append(selector, selector_up)

max_selector_length = len(selector)

selector_wp = wp.zeros(shape=max_selector_length, dtype=wp.int32)


# while True:
for step in range(40):

    grid_v.zero_()
    grid_old_v.zero_()
    grid_f.zero_()
    grid_m.zero_()


    old_solution.zero_()
    new_solution.zero_()
    increment.zero_()

    error_numerator.zero_()
    error_denominator.zero_()

    # P2G
    wp.launch(kernel=set_boundary_dofs,
              dim=(n_grid_x+1, n_grid_y+1),
              inputs=[dofStruct, n_grid_x, n_nodes])

    wp.launch(kernel=P2G,
              dim=n_particles,
              inputs=[x_particles, inv_dx, dx, n_grid_x, n_nodes, dofStruct])

    activate_flag_array_np = dofStruct.activate_flag_array.numpy()


    # Newton iteration
    # tic = time.perf_counter()
    for iter_id in range(n_iter):

        rhs.zero_()

        tape = wp.Tape()
        with tape:
            # tic = time.perf_counter()
            # assemble residual
            wp.launch(kernel=assemble_residual,
                      dim=n_particles,
                      inputs=[x_particles, inv_dx, dx, rhs, new_solution, old_solution, p_vol, p_rho, lame_lambda, lame_mu, deformation_gradient, n_grid_x, n_nodes, dofStruct, step])
            # toc = time.perf_counter()
            # print('Time for assemble_residual:', toc-tic)


        # # Naive way for auto-diff
        # for output_index in range(n_matrix_size): # Loop on dofs

        #     if activate_flag_array_np[output_index]==False: # This is important for efficient assemblage
        #         continue

        #     select_index = np.zeros(n_matrix_size)
        #     select_index[output_index] = 1.0
        #     e = wp.array(select_index, dtype=wp.float64)

        #     tape.backward(grads={rhs: e})
        #     q_grad_i = tape.gradients[new_solution]


        #     wp.launch(kernel=from_jacobian_to_vector,
        #           dim=n_matrix_size,
        #           inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, output_index, dofStruct])

                        
        #     tape.zero()


        # tic = time.perf_counter()
        # Auto-diff for sparse matrix
        # x dof
        for c_iter in range(5):
            for r_iter in range(5):
                current_node_id = c_iter + r_iter * (n_grid_x + 1)
                select_index = np.zeros(n_matrix_size)

                selector_right = np.arange(current_node_id, r_iter*(n_grid_x+1)+n_grid_x+1, 5)

                selector = np.empty(0, int)
                selector = np.append(selector, selector_right)

                for selector_iter in range(len(selector_right)):
                    this_selector = selector_right[selector_iter]
                    this_selector_c = this_selector % (n_grid_x+1)
                    selector_up = np.arange(this_selector+5*(n_grid_x+1), this_selector_c+n_grid_y*(n_grid_x+1)+1, 5*(n_grid_x+1))

                    selector = np.append(selector, selector_up)



                select_index[selector] = 1.0
                e = wp.array(select_index, dtype=wp.float64)

                # tic = time.perf_counter()
                tape.backward(grads={rhs: e})
                q_grad_i = tape.gradients[new_solution]
                # toc = time.perf_counter()
                # print('Time for auto-diff:', toc-tic)

                # tic = time.perf_counter()

                # for selector_iter in range(len(selector)):
                #     output_index = selector[selector_iter]
                #     wp.launch(kernel=from_jacobian_to_vector, TODO: THINK OF A FAST WAY
                #               dim=1, 
                #               inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, output_index, dofStruct])
                
                # fill the selector to the max length
                selector.resize(max_selector_length)
                selector_wp = wp.from_numpy(selector, dtype=wp.int32)

                wp.launch(kernel=from_jacobian_to_vector_parallel,
                          dim=max_selector_length,
                          inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, selector_wp, dofStruct])
                
                # toc = time.perf_counter()
                # print('Time for from_jacobian_to_vector:', toc-tic)

                tape.zero()

        # y dof
        for c_iter in range(5):
            for r_iter in range(5):
                current_node_id = c_iter + r_iter * (n_grid_x + 1)
                select_index = np.zeros(n_matrix_size)

                selector_right = np.arange(current_node_id, r_iter*(n_grid_x+1)+n_grid_x+1, 5)

                selector = np.empty(0, int)
                selector = np.append(selector, selector_right)

                for selector_iter in range(len(selector_right)):
                    this_selector = selector_right[selector_iter]
                    this_selector_c = this_selector % (n_grid_x+1)
                    selector_up = np.arange(this_selector+5*(n_grid_x+1), this_selector_c+n_grid_y*(n_grid_x+1)+1, 5*(n_grid_x+1))

                    selector = np.append(selector, selector_up)

                # from node id (x) to y dof
                selector += n_nodes

                select_index[selector] = 1.0
                e = wp.array(select_index, dtype=wp.float64)

                tape.backward(grads={rhs: e})
                q_grad_i = tape.gradients[new_solution]

                # for selector_iter in range(len(selector)):
                #     output_index = selector[selector_iter]

                #     wp.launch(kernel=from_jacobian_to_vector,
                #               dim=1,
                #               inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, output_index, dofStruct])

                # fill the selector to the max length
                selector.resize(max_selector_length)
                selector_wp = wp.from_numpy(selector, dtype=wp.int32)

                wp.launch(kernel=from_jacobian_to_vector_parallel,
                          dim=max_selector_length,
                          inputs=[q_grad_i, rows, cols, vals, n_grid_x, n_grid_y, n_nodes, n_matrix_size, selector_wp, dofStruct])
                

                tape.zero()
        # toc = time.perf_counter()
        # print('Time for auto-diff:', toc-tic)



        # Assemble from vectors to sparse matrix
        wps.bsr_set_from_triplets(bsr_matrix, rows, cols, vals)

        # Solve
        preconditioner = wp.optim.linear.preconditioner(bsr_matrix, ptype='diag')
        solver_state = wp.optim.linear.bicgstab(A=bsr_matrix, b=rhs, x=increment, tol=1e-10, M=preconditioner)

        # From increment to solution
        wp.launch(kernel=from_increment_to_solution,
                  dim=n_matrix_size,
                  inputs=[increment, new_solution])



        with np.printoptions(threshold=np.inf):
            # print(bsr_matrix.values.numpy())
            # print(rows.numpy())
            # print(rhs.numpy())
            # print('solver state:', solver_state)
            print('residual.norm:', np.linalg.norm(rhs.numpy()))
            # print(solution.numpy())
            # pass

        # TODO: break the Newton iteration based on the norm of the residual

    # toc = time.perf_counter()
    # print('Time for newton iteration:', toc-tic)


    # G2P
    wp.launch(kernel=G2P,
              dim=n_particles,
              inputs=[x_particles, delta_u_particles, deformation_gradient, particle_Cauchy_stress_array, inv_dx, dx, lame_lambda, lame_mu, n_grid_x, n_nodes, new_solution, old_solution, dofStruct])


    # Calculate error
    wp.launch(kernel=calculate_error,
              dim=n_particles,
              inputs=[error_numerator, error_denominator, particle_Cauchy_stress_array, x0_particles, p_rho, p_vol, start_y, n_particles])

    print('Relative error:', error_numerator.numpy()[0]/error_denominator.numpy()[0])
    

    # Post-processing
    x_numpy = np.array(x_particles.numpy())
    output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': particle_Cauchy_stress_array.numpy()[:,1,1]})
    output_particles.write("2d_compaction_particles_%d.vtk" % (step+1))

    output_frame += 1



