import warp as wp
import numpy as np

def init_particles_rectangle(start_x, start_y, end_x, end_y, dx, n_grid_x, n_grid_y, PPD, n_particles):
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
def initialization(deformation_gradient: wp.array(dtype=wp.mat33d)):
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




# P2G
@wp.kernel
def P2G(x_particles: wp.array(dtype=wp.vec2d),
        inv_dx: wp.float64,
        dx: wp.float64,
        n_grid_x: wp.int32,
        n_nodes: wp.int32,
        activate_flag_array: wp.array(dtype=wp.bool),
        ):
    p = wp.tid()


    # Get base
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

            activate_flag_array[index_ij_x] = True
            activate_flag_array[index_ij_y] = True



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
                      boundary_flag_array: wp.array(dtype=wp.bool),
                      current_step: wp.float64,
                      total_step: wp.float64
                      ):
    p = wp.tid()

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)
    float64_pi = wp.float64(3.141592653)

    standard_gravity = wp.vec2d(float64_zero, wp.float64(-10.0)*(current_step+float64_one)/total_step)

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



    # # MLS modification
    # if base_int[1]==0:
    #     w = wp.matrix(
    #         wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[0], wp.float64(2.0)), float64_zero,
    #         wp.float64(0.75)-wp.pow(fx[0]-wp.float64(1.0), wp.float64(2.0)), float64_one - (fx[1]-float64_one),
    #         wp.float64(0.5)*wp.pow(fx[0]-wp.float64(0.5), wp.float64(2.0)), fx[1]-float64_one,
    #         shape=(3,2)
    #     )

    #     grad_w = wp.matrix(
    #         (fx[0]-wp.float64(1.5))*inv_dx, float64_zero,
    #         (wp.float64(2.0)-wp.float64(2.0)*fx[0])*inv_dx, -inv_dx,
    #         (fx[0]-wp.float64(0.5))*inv_dx, inv_dx,
    #         shape=(3,2)
    #     )


        




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



            if (boundary_flag_array[index_ij_x]==False):
                wp.atomic_add(rhs, index_ij_x, rhs_value[0])

            if (boundary_flag_array[index_ij_y]==False):
                wp.atomic_add(rhs, index_ij_y, rhs_value[1])



# Assemble the Jacobian matrix in COOrdinate format
@wp.kernel
def assemble_Jacobian_coo_format(jacobian_wp: wp.array(dtype=wp.float64),
                                 rows: wp.array(dtype=wp.int32),
                                 cols: wp.array(dtype=wp.int32),
                                 vals: wp.array(dtype=wp.float64),
                                 n_grid_x: wp.int32,
                                 n_grid_y: wp.int32,
                                 n_nodes: wp.int32,
                                 n_matrix_size: wp.int32,
                                 dof_iter: wp.int32,
                                 boundary_flag_array: wp.array(dtype=wp.bool)):
    
    column_index = wp.tid()

    # from dof to node_id
    node_idx = wp.int(0)
    node_idy = wp.int(0)

    if dof_iter<n_nodes:
        node_idx = wp.mod(dof_iter, n_grid_x+1)
        node_idy = wp.int((dof_iter-node_idx)/(n_grid_x+1))
    else:
        node_idx = wp.mod((dof_iter-n_nodes), n_grid_x+1)
        node_idy = wp.int((dof_iter-n_nodes)/(n_grid_x+1))


    for i in range(5):
        adj_node_idx = node_idx + (i-2)
        for j in range(5):
            adj_node_idy = node_idy + (j-2)

            adj_index_x = adj_node_idx + adj_node_idy*(n_grid_x+1)
            adj_index_y = adj_index_x + n_nodes

            if adj_node_idx>=0 and adj_node_idx<=n_grid_x and adj_node_idy>=0 and adj_node_idy<=n_grid_y: # adj_node is reasonable
                if boundary_flag_array[dof_iter]==False:
                    if boundary_flag_array[adj_index_x]==False:
                        rows[dof_iter*25 + (i+j*5)] = dof_iter
                        cols[dof_iter*25 + (i+j*5)] = adj_index_x
                        vals[dof_iter*25 + (i+j*5)] = -jacobian_wp[adj_index_x]

                    if boundary_flag_array[adj_index_y]==False:
                        rows[25*n_matrix_size + dof_iter*25 + (i+j*5)] = dof_iter
                        cols[25*n_matrix_size + dof_iter*25 + (i+j*5)] = adj_index_y
                        vals[25*n_matrix_size + dof_iter*25 + (i+j*5)] = -jacobian_wp[adj_index_y]
                
                
# From increment to solution
@wp.kernel
def from_increment_to_solution(increment: wp.array(dtype=wp.float64),
                               solution: wp.array(dtype=wp.float64)):
    i = wp.tid()

    solution[i] += increment[i]




# G2P
@wp.kernel
def G2P(x_particles: wp.array(dtype=wp.vec2d),
        deformation_gradient: wp.array(dtype=wp.mat33d),
        particle_Cauchy_stress_array: wp.array(dtype=wp.mat33d),
        inv_dx: wp.float64,
        dx: wp.float64,
        lame_lambda: wp.float64,
        lame_mu: wp.float64,
        n_grid_x: wp.int32,
        n_nodes: wp.int32,
        new_solution: wp.array(dtype=wp.float64),
        old_solution: wp.array(dtype=wp.float64)):

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


    # # MLS modification
    # if base_int[1]==0:
    #     w = wp.matrix(
    #         wp.float64(0.5)*wp.pow(wp.float64(1.5)-fx[0], wp.float64(2.0)), float64_zero,
    #         wp.float64(0.75)-wp.pow(fx[0]-wp.float64(1.0), wp.float64(2.0)), float64_one - (fx[1]-float64_one),
    #         wp.float64(0.5)*wp.pow(fx[0]-wp.float64(0.5), wp.float64(2.0)), fx[1]-float64_one,
    #         shape=(3,2)
    #     )

    #     grad_w = wp.matrix(
    #         (fx[0]-wp.float64(1.5))*inv_dx, float64_zero,
    #         (wp.float64(2.0)-wp.float64(2.0)*fx[0])*inv_dx, -inv_dx,
    #         (fx[0]-wp.float64(0.5))*inv_dx, inv_dx,
    #         shape=(3,2)
    #     )


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

    x_particles[p] += delta_u

    new_F_3d_inv = wp.inverse(deformation_gradient[p])
    particle_J = wp.determinant(deformation_gradient[p])
    particle_PK1_stress_3d = lame_mu * (deformation_gradient[p] - wp.transpose(new_F_3d_inv)) + lame_lambda * wp.log(particle_J) * wp.transpose(new_F_3d_inv)
    particle_Cauchy_stress_3d = float64_one/particle_J * particle_PK1_stress_3d @ wp.transpose(deformation_gradient[p])
    particle_Cauchy_stress_array[p] = particle_Cauchy_stress_3d







