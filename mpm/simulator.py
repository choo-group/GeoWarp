import sys
sys.path.append('..')

import warp as wp
import warp.sparse as wps
import warp.optim.linear

import numpy as np

from mpm.mpm_utils import init_particles_rectangle, initialization, P2G, assemble_residual, assemble_Jacobian_coo_format, from_increment_to_solution, G2P


@wp.struct
class DofStruct:
    activate_flag_array: wp.array(dtype=wp.bool)
    boundary_flag_array: wp.array(dtype=wp.bool)


class SimulatorQuasiStatic:
    def __init__(self,
                 n_grid_x,
                 n_grid_y,
                 max_x,
                 n_iter,
                 n_particles,
                 start_x,
                 end_x,
                 start_y,
                 end_y,
                 PPD,
                 p_vol,
                 p_rho,
                 youngs_modulus,
                 poisson_ratio,
                 material_name,
                 boundary_function_warp,
                 tol,
                 gravity_load_scale=1.0
        ):
        # Grid quantities
        self.n_grid_x = n_grid_x
        self.n_grid_y = n_grid_y
        self.n_nodes = (self.n_grid_x+1) * (self.n_grid_y+1)
        self.max_x = max_x
        self.dx = self.max_x/self.n_grid_x
        self.inv_dx = float(self.n_grid_x/self.max_x)
        
        # Dof quantities
        self.n_matrix_size = 2 * self.n_nodes # 2 indicates the spatial dimension
        self.dofStruct = DofStruct()
        self.dofStruct.activate_flag_array = wp.zeros(shape=self.n_matrix_size, dtype=wp.bool)
        self.dofStruct.boundary_flag_array = wp.zeros(shape=self.n_matrix_size, dtype=wp.bool)

        self.bsr_matrix = wps.bsr_zeros(self.n_matrix_size, self.n_matrix_size, block_type=wp.float64)
        self.rhs = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64, requires_grad=True)
        self.increment = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64)
        self.old_solution = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64)
        self.new_solution = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64, requires_grad=True)

        self.rows = wp.zeros(shape=2*25*self.n_matrix_size, dtype=wp.int32) # 2 indicates the spatial dimension
        self.cols = wp.zeros(shape=2*25*self.n_matrix_size, dtype=wp.int32)
        self.vals = wp.zeros(shape=2*25*self.n_matrix_size, dtype=wp.float64)

        # Solver
        self.n_iter = n_iter
        self.boundary_function_warp = boundary_function_warp # TODO: delete
        self.tol = tol

        # Load
        self.gravity_load_scale = gravity_load_scale
        self.total_external_force = wp.array(shape=1, dtype=wp.vec2d)


        # Material point quantities
        self.n_particles = n_particles
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.PPD = PPD
        self.p_vol = p_vol
        self.p_rho = p_rho

        particles_pos_np = init_particles_rectangle(self.start_x, self.start_y, self.end_x, self.end_y, self.dx, self.n_grid_x, self.n_grid_y, self.PPD, self.n_particles)
        self.x_particles = wp.from_numpy(particles_pos_np, dtype=wp.vec2d)
        self.x0_particles = wp.from_numpy(particles_pos_np, dtype=wp.vec2d)
        self.deformation_gradient = wp.zeros(shape=self.n_particles, dtype=wp.mat33d)
        self.particle_external_flag_array = wp.zeros(shape=self.n_particles, dtype=wp.bool)
        self.particle_Cauchy_stress_array = wp.zeros(shape=self.n_particles, dtype=wp.mat33d)

        # Material properties
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.lame_lambda = youngs_modulus*poisson_ratio / ((1.0+poisson_ratio) * (1.0-2.0*poisson_ratio))
        self.lame_mu = youngs_modulus / (2.0*(1.0+poisson_ratio))

        self.material_type = 100
        if material_name=='Neo-Hookean':
            self.material_type = 0
        elif material_name=='Hencky elasticity':
            self.material_type = 1


        # Initialization
        wp.launch(kernel=initialization,
                  dim=self.n_particles,
                  inputs=[self.deformation_gradient])

        # # Specify Dirichlet boundary conditions
        # wp.launch(kernel=self.boundary_function_warp,
        #           dim=(self.n_grid_x+1, self.n_grid_y+1),
        #           inputs=[self.dofStruct.boundary_flag_array, self.n_grid_x, self.n_nodes])



    def reset_grid_quantities(self):
        self.old_solution.zero_()
        self.new_solution.zero_()
        self.increment.zero_()


    def newton_iter(self, current_step, total_step):

        activate_flag_array_np = self.dofStruct.activate_flag_array.numpy()

        for iter_id in range(self.n_iter):
            self.rhs.zero_()
            self.rows.zero_()
            self.cols.zero_()
            self.vals.zero_() # Check whether this can improve the 2D compaction involving plasticity

            tape = wp.Tape()
            with tape:
                # assemble residual
                wp.launch(kernel=assemble_residual,
                          dim=self.n_particles,
                          inputs=[self.x_particles, self.inv_dx, self.dx, self.rhs, self.new_solution, self.old_solution, self.p_vol, self.p_rho, self.lame_lambda, self.lame_mu, self.material_type, self.deformation_gradient, self.particle_Cauchy_stress_array, self.n_grid_x, self.n_nodes, self.dofStruct.boundary_flag_array, self.total_external_force, self.particle_external_flag_array, self.gravity_load_scale, current_step, total_step])

            # Assemble the Jacobian matrix using auto-differentiation
            for dof_iter in range(self.n_matrix_size): # Loop on dofs
                if activate_flag_array_np[dof_iter]==False: # This is important for efficient assemblage
                    continue

                # get the gradient of the rhs[dof] w.r.t. the solution vector. Refer to: https://nvidia.github.io/warp/modules/differentiability.html#jacobians
                select_index = np.zeros(self.n_matrix_size)
                select_index[dof_iter] = 1.0
                e = wp.array(select_index, dtype=wp.float64)

                tape.backward(grads={self.rhs: e})
                q_grad_i = tape.gradients[self.new_solution]

                # assemble the Jacobian matrix in COOrdinate format
                wp.launch(kernel=assemble_Jacobian_coo_format,
                          dim=self.n_matrix_size,
                          inputs=[q_grad_i, self.rows, self.cols, self.vals, self.n_grid_x, self.n_grid_y, self.n_nodes, self.n_matrix_size, dof_iter, self.dofStruct.boundary_flag_array])

                tape.zero()


            # Create sparse matrix from a corresponding COOrdinate (a.k.a. triplet) format
            wps.bsr_set_from_triplets(self.bsr_matrix, self.rows, self.cols, self.vals)

            # Solve
            preconditioner = wp.optim.linear.preconditioner(self.bsr_matrix, ptype='diag')
            solver_state = wp.optim.linear.bicgstab(A=self.bsr_matrix, b=self.rhs, x=self.increment, tol=1e-10, M=preconditioner)

            # From increment to solution
            wp.launch(kernel=from_increment_to_solution,
                      dim=self.n_matrix_size,
                      inputs=[self.increment, self.new_solution])

            with np.printoptions(threshold=np.inf):
                print('residual.norm:', np.linalg.norm(self.rhs.numpy()))


            if np.linalg.norm(self.rhs.numpy())<self.tol:
                break




    def advance_one_step(self, current_step, total_step):
        self.reset_grid_quantities()

        # P2G
        wp.launch(kernel=P2G,
                  dim=self.n_particles,
                  inputs=[self.x_particles, self.inv_dx, self.dx, self.n_grid_x, self.n_nodes, self.dofStruct.activate_flag_array])



        # Newton iteration
        self.newton_iter(current_step, total_step)

        # G2P
        wp.launch(kernel=G2P,
              dim=self.n_particles,
              inputs=[self.x_particles, self.deformation_gradient, self.inv_dx, self.dx, self.lame_lambda, self.lame_mu, self.n_grid_x, self.n_nodes, self.new_solution, self.old_solution])

        # Calculate error
        # TODO


