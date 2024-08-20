import sys
sys.path.append('..')

import warp as wp
import warp.sparse as wps
import warp.optim.linear

import numpy as np

import scipy.linalg as sla # check whether this is needed
from scipy import sparse
from pyamg import smoothed_aggregation_solver

from simulators.material_tests_utils import initialize_elastic_cto, set_initial_stress, initial_loading_at_this_step, calculate_stress_residual_J2, calculate_stress_residual_DP, \
assemble_Jacobian_coo_format_material_tests, from_increment_to_strain_vector_initial, from_increment_to_solution_triaxial, set_new_strain_vector_to_elastic_strain


from material_models.material_utils import return_mapping_DP_kernel


class SimulatorTriaxial:
    def __init__(self,
                 n_iter,
                 youngs_modulus,
                 poisson_ratio,
                 material_name,
                 tol,
                 plasticity_dict,
                 loading_rate,
                 target_stress_xx,
                 target_stress_yy,
                 target_stress_zz,
                 solver_name='Warp'
        ):
        
        
        # Dof quantities
        self.n_matrix_size = 6


        self.bsr_matrix = wps.bsr_zeros(self.n_matrix_size, self.n_matrix_size, block_type=wp.float64)
        self.rhs = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64, requires_grad=True)
        self.rhs_initial_stress = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64)
        self.new_strain_vector = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64, requires_grad=True)
        self.new_elastic_strain_vector = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64, requires_grad=True)
        self.strain_increment = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64)


        self.rows_elastic_cto = wp.zeros(shape=36, dtype=wp.int32)
        self.cols_elastic_cto = wp.zeros(shape=36, dtype=wp.int32)
        self.vals_elastic_cto = wp.zeros(shape=36, dtype=wp.float64)

        self.rows = wp.zeros(shape=36, dtype=wp.int32) # TODO: change the shape
        self.cols = wp.zeros(shape=36, dtype=wp.int32)
        self.vals = wp.zeros(shape=36, dtype=wp.float64)

        # Solver
        self.n_iter = n_iter
        self.tol = tol
        self.solver_name = solver_name



        
        # Material properties
        self.youngs_modulus = youngs_modulus
        self.poisson_ratio = poisson_ratio
        self.lame_lambda = youngs_modulus*poisson_ratio / ((1.0+poisson_ratio) * (1.0-2.0*poisson_ratio))
        self.lame_mu = youngs_modulus / (2.0*(1.0+poisson_ratio))
        self.elastic_cto = wps.bsr_zeros(6, 6, block_type=wp.float64)

        self.plasticity_dict = plasticity_dict

        self.material_name = material_name
        


        # Loading
        self.loading_rate = loading_rate
        self.target_stress_xx = target_stress_xx
        self.target_stress_yy = target_stress_yy
        self.target_stress_zz = target_stress_zz


        # Post-processing
        self.saved_stress = wp.array(shape=1, dtype=wp.mat33d)


        # Intermidiate quantities for local return mapping
        self.real_strain_array = wp.zeros(shape=16, dtype=wp.mat33d, requires_grad=True)
        self.P_trial_array = wp.zeros(shape=16, dtype=wp.float64, requires_grad=True)
        self.Q_trial_array = wp.zeros(shape=16, dtype=wp.float64, requires_grad=True)
        self.tau_trial_array = wp.zeros(shape=16, dtype=wp.mat33d, requires_grad=True)
        self.delta_lambda_array = wp.zeros(shape=16, dtype=wp.float64, requires_grad=True)


        # initialize stress and strain (TODO: SEEMS THIS CAUSES THE NAN BUG)
        wp.launch(kernel=initialize_elastic_cto,
                  dim=1,
                  inputs=[self.rows_elastic_cto, self.cols_elastic_cto, self.vals_elastic_cto, self.lame_lambda, self.lame_mu])
        wps.bsr_set_from_triplets(self.elastic_cto, self.rows_elastic_cto, self.cols_elastic_cto, self.vals_elastic_cto, prune_numerical_zeros=False)

        print('elastic_cto:', self.elastic_cto)

        wp.launch(kernel=set_initial_stress,
                  dim=1,
                  inputs=[self.rhs_initial_stress, self.target_stress_xx, self.target_stress_yy, self.target_stress_zz])

        # Solve for initial strain
        preconditioner = wp.optim.linear.preconditioner(self.elastic_cto, ptype='diag')
        solver_state = wp.optim.linear.bicgstab(A=self.elastic_cto, b=self.rhs_initial_stress, x=self.strain_increment, tol=self.tol, M=preconditioner)
        # From increment to solution
        wp.launch(kernel=from_increment_to_strain_vector_initial,
                  dim=self.n_matrix_size,
                  inputs=[self.strain_increment, self.new_strain_vector])



    def reset_grid_quantities(self):
        self.strain_increment.zero_()


    def newton_iter(self, current_step, total_step):


        for iter_id in range(self.n_iter):
            self.rhs.zero_()
            self.rows.zero_()
            self.cols.zero_()
            self.vals.zero_() 

            self.real_strain_array.zero_()
            self.P_trial_array.zero_()
            self.Q_trial_array.zero_()
            self.tau_trial_array.zero_()
            self.delta_lambda_array.zero_()

            tape = wp.Tape()
            with tape:
                # calculate stress residual
                if self.material_name=='J2':
                    wp.launch(kernel=calculate_stress_residual_J2,
                              dim=1,
                              inputs=[self.new_strain_vector, self.lame_lambda, self.lame_mu, self.plasticity_dict['kappa'], self.target_stress_xx, self.target_stress_yy, self.rhs, self.saved_stress])
                elif self.material_name=='Drucker-Prager':
                    wp.launch(kernel=calculate_stress_residual_DP,
                              dim=1,
                              inputs=[self.new_strain_vector, self.new_elastic_strain_vector, self.lame_lambda, self.lame_mu, self.plasticity_dict['friction_angle'], self.plasticity_dict['dilation_angle'], self.plasticity_dict['cohesion'], self.plasticity_dict['shape_factor'], self.tol, self.target_stress_xx, self.target_stress_yy, self.rhs, self.saved_stress, self.real_strain_array, self.P_trial_array, self.Q_trial_array, self.tau_trial_array, self.delta_lambda_array])

                    # wp.launch(kernel=return_mapping_DP_kernel,
                    #           dim=1,
                    #           inputs=[self.new_strain_vector, self.lame_lambda, self.lame_mu, self.plasticity_dict['friction_angle'], self.plasticity_dict['dilation_angle'], self.plasticity_dict['cohesion'], self.plasticity_dict['shape_factor'], self.tol, self.target_stress_xx, self.target_stress_yy, self.rhs, self.saved_stress, self.real_strain_array, self.delta_lambda_array])


            # Assemble the Jacobian matrix using auto-differentiation
            for dof_iter in range(self.n_matrix_size): # Loop on dofs
                

                # get the gradient of the rhs[dof] w.r.t. the solution vector. Refer to: https://nvidia.github.io/warp/modules/differentiability.html#jacobians
                select_index = np.zeros(self.n_matrix_size)
                select_index[dof_iter] = 1.0
                e = wp.array(select_index, dtype=wp.float64)

                tape.backward(grads={self.rhs: e})
                q_grad_i = tape.gradients[self.new_strain_vector]

                # assemble the Jacobian matrix in COOrdinate format
                wp.launch(kernel=assemble_Jacobian_coo_format_material_tests,
                          dim=self.n_matrix_size,
                          inputs=[q_grad_i, self.rows, self.cols, self.vals, dof_iter])

                tape.zero()

            tape.reset()





            if self.solver_name=='Warp':
                # print('vals:', self.vals.numpy())
                # print('rows:', self.rows.numpy())
                # print('cols:', self.cols.numpy())
                # Create sparse matrix from a corresponding COOrdinate (a.k.a. triplet) format
                wps.bsr_set_from_triplets(self.bsr_matrix, self.rows, self.cols, self.vals, prune_numerical_zeros=False) # TODO: why there are NaNs even vals looks correct?

                # print('self.bsr_matrix:', self.bsr_matrix)

                # Warp solve
                preconditioner = wp.optim.linear.preconditioner(self.bsr_matrix, ptype='diag')
                # print('rhs:', self.rhs.numpy())
                solver_state = wp.optim.linear.bicgstab(A=self.bsr_matrix, b=self.rhs, x=self.strain_increment, tol=1e-10, M=preconditioner)
                # print('strain_increment:', self.strain_increment.numpy())
            elif self.solver_name=='pyamg':
                # print('vals:', self.vals.numpy())
                # print('rows:', self.rows.numpy())
                # print('cols:', self.cols.numpy())
                bsr_matrix_pyamg = sparse.coo_matrix((self.vals.numpy(), (self.rows.numpy(), self.cols.numpy())), shape=(self.n_matrix_size, self.n_matrix_size)).asformat('csr')

                # print('bsr_matrix_pyamg:', bsr_matrix_pyamg)
                # Pyamg solver
                mls = smoothed_aggregation_solver(bsr_matrix_pyamg)
                b = self.rhs.numpy()
                # print('b:', b)
                x_pyamg = mls.solve(b, tol=self.tol, accel='bicgstab')
                self.strain_increment = wp.from_numpy(x_pyamg, dtype=wp.float64)

                # print('x_pyamg:', x_pyamg)
                # print("residual from pyamg:", np.linalg.norm(b-bsr_matrix_pyamg*x_pyamg))




            # From increment to solution
            wp.launch(kernel=from_increment_to_solution_triaxial,
                      dim=self.n_matrix_size,
                      inputs=[self.strain_increment, self.new_strain_vector])

            with np.printoptions(threshold=np.inf):
                print('residual.norm:', np.linalg.norm(self.rhs.numpy()))

            # # Print the converged stress
            # saved_stress_np = self.saved_stress.numpy()
            # print('stress:', saved_stress_np)


            if np.linalg.norm(self.rhs.numpy())<self.tol:
                break

        # Print the converged stress
        saved_stress_np = self.saved_stress.numpy()
        # print('stress:', saved_stress_np)
        # print((current_step+1)*self.loading_rate, -saved_stress_np[0][2,2])


        # Set the new_strain_vector to elastic strain
        wp.launch(kernel=set_new_strain_vector_to_elastic_strain,
                  dim=1,
                  inputs=[self.new_strain_vector, self.new_elastic_strain_vector])


        


            





    def advance_one_step(self, current_step, total_step):
        self.reset_grid_quantities()

        # Initial loading at this step
        wp.launch(kernel=initial_loading_at_this_step,
                  dim=1,
                  inputs=[self.new_strain_vector, self.loading_rate])

        # print('initial_strain_vector for this step:', self.new_strain_vector.numpy())

        # Newton iteration
        self.newton_iter(current_step, total_step)

        # Calculate error
        # TODO


