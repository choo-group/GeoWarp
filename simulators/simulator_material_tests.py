import sys
sys.path.append('..')

import warp as wp
import warp.sparse as wps
import warp.optim.linear

import numpy as np

import scipy.linalg as sla # check whether this is needed
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pyamg import smoothed_aggregation_solver

from simulators.material_tests_utils import initialize_elastic_cto, set_initial_stress, initial_loading_at_this_step, calculate_stress_residual_NorSand, \
assemble_Jacobian_coo_format_material_tests, from_increment_to_strain_vector_initial, from_increment_to_solution_triaxial, set_new_strain_vector_to_elastic_strain


from material_models.material_utils import initialize_pi_for_NorSand, set_old_pi_to_new_NorSand


class SimulatorTriaxial:
    def __init__(self,
                 n_iter,
                 elasticity_dict,
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
        self.total_strain_vector = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64)
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
        self.elasticity_dict = elasticity_dict
        self.youngs_modulus = self.elasticity_dict['youngs_modulus_initial']
        self.poisson_ratio = self.elasticity_dict['poisson_ratio_initial']
        self.lame_lambda = self.youngs_modulus*self.poisson_ratio / ((1.0+self.poisson_ratio) * (1.0-2.0*self.poisson_ratio))
        self.lame_mu = self.youngs_modulus / (2.0*(1.0+self.poisson_ratio))
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


        # Intermidiate quantities for local return mapping. This is important to ensure correct gradients
        self.real_strain_array = wp.zeros(shape=16, dtype=wp.mat33d, requires_grad=True) # 16 indicates the maximum number for local iteration
        self.delta_lambda_array = wp.zeros(shape=16, dtype=wp.float64, requires_grad=True)
        self.pi_array = wp.zeros(shape=16*16, dtype=wp.float64, requires_grad=True) # For Nor-Sand
        self.saved_pi = wp.zeros(shape=1, dtype=wp.float64) # For Nor-Sand
        self.old_pi = wp.zeros(shape=1, dtype=wp.float64) # For Nor-Sand
        self.saved_H = wp.zeros(shape=1, dtype=wp.float64) # saved hardening modulus

        self.saved_local_residual = wp.zeros(shape=1, dtype=wp.float64)
        self.saved_residual = wp.zeros(shape=15, dtype=wp.vec4d)


        # initialize stress and strain
        wp.launch(kernel=initialize_elastic_cto,
                  dim=1,
                  inputs=[self.rows_elastic_cto, self.cols_elastic_cto, self.vals_elastic_cto, self.lame_lambda, self.lame_mu])
        wps.bsr_set_from_triplets(self.elastic_cto, self.rows_elastic_cto, self.cols_elastic_cto, self.vals_elastic_cto, prune_numerical_zeros=False)

        # print('elastic_cto:', self.elastic_cto)

        wp.launch(kernel=set_initial_stress,
                  dim=1,
                  inputs=[self.rhs_initial_stress, self.target_stress_xx, self.target_stress_yy, self.target_stress_zz])

        if self.material_name=='Nor-Sand':
            wp.launch(kernel=initialize_pi_for_NorSand,
                      dim=1,
                      inputs=[self.plasticity_dict['pi'], self.saved_pi])

            wp.launch(kernel=set_old_pi_to_new_NorSand,
                      dim=1,
                      inputs=[self.saved_pi, self.old_pi])

        # Solve for initial strain
        preconditioner = wp.optim.linear.preconditioner(self.elastic_cto, ptype='diag')
        solver_state = wp.optim.linear.bicgstab(A=self.elastic_cto, b=self.rhs_initial_stress, x=self.strain_increment, tol=self.tol, M=preconditioner)
        # From increment to solution
        wp.launch(kernel=from_increment_to_strain_vector_initial,
                  dim=self.n_matrix_size,
                  inputs=[self.strain_increment, self.new_strain_vector])

        wp.launch(kernel=from_increment_to_strain_vector_initial,
                  dim=self.n_matrix_size,
                  inputs=[self.strain_increment, self.total_strain_vector])

        total_strain_vector_np = self.total_strain_vector.numpy()
        self.initial_volumetric_strain = -1. * (total_strain_vector_np[0] + total_strain_vector_np[1] + total_strain_vector_np[2]) * 100.0



    def reset_grid_quantities(self):
        self.strain_increment.zero_()


    def newton_iter(self, current_step, total_step):


        for iter_id in range(self.n_iter):
            self.rhs.zero_()
            self.rows.zero_()
            self.cols.zero_()
            self.vals.zero_() 

            self.real_strain_array.zero_()
            self.pi_array.zero_()
            self.delta_lambda_array.zero_()
            self.saved_local_residual.zero_()
            self.saved_residual.zero_()
            self.saved_H.zero_()


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
                              inputs=[self.new_strain_vector, self.new_elastic_strain_vector, self.lame_lambda, self.lame_mu, self.plasticity_dict['friction_angle'], self.plasticity_dict['dilation_angle'], self.plasticity_dict['cohesion'], self.plasticity_dict['shape_factor'], self.tol, self.target_stress_xx, self.target_stress_yy, self.rhs, self.saved_stress, self.real_strain_array])
                elif self.material_name=='Nor-Sand':
                    wp.launch(kernel=calculate_stress_residual_NorSand,
                              dim=1,
                              inputs=[self.new_strain_vector, self.total_strain_vector, self.new_elastic_strain_vector, self.lame_lambda, self.lame_mu, self.plasticity_dict['M'], self.plasticity_dict['N'], self.saved_pi, self.old_pi, self.plasticity_dict['tilde_lambda'], self.plasticity_dict['beta'], self.plasticity_dict['v_c0'], self.plasticity_dict['v_0'], self.plasticity_dict['h'], self.tol, self.target_stress_xx, self.target_stress_yy, self.rhs, self.saved_stress, self.real_strain_array, self.pi_array, self.delta_lambda_array, self.saved_local_residual, self.saved_residual, self.saved_H])
                    

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
                wps.bsr_set_from_triplets(self.bsr_matrix, self.rows, self.cols, self.vals, prune_numerical_zeros=False) # if setting prune_numerical_zeros==True (by default), bsr_matrix will contain NaN. See this: https://github.com/NVIDIA/warp/issues/293

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
            elif self.solver_name=='direct':
                bsr_matrix_direct = sparse.coo_matrix((self.vals.numpy(), (self.rows.numpy(), self.cols.numpy())), shape=(self.n_matrix_size, self.n_matrix_size)).asformat('csr')
                b = self.rhs.numpy()
                x_direct = spsolve(bsr_matrix_direct, b)
                self.strain_increment = wp.from_numpy(x_direct, dtype=wp.float64)






            # From increment to solution
            wp.launch(kernel=from_increment_to_solution_triaxial,
                      dim=self.n_matrix_size,
                      inputs=[self.strain_increment, self.new_strain_vector])

            wp.launch(kernel=from_increment_to_solution_triaxial,
                      dim=self.n_matrix_size,
                      inputs=[self.strain_increment, self.total_strain_vector])

            # if current_step==1:
            #     with np.printoptions(threshold=np.inf):
            #         # print(np.linalg.norm(self.rhs.numpy())) # print residual
            #         # print('local residual p:', self.saved_local_residual.numpy())
            #         for local_iter in range(15):
            #             print(local_iter, np.linalg.norm(self.saved_residual.numpy()[local_iter,:]))

            # # Print the converged stress
            # saved_stress_np = self.saved_stress.numpy()
            # print('stress:', saved_stress_np)


            if np.linalg.norm(self.rhs.numpy())<self.tol:
                break

        # Set old to new
        wp.launch(kernel=set_old_pi_to_new_NorSand,
                  dim=1,
                  inputs=[self.saved_pi, self.old_pi])

        # Print the converged stress
        saved_stress_np = self.saved_stress.numpy()
        # print((current_step+1)*self.loading_rate, -saved_stress_np[0][2,2]) # print strain-stress

        # Q-P
        q_invariant = -saved_stress_np[0][2,2]-(-saved_stress_np[0][0,0])
        p_invariant = -1./3. * (saved_stress_np[0][0,0] + saved_stress_np[0][1,1] + saved_stress_np[0][2,2])
        pi = self.saved_pi[0:].numpy()[0]
        hardening_H = self.saved_H[0:].numpy()[0]
        # print((current_step+1)*self.loading_rate*100.0, q_invariant) # print axial strain-stress
        
        # Plastic strain
        total_strain_vector_np = self.total_strain_vector.numpy()
        plastic_strain_vector_np = self.total_strain_vector.numpy() - self.new_elastic_strain_vector.numpy()
        # triaxial_shear_strain = (-2.0/3.0 * (plastic_strain_vector_np[2] - plastic_strain_vector_np[0])) * 100.0
        triaxial_shear_strain = (-2.0/3.0 * (total_strain_vector_np[2] - total_strain_vector_np[0])) * 100.0
        stress_ratio = q_invariant/p_invariant
        # print(triaxial_shear_strain, stress_ratio)


        # Volumetric strain-axial strain
        total_strain_vector_np = self.total_strain_vector.numpy()
        volumetric_strain = -1. * (total_strain_vector_np[0] + total_strain_vector_np[1] + total_strain_vector_np[2]) * 100.0 - self.initial_volumetric_strain
        eps_v_total = total_strain_vector_np[0] + total_strain_vector_np[1] + total_strain_vector_np[2]
        new_J = np.exp(eps_v_total)
        vf = new_J * (self.plasticity_dict['v_0'])
        ef = vf - 1.0
        # volumetric_strain = (-(ef-(self.plasticity_dict['v_0']-1.0))/(1.0+(self.plasticity_dict['v_0']-1.0))) * 100.0
        # print((current_step+1)*self.loading_rate*100.0, volumetric_strain)
        print(triaxial_shear_strain, volumetric_strain)


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

        wp.launch(kernel=initial_loading_at_this_step,
                  dim=1,
                  inputs=[self.total_strain_vector, self.loading_rate])

        # print('initial_strain_vector for this step:', self.new_strain_vector.numpy())

        # Newton iteration
        self.newton_iter(current_step, total_step)

        # Calculate error
        # TODO


