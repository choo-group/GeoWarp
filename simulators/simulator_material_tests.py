import sys
sys.path.append('..')

import warp as wp
import warp.sparse as wps
import warp.optim.linear

import numpy as np

import scipy.linalg as sla # check whether this is needed
from scipy import sparse
from pyamg import smoothed_aggregation_solver

from material_models.material_utils import return_mapping_J2


@wp.struct
class DofStruct:
    activate_flag_array: wp.array(dtype=wp.bool)
    boundary_flag_array: wp.array(dtype=wp.bool)


@wp.kernel
def initialize_elastic_cto(rows: wp.array(dtype=wp.int32),
                           cols: wp.array(dtype=wp.int32),
                           vals: wp.array(dtype=wp.float64),
                           lame_lambda: wp.float64,
                           lame_mu: wp.float64):

    # elastic stiffness (isotropic) in Voigt notation (refer to: https://en.wikipedia.org/wiki/Hooke%27s_law)
    for i in range(3):
        for j in range(3):
            flattened_id = i*6 + j
            rows[flattened_id] = i
            cols[flattened_id] = j
            vals[flattened_id] = lame_lambda
            if i==j:
                vals[flattened_id] += wp.float64(2.)*lame_mu

    for i in range(3, 6):
        j = i
        flattened_id = i*6 + j
        rows[flattened_id] = i
        cols[flattened_id] = j
        vals[flattened_id] = lame_mu

@wp.kernel
def set_initial_stress(rhs: wp.array(dtype=wp.float64),
                       target_stress_xx: wp.float64,
                       target_stress_yy: wp.float64,
                       target_stress_zz: wp.float64):

    rhs[0] = target_stress_xx
    rhs[1] = target_stress_yy
    rhs[2] = target_stress_zz



@wp.kernel
def initial_loading_at_this_step(new_strain_vector: wp.array(dtype=wp.float64),
                                 loading_rate: wp.float64):
    new_strain_vector[2] += -loading_rate


@wp.kernel
def calculate_stress_residual_J2(new_strain_vector: wp.array(dtype=wp.float64), # Voigt notation
                                 lame_lambda: wp.float64,
                                 lame_mu: wp.float64,
                                 kappa: wp.float64,
                                 target_stress_xx: wp.float64,
                                 target_stress_yy: wp.float64,
                                 rhs: wp.array(dtype=wp.float64),
                                 saved_stress: wp.array(dtype=wp.mat33d)):

    float64_one = wp.float64(1.0)
    float64_zero = wp.float64(0.0)

    # Here assuming the strain only involves normal components (i.e., no shearing)
    new_strain = wp.matrix(
                 new_strain_vector[0], float64_zero, float64_zero,
                 float64_zero, new_strain_vector[1], float64_zero,
                 float64_zero, float64_zero, new_strain_vector[2],
                 shape=(3,3)
                 )

    e_real = return_mapping_J2(new_strain, lame_lambda, lame_mu, kappa)

    e_trace = wp.trace(e_real)

    stress_principal = lame_lambda*e_trace*wp.identity(n=3, dtype=wp.float64) + wp.float64(2.)*lame_mu*e_real

    saved_stress[0] = stress_principal

    # Here assuming the stress only involves normal components (i.e., no shearing)
    target_stress = wp.matrix(
                    target_stress_xx, float64_zero, float64_zero,
                    float64_zero, target_stress_yy, float64_zero,
                    float64_zero, float64_zero, stress_principal[2,2],
                    shape=(3,3)
                    )
    stress_residual = target_stress - stress_principal

    # Assemble to rhs
    wp.atomic_add(rhs, 0, stress_residual[0,0])
    wp.atomic_add(rhs, 1, stress_residual[1,1])



@wp.kernel
def assemble_Jacobian_coo_format_material_tests(jacobian_wp: wp.array(dtype=wp.float64),
                                                rows: wp.array(dtype=wp.int32),
                                                cols: wp.array(dtype=wp.int32),
                                                vals: wp.array(dtype=wp.float64),
                                                dof_iter: wp.int32):
    column_index = wp.tid()

    rows[6*dof_iter + column_index] = dof_iter
    cols[6*dof_iter + column_index] = column_index
    vals[6*dof_iter + column_index] = jacobian_wp[column_index]


@wp.kernel
def from_increment_to_solution(strain_increment: wp.array(dtype=wp.float64),
                               new_strain_vector: wp.array(dtype=wp.float64)):
    i = wp.tid()

    new_strain_vector[i] += strain_increment[i]

@wp.kernel
def from_increment_to_solution_triaxial(strain_increment: wp.array(dtype=wp.float64),
                                        new_strain_vector: wp.array(dtype=wp.float64)):
    i = wp.tid()

    if i!=2 and i!=3 and i!=4:
        new_strain_vector[i] -= strain_increment[i] # TODO: check why


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
        self.dofStruct = DofStruct()
        self.dofStruct.activate_flag_array = wp.zeros(shape=self.n_matrix_size, dtype=wp.bool)
        self.dofStruct.boundary_flag_array = wp.zeros(shape=self.n_matrix_size, dtype=wp.bool)

        self.bsr_matrix = wps.bsr_zeros(self.n_matrix_size, self.n_matrix_size, block_type=wp.float64)
        self.rhs = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64, requires_grad=True)
        self.new_strain_vector = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64, requires_grad=True)
        self.strain_increment = wp.zeros(shape=self.n_matrix_size, dtype=wp.float64)

        self.rows = wp.zeros(shape=self.n_matrix_size*self.n_matrix_size, dtype=wp.int32) # TODO: change the shape
        self.cols = wp.zeros(shape=self.n_matrix_size*self.n_matrix_size, dtype=wp.int32)
        self.vals = wp.zeros(shape=self.n_matrix_size*self.n_matrix_size, dtype=wp.float64)

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

        self.material_type = 100
        if material_name=='Neo-Hookean':
            self.material_type = 0
        elif material_name=='Hencky elasticity':
            self.material_type = 1
        elif material_name=='J2':
            self.material_type = 2


        # Loading
        self.loading_rate = loading_rate
        self.target_stress_xx = target_stress_xx
        self.target_stress_yy = target_stress_yy
        self.target_stress_zz = target_stress_zz


        # Post-processing
        self.saved_stress = wp.array(shape=1, dtype=wp.mat33d)


        # initialize stress and strain
        wp.launch(kernel=initialize_elastic_cto,
                  dim=1,
                  inputs=[self.rows, self.cols, self.vals, self.lame_lambda, self.lame_mu])
        wps.bsr_set_from_triplets(self.elastic_cto, self.rows, self.cols, self.vals)

        wp.launch(kernel=set_initial_stress,
                  dim=1,
                  inputs=[self.rhs, self.target_stress_xx, self.target_stress_yy, self.target_stress_zz])

        # Solve for initial strain
        preconditioner = wp.optim.linear.preconditioner(self.elastic_cto, ptype='diag')
        solver_state = wp.optim.linear.bicgstab(A=self.elastic_cto, b=self.rhs, x=self.strain_increment, tol=self.tol, M=preconditioner)
        # From increment to solution
        wp.launch(kernel=from_increment_to_solution,
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

            tape = wp.Tape()
            with tape:
                # calculate stress residual
                wp.launch(kernel=calculate_stress_residual_J2,
                          dim=1,
                          inputs=[self.new_strain_vector, self.lame_lambda, self.lame_mu, self.plasticity_dict['kappa'], self.target_stress_xx, self.target_stress_yy, self.rhs, self.saved_stress])


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


            if self.solver_name=='Warp':
                # Create sparse matrix from a corresponding COOrdinate (a.k.a. triplet) format
                wps.bsr_set_from_triplets(self.bsr_matrix, self.rows, self.cols, self.vals)

                # Warp solve
                preconditioner = wp.optim.linear.preconditioner(self.bsr_matrix, ptype='diag')
                solver_state = wp.optim.linear.bicgstab(A=self.bsr_matrix, b=self.rhs, x=self.strain_increment, tol=1e-10, M=preconditioner)
            elif self.solver_name=='pyamg':
                bsr_matrix_pyamg = sparse.coo_matrix((self.vals.numpy(), (self.rows.numpy(), self.cols.numpy())), shape=(self.n_matrix_size, self.n_matrix_size)).asformat('csr')

                # Pyamg solver
                mls = smoothed_aggregation_solver(bsr_matrix_pyamg)
                b = self.rhs.numpy()
                x_pyamg = mls.solve(b, tol=self.tol, accel='bicgstab')
                self.strain_increment = wp.from_numpy(x_pyamg, dtype=wp.float64)



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
        print('stress:', saved_stress_np)

            





    def advance_one_step(self, current_step, total_step):
        self.reset_grid_quantities()

        # Initial loading at this step
        wp.launch(kernel=initial_loading_at_this_step,
                  dim=1,
                  inputs=[self.new_strain_vector, self.loading_rate])

        print('new_strain_vector:', self.new_strain_vector.numpy())

        # Newton iteration
        self.newton_iter(current_step, total_step)

        # Calculate error
        # TODO


