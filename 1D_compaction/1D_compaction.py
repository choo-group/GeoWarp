import sys
sys.path.append('..')

from mpm.simulator import SimulatorQuasiStatic

import warp as wp
import numpy as np

import meshio




# MPM setup parameters
# Grid quantities
n_grid_x = 20
n_grid_y = 20
max_x = 100.0 # m
dx = max_x/n_grid_x
inv_dx = float(n_grid_x/max_x)

# Material point quantities
n_particles = 40
start_x = dx
end_x = start_x + dx
start_y = dx
end_y = start_y + 50.0 # m
PPD = 2 # Particles per direction
p_vol = (dx/PPD)**2 # 2 indicates the spatial dimension
p_rho = 0.08 # t/m^3
youngs_modulus = 10.0 # kPa
poisson_ratio = 0.0

# Material model
material_name = 'Hencky elasticity' #'Neo-Hookean'

# Solver
n_iter = 6
tol = 1e-10



# Set boundary dofs
@wp.kernel
def set_boundary_dofs(boundary_flag_array: wp.array(dtype=wp.bool),
                      n_grid_x: wp.int32,
                      n_nodes: wp.int32):
    
    node_idx, node_idy = wp.tid()
    dof_x = node_idx + node_idy*(n_grid_x + 1)
    dof_y = dof_x + n_nodes

    # Modify Dirichlet B.C. accordingly
    if node_idx<=1 or node_idx>=2:
        boundary_flag_array[dof_x] = True

    if node_idy<=1:
        boundary_flag_array[dof_y] = True



sim = SimulatorQuasiStatic(
				 n_grid_x=n_grid_x,
                 n_grid_y=n_grid_y,
                 max_x=max_x,
                 n_iter=n_iter,
                 n_particles=n_particles,
                 start_x=start_x,
                 end_x=end_x,
                 start_y=start_y,
                 end_y=end_y,
                 PPD=PPD,
                 p_vol=p_vol,
                 p_rho=p_rho,
                 youngs_modulus=youngs_modulus,
                 poisson_ratio=poisson_ratio,
                 material_name=material_name,
                 boundary_function_warp=set_boundary_dofs,
                 tol=tol
                 )

# Specify Dirichlet boundary conditions
wp.launch(kernel=set_boundary_dofs,
          dim=(sim.n_grid_x+1, sim.n_grid_y+1),
          inputs=[sim.dofStruct.boundary_flag_array, sim.n_grid_x, sim.n_nodes])

# Initial particle configuration
x_numpy = np.array(sim.x_particles.numpy())
output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': sim.particle_Cauchy_stress_array.numpy()[:,1,1]})
output_particles.write("./vtk/1d_compaction_particles_%d.vtk" % (0))

n_steps = 40
for step in range(n_steps):
    print('Load step:', step+1)
    sim.advance_one_step(step, n_steps)

	# Post-processing
    x_numpy = np.array(sim.x_particles.numpy())
    output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': sim.particle_Cauchy_stress_array.numpy()[:,1,1]})
    output_particles.write("./vtk/1d_compaction_particles_%d.vtk" % (step+1))

