import sys
sys.path.append('../..')

from simulators.simulator import SimulatorQuasiStatic

import warp as wp
import numpy as np

import meshio




# MPM setup parameters
# Grid quantities
n_grid_x = 60 #30
n_grid_y = 60 #30
max_x = 15.0 # m
dx = max_x/n_grid_x
inv_dx = float(n_grid_x/max_x)

# Material point quantities
n_particles = 5760 #1440
l0 = 10.0
d0 = 1.0
start_x = dx
end_x = start_x + l0 # m
start_y = l0-d0
end_y = l0 # m
PPD = 6 # Particles per direction
p_vol = (dx/PPD)**2 # 2 indicates the spatial dimension
p_rho = 1.0 # t/m^3
youngs_modulus = 12000.0 # kPa
poisson_ratio = 0.2

# Material model
material_name = 'Hencky elasticity' #'Neo-Hookean'

# Solver
n_iter = 10
tol = 1e-9
solver_name = 'pyamg' # 'Warp'


# Set boundary dofs
@wp.kernel
def set_boundary_dofs(boundary_flag_array: wp.array(dtype=wp.bool),
                      n_grid_x: wp.int32,
                      n_grid_y: wp.int32,
                      n_nodes: wp.int32,
                      end_y: wp.float64,
                      d0: wp.float64,
                      dx: wp.float64):
    
    node_idx, node_idy = wp.tid()
    dof_x = node_idx + node_idy*(n_grid_x + 1)
    dof_y = dof_x + n_nodes

    if node_idx<=1:
        boundary_flag_array[dof_x] = True

    if node_idx<=1 and node_idy==wp.int((end_y-d0/wp.float64(2.)+wp.float64(0.5)*dx)/dx):
        boundary_flag_array[dof_y] = True


# Kernel for setting external force flag
@wp.kernel
def set_external_force_flag(x_particles: wp.array(dtype=wp.vec2d),
                            particle_external_flag_array: wp.array(dtype=wp.bool),
                            start_x: wp.float64,
                            start_y: wp.float64,
                            dx: wp.float64,
                            PPD: wp.float64,
                            l0: wp.float64,
                            d0: wp.float64):
    p = wp.tid()

    this_particle = x_particles[p]

    if this_particle[0]>start_x+l0-dx/PPD:
        if this_particle[1]>l0-d0/wp.float64(2.)-dx/PPD and this_particle[1]<l0-d0/wp.float64(2.)+dx/PPD:
            particle_external_flag_array[p] = True

# Kernel for setting total external force
@wp.kernel
def set_external_force(total_external_force: wp.array(dtype=wp.vec2d)):
    total_external_force[0] = wp.vec2d(wp.float64(0.0), -wp.float64(100.0)/wp.float64(2.0)) # 2 indicates 2 particles


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
                 tol=tol,
                 solver_name=solver_name,
                 gravity_load_scale=0.0
                 )

# Set boundary flag
wp.launch(kernel=set_boundary_dofs,
          dim=(n_grid_x+1, n_grid_y+1),
          inputs=[sim.dofStruct.boundary_flag_array, sim.n_grid_x, sim.n_grid_y, sim.n_nodes, end_y, d0, dx])


# Set external force flag
wp.launch(kernel=set_external_force_flag,
          dim=n_particles,
          inputs=[sim.x_particles, sim.particle_external_flag_array, start_x, start_y, dx, PPD, l0, d0])

# Set total external force
wp.launch(kernel=set_external_force,
          dim=1,
          inputs=[sim.total_external_force])


# Initial particle configuration
x_numpy = np.array(sim.x_particles.numpy())
output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': sim.particle_Cauchy_stress_array.numpy()[:,1,1], 'external_boundary_flag': sim.particle_external_flag_array.numpy().astype(float)})
output_particles.write("./vtk/2d_beam_particles_%d.vtk" % (0))

n_steps = 50
for step in range(n_steps):
    print('Load step:', step+1)
    sim.advance_one_step(step, n_steps)

	# Post-processing
    x_numpy = np.array(sim.x_particles.numpy())
    output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': sim.particle_Cauchy_stress_array.numpy()[:,1,1]})
    output_particles.write("./vtk/2d_beam_particles_%d.vtk" % (step+1))

