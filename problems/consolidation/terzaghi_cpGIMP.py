import sys
sys.path.append('../..')

from solvers.quasi_static_coupled_solver_2d import quasi_static_coupled_solver_2d

import warp as wp
import numpy as np

import meshio


# ============ Background grid ============
n_grid_x = 6
n_grid_y = 120
max_x = 0.6 # m
max_y = 12.0 # m
dx = max_x/n_grid_x # 0.1 m
dy = max_y/n_grid_y

start_x = dx 
end_x = start_x + dx
start_y = 0.0
end_y = start_y + 10.0 # m

domain_min_x = start_x
domain_max_x = end_x
domain_min_y = start_y
domain_max_y = max_y

background_grid_dict = {'n_grid_x': n_grid_x, 'n_grid_y': n_grid_y, 'max_x': max_x, 'dx': dx, 'max_y': max_y, 'dy': dy, 'start_x': start_x, 'end_x': end_x, 'start_y': start_y, 'end_y': end_y, 'domain_min_x': domain_min_x, 'domain_max_x': domain_max_x, 'domain_min_y': domain_min_y, 'domain_max_y': domain_max_y}

# ============ Particles ============
n_particles = 400
PPD = 2 # particles per direction
p_vol = (dx/PPD)**2 
p_rho = 1.0 # t/m^3

particles_dict = {'n_particles': n_particles, 'PPD': PPD, 'p_vol': p_vol, 'p_rho': p_rho}

# ============ Newton iteration ============
n_iter = 10
tol = 1e-8
solver_name = 'scipy' #'pyamg' #'warp'

dt = 100.0 # s

iteration_dict = {'n_iter': n_iter, 'tol': tol, 'solver_name': solver_name, 'dt': dt}

# ============ Material properties ============
youngs_modulus = 1500.0 # kPa
poisson_ratio = 0.25

material_name = 'Neo-Hookean' #'Hencky elasticity'

material_dict = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, 'material_name': material_name}

# ============ Porous media properties ============
phi_initial = 0.5
mobility_constant = 1e-6

porous_media_dict = {'phi_initial': phi_initial, 'mobility_constant': mobility_constant}

# ============ Loading ============
gravity_mag = 0.0

traction_value_x = 0.
traction_value_y = -1.0 # kPa

point_load_value_x = 0.
point_load_value_y = 0.

load_dict = {'gravity_mag': gravity_mag, 'traction_value_x': traction_value_x, 'traction_value_y': traction_value_y, 'point_load_value_x': point_load_value_x, 'point_load_value_y': point_load_value_y}


# ============ Print problem information ============
print("=" * 72)
print("  PROBLEM DESCRIPTION")
print("    Problem name       :: Terzaghi consolidation\n")

print("  MPM SETUP")
print(f"    Nodes number       :: {n_grid_x+1} x {n_grid_y+1}")
print(f"    Grid size          :: {dx} x {dx}")
print(f"    Particles per cell :: {PPD} x {PPD}")
print(f"    Total particles    :: {n_particles}\n")

print("  MATERIAL MODEL")
print(f"    Name               :: {material_name}\n")

print("  MATRIX SOLVER")
if solver_name=='warp':
	print("    Name               :: Warp iterative solver\n")
elif solver_name=='pyamg':
	print("    Name               :: pyamg iterative solver\n")
elif solver_name=='scipy':
	print("    Name               :: scipy direct solver\n")

print("=" * 72)





# ============ Warp kernel for boundary dofs ============
@wp.kernel
def set_boundary_dofs(boundary_flag_array: wp.array(dtype=wp.bool),
					  n_grid_x: wp.int32,
					  n_nodes: wp.int32,
					  ):
	
	node_idx, node_idy = wp.tid()
	dof_x = node_idx + node_idy*(n_grid_x + 1)
	dof_y = dof_x + n_nodes
	dof_p = dof_x + 2*n_nodes

	if node_idx<=1 or node_idx>=2:
		boundary_flag_array[dof_x] = True

	if node_idy<=0:
		boundary_flag_array[dof_y] = True

	# # Dry case
	# boundary_flag_array[dof_p] = True




@wp.kernel
def set_traction_boundary(x_particles: wp.array(dtype=wp.vec2d),
						  particle_traction_flag_array: wp.array(dtype=wp.bool),
						  end_y: wp.float64,
						  dx: wp.float64,
						  PPD: wp.float64
						  ):
	p = wp.tid()

	if x_particles[p][1]>end_y-dx/PPD:
		particle_traction_flag_array[p] = True

@wp.kernel
def set_pressure_boundary(x_particles: wp.array(dtype=wp.vec2d),
						  particle_pressure_boundary_flag_array: wp.array(dtype=wp.bool),
						  end_y: wp.float64,
						  dx: wp.float64,
						  PPD: wp.float64
						  ):
	p = wp.tid()

	if x_particles[p][1]>end_y-dx/PPD:
		particle_pressure_boundary_flag_array[p] = True




# ============ Initialize the solver ============
solver = quasi_static_coupled_solver_2d(background_grid_dict=background_grid_dict,
										particles_dict=particles_dict,
										iteration_dict=iteration_dict,
										material_dict=material_dict,
										porous_media_dict=porous_media_dict,
										load_dict=load_dict,
										set_boundary_dofs=set_boundary_dofs
										)


# ============ Specify Dirichlet boundary conditions ============
wp.launch(kernel=set_pressure_boundary,
		  dim=solver.n_particles,
		  inputs=[solver.x_particles, solver.particle_pressure_boundary_flag_array, end_y, dx, PPD])

# ============ Specify traction flag ============
wp.launch(kernel=set_traction_boundary,
		  dim=solver.n_particles,
		  inputs=[solver.x_particles, solver.particle_traction_flag_array, end_y, dx, PPD])

# ============ Post-processing for the initial step ============
x_numpy = np.array(solver.x_particles.numpy())
output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_xx': solver.particle_Cauchy_stress_array.numpy()[:,0,0], 'stress_yy': solver.particle_Cauchy_stress_array.numpy()[:,1,1], 'pressure': solver.particle_pressure_array.numpy()})
output_particles.write("./vtk/terzaghi_particles_%d.vtk" % 0)

# ============ Load steps ============
n_steps = 600
for step in range(n_steps):
	print('Load step', step)
	solver.advance_one_step(step, n_steps)

	# Post-processing
	x_numpy = np.array(solver.x_particles.numpy())
	output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_xx': solver.particle_Cauchy_stress_array.numpy()[:,0,0], 'stress_yy': solver.particle_Cauchy_stress_array.numpy()[:,1,1], 'pressure': solver.particle_pressure_array.numpy()})
	output_particles.write("./vtk/terzaghi_particles_%d.vtk" % (step+1))



