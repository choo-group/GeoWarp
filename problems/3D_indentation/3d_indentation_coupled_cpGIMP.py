import sys
sys.path.append('../..')

from solvers.quasi_static_coupled_solver_3d import quasi_static_coupled_solver_3d

import warp as wp
import numpy as np

import meshio

# ============ Background grid ============
n_grid_x = 24
n_grid_y = 24
n_grid_z = 24
max_x = 6.0 # m
max_y = 6.0 # m
max_z = 6.0 # m
dx = max_x/n_grid_x # 0.25 m
dy = max_y/n_grid_y
dz = max_z/n_grid_z

start_x = dx 
end_x = start_x + 5.0 # m
start_y = dx
end_y = start_y + 5.0 # m
start_z = 0.0
end_z = start_z + 5.0 # m

domain_min_x = start_x
domain_max_x = end_x
domain_min_y = start_y
domain_max_y = end_y
domain_min_z = start_z
domain_max_z = max_z

background_grid_dict = {'n_grid_x': n_grid_x, 'n_grid_y': n_grid_y, 'n_grid_z': n_grid_z, 'max_x': max_x, 'dx': dx, 'max_y': max_y, 'dy': dy, 'max_z': max_z, 'dz': dz, 'start_x': start_x, 'end_x': end_x, 'start_y': start_y, 'end_y': end_y, 'start_z': start_z, 'end_z': end_z, 'domain_min_x': domain_min_x, 'domain_max_x': domain_max_x, 'domain_min_y': domain_min_y, 'domain_max_y': domain_max_y, 'domain_min_z': domain_min_z, 'domain_max_z': domain_max_z}

# ============ Particles ============
n_particles = 512000
PPD = 4 # particles per direction
p_vol = (dx/PPD)**3
p_rho = 2.0 # t/m^3

particles_dict = {'n_particles': n_particles, 'PPD': PPD, 'p_vol': p_vol, 'p_rho': p_rho}

# ============ Newton iteration ============
n_iter = 10
tol = 1e-9
solver_name = 'pypardiso' #'scipy' #'pyamg' #'warp'

dt = 0.1 # s

iteration_dict = {'n_iter': n_iter, 'tol': tol, 'solver_name': solver_name, 'dt': dt}

# ============ Material properties ============
youngs_modulus = 1e4 # kPa
poisson_ratio = 0.3

material_name = 'Neo-Hookean' #'Hencky elasticity'

material_dict = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, 'material_name': material_name}

# ============ Porous media properties ============
phi_initial = 0.5
mobility_constant = 1e-8

porous_media_dict = {'phi_initial': phi_initial, 'mobility_constant': mobility_constant}

# ============ Loading ============
gravity_mag = 0.0

traction_value_x = 0.
traction_value_y = 0. # kPa
traction_value_z = 0.

point_load_value_x = 0.
point_load_value_y = 0.
point_load_value_z = 0.

footing_region_min_x = start_x
footing_region_max_x = start_x + 1.0 # m
footing_region_min_y = start_y
footing_region_max_y = start_y + 1.0 # m
footing_initial_height = end_z - dz/PPD
footing_incr_disp = -0.02 # m
penalty_factor = 1000.0

load_dict = {'gravity_mag': gravity_mag, 'traction_value_x': traction_value_x, 'traction_value_y': traction_value_y, 'traction_value_z': traction_value_z, 'point_load_value_x': point_load_value_x, 'point_load_value_y': point_load_value_y, 'point_load_value_z': point_load_value_z, 'footing_region_min_x': footing_region_min_x, 'footing_region_max_x': footing_region_max_x, 'footing_region_min_y': footing_region_min_y, 'footing_region_max_y': footing_region_max_y, 'footing_initial_height': footing_initial_height, 'footing_incr_disp': footing_incr_disp, 'penalty_factor': penalty_factor}

# ============ Print problem information ============
print("=" * 72)
print("  PROBLEM DESCRIPTION")
print("    Problem name       :: 3D indentation\n")

print("  MPM SETUP")
print(f"    Nodes number       :: {n_grid_x+1} x {n_grid_y+1} x {n_grid_z+1}")
print(f"    Grid size          :: {dx} x {dy} x {dz}")
print(f"    Particles per cell :: {PPD} x {PPD} x {PPD}")
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
elif solver_name=='pypardiso':
	print("    Name               :: pypardiso direct solver\n")

print("=" * 72)





# ============ Warp kernel for boundary dofs ============
@wp.kernel
def set_boundary_dofs(boundary_flag_array: wp.array(dtype=wp.bool),
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
		boundary_flag_array[dof_x] = True

	if node_idy<=1 or (wp.float64(node_idy)+wp.float64(0.5))*dy>end_y:
		boundary_flag_array[dof_y] = True

	if node_idz<=0:
		boundary_flag_array[dof_z] = True

	# # Dry case
	# boundary_flag_array[dof_p] = True


@wp.kernel
def set_pressure_boundary(x_particles: wp.array(dtype=wp.vec3d),
						  particle_pressure_boundary_flag_array: wp.array(dtype=wp.bool),
						  end_z: wp.float64,
						  dz: wp.float64,
						  PPD: wp.float64
						  ):
	p = wp.tid()

	if x_particles[p][2]>end_z-dz/PPD:
		particle_pressure_boundary_flag_array[p] = True




# ============ Initialize the solver ============
solver = quasi_static_coupled_solver_3d(background_grid_dict=background_grid_dict,
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
		  inputs=[solver.x_particles, solver.particle_pressure_boundary_flag_array, end_z, dz, PPD])

# ============ Post-processing for the initial step ============
x_numpy = np.array(solver.x_particles.numpy())
output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'pressure': solver.particle_pressure_array.numpy()})
output_particles.write("./vtk/indentation_particles_%d.vtk" % 0)

# ============ Load steps ============
n_steps = 25
for step in range(n_steps):
	print('Load step', step)
	solver.advance_one_step(step, n_steps)

	# Moving mesh
	new_dz = dz + footing_incr_disp/20.
	new_inv_dz = 1. / new_dz
	solver.dz = new_dz
	solver.inv_dz = new_inv_dz
	dz = new_dz

	# Post-processing
	x_numpy = np.array(solver.x_particles.numpy())
	output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'pressure': solver.particle_pressure_array.numpy()})
	output_particles.write("./vtk/indentation_particles_%d.vtk" % (step+1))



