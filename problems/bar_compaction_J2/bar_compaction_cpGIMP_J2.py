import sys
sys.path.append('../..')

from solvers.quasi_static_solver_2d import quasi_static_solver_2d

import warp as wp
import numpy as np

import meshio


# ============ Background grid ============
n_grid_x = 6 
n_grid_y = 66 #6 #10 #18 #34 # #130 #258 #514 
max_x = 4.6875 #75. #37.5 #18.75 #9.375 # #2.34375 #1.171875 #0.5859375 # m
dx = max_x/n_grid_x 

start_x = dx 
end_x = start_x + dx
start_y = dx
end_y = start_y + 50. # m

domain_min_x = start_x
domain_max_x = end_x
domain_min_y = start_y
domain_max_y = end_y

background_grid_dict = {'n_grid_x': n_grid_x, 'n_grid_y': n_grid_y, 'max_x': max_x, 'dx': dx, 'start_x': start_x, 'end_x': end_x, 'start_y': start_y, 'end_y': end_y, 'domain_min_x': domain_min_x, 'domain_max_x': domain_max_x, 'domain_min_y': domain_min_y, 'domain_max_y': domain_max_y}

# ============ Particles ============
n_particles = 256 #16 #32 #64 #128 # #512 #1024 #2048 
PPD = 2 # particles per direction
p_vol = (dx/PPD)**2 
p_rho = 0.08 # t/m^3

particles_dict = {'n_particles': n_particles, 'PPD': PPD, 'p_vol': p_vol, 'p_rho': p_rho}

# ============ Newton iteration ============
n_iter = 10
tol = 1e-8
solver_name = 'scipy' #'pyamg' #'warp'

iteration_dict = {'n_iter': n_iter, 'tol': tol, 'solver_name': solver_name}

# ============ Material properties ============
youngs_modulus = 10 # kPa
poisson_ratio = 0.0

kappa = 5.0 # kPa

material_name = 'J2'

material_dict = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, 'kappa': kappa, 'material_name': material_name}

# ============ Loading ============
gravity_mag = 10.0

traction_value_x = 0.
traction_value_y = 0.

point_load_value_x = 0.
point_load_value_y = 0.

load_dict = {'gravity_mag': gravity_mag, 'traction_value_x': traction_value_x, 'traction_value_y': traction_value_y, 'point_load_value_x': point_load_value_x, 'point_load_value_y': point_load_value_y}


# ============ Print problem information ============
print("=" * 72)
print("  PROBLEM DESCRIPTION")
print("    Problem name       :: Bar compaction\n")

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
					  n_nodes: wp.int32
					  ):
	
	node_idx, node_idy = wp.tid()
	dof_x = node_idx + node_idy*(n_grid_x + 1)
	dof_y = dof_x + n_nodes

	if node_idx<=1 or node_idx>=2:
		boundary_flag_array[dof_x] = True

	if node_idy<=1:
		boundary_flag_array[dof_y] = True


# ============ Warp kernel for calculating error ============
@wp.kernel
def calculate_error(error_numerator: wp.array(dtype=wp.float64),
					error_denominator: wp.array(dtype=wp.float64),
					particle_Cauchy_stress_array: wp.array(dtype=wp.mat33d),
					x0: wp.array(dtype=wp.vec2d),
					p_rho: wp.float64,
					p_vol: wp.float64,
					start_y: wp.float64,
					n_particles: wp.float64
					):
	p = wp.tid()

	analytical_stress_yy = p_rho * wp.float64(-10.0) * (wp.float64(50.0) - (x0[p][1]-start_y))
	particle_stress_yy = particle_Cauchy_stress_array[p][1,1]

 
	wp.atomic_add(error_numerator, 0, wp.abs(particle_stress_yy-analytical_stress_yy) * p_vol)
	wp.atomic_add(error_denominator, 0, wp.float64(10.0) * p_rho * wp.float64(50.0) * p_vol)

# ============ Initialize the solver ============
solver = quasi_static_solver_2d(background_grid_dict=background_grid_dict,
								particles_dict=particles_dict,
								iteration_dict=iteration_dict,
								material_dict=material_dict,
								load_dict=load_dict
							   )

# ============ Specify Dirichlet boundary conditions ============
wp.launch(kernel=set_boundary_dofs,
		  dim=(solver.n_grid_x+1, solver.n_grid_y+1),
		  inputs=[solver.dofStruct.boundary_flag_array, solver.n_grid_x, solver.n_nodes])

# ============ Post-processing for the initial step ============
x_numpy = np.array(solver.x_particles.numpy())
output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': solver.particle_Cauchy_stress_array.numpy()[:,1,1]})
output_particles.write("./vtk/bar_compaction_particles_%d.vtk" % 0)

# ============ Load steps ============
n_steps = 40
for step in range(n_steps):
	print('Load step', step)
	solver.advance_one_step(step, n_steps)

	# Post-processing
	x_numpy = np.array(solver.x_particles.numpy())
	output_particles = meshio.Mesh(points=x_numpy, cells=[], point_data={'stress_yy': solver.particle_Cauchy_stress_array.numpy()[:,1,1]})
	output_particles.write("./vtk/bar_compaction_particles_%d.vtk" % (step+1))

