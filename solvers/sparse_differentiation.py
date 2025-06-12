import sys
sys.path.append('..')

import warp as wp
import numpy as np


def pick_grid_nodes_2d(n_grid_x, n_grid_y, first_grid_node):
	# Determine the row and column of the first grid node
	row_start = first_grid_node // (n_grid_x + 1)
	col_start = first_grid_node % (n_grid_x + 1)

	# Step size for the 5x5 grid
	step = 5

	# Create arrays for rows and columns to pick nodes efficiently
	rows = np.arange(row_start, n_grid_y + 1, step)
	cols = np.arange(col_start, n_grid_x + 1, step)

	# Generate a grid of row and column indices
	row_indices, col_indices = np.meshgrid(rows, cols, indexing='ij')

	# Compute the node indices
	picked_nodes = row_indices * (n_grid_x + 1) + col_indices

	# Flatten the array and filter out indices exceeding the total number of nodes
	picked_nodes = picked_nodes.flatten()
	
	return picked_nodes

def pick_grid_nodes_3d(n_grid_x, n_grid_y, n_grid_z, first_grid_node):
	# Determine the row, column, and layer of the first grid node
	layer_start = first_grid_node // ((n_grid_x + 1) * (n_grid_y + 1))
	row_start = (first_grid_node % ((n_grid_x + 1) * (n_grid_y + 1))) // (n_grid_x + 1)
	col_start = first_grid_node % (n_grid_x + 1)

	# Step size for the 3x3x3 grid
	step = 5

	# Create arrays for layers, rows, and columns to pick nodes efficiently
	layers = np.arange(layer_start, n_grid_z + 1, step)
	rows = np.arange(row_start, n_grid_y + 1, step)
	cols = np.arange(col_start, n_grid_x + 1, step)

	# Generate a grid of layer, row, and column indices
	layer_indices, row_indices, col_indices = np.meshgrid(layers, rows, cols, indexing='ij')

	# Compute the node indices
	picked_nodes = (layer_indices * (n_grid_x + 1) * (n_grid_y + 1) +
					row_indices * (n_grid_x + 1) +
					col_indices)

	# Flatten the array and filter out indices exceeding the total number of nodes
	picked_nodes = picked_nodes.flatten()

	return picked_nodes


def precompute_seed_vectors_2d(n_grid_x, n_grid_y, n_nodes, n_matrix_size, max_selector_length):
	selector_x_list = []
	selector_y_list = []
	e_x_list = []
	e_y_list = []
	for c_iter in range(5):
		for r_iter in range(5):
			current_node_id = c_iter + r_iter * (n_grid_x+1) 
			select_index = np.zeros(n_matrix_size)

			# x
			selector_x = pick_grid_nodes_2d(n_grid_x, n_grid_y, current_node_id)
			selector_x_resize = selector_x + 0
			selector_x_resize.resize(max_selector_length)
			selector_x_wp = wp.from_numpy(selector_x_resize, dtype=wp.int32)
			selector_x_list.append(selector_x_wp)

			select_index[selector_x] = 1.
			e = wp.array(select_index, dtype=wp.float64)
			e_x_list.append(e)

			# y
			selector_y = selector_x + n_nodes
			selector_y_resize = selector_y + 0
			selector_y_resize.resize(max_selector_length)
			selector_y_wp = wp.from_numpy(selector_y_resize, dtype=wp.int32)
			selector_y_list.append(selector_y_wp)

			select_index = np.zeros(n_matrix_size)
			select_index[selector_y] = 1.
			e = wp.array(select_index, dtype=wp.float64)
			e_y_list.append(e)


	return selector_x_list, selector_y_list, e_x_list, e_y_list


def precompute_seed_vectors_coupled_2d(n_grid_x, n_grid_y, n_nodes, n_matrix_size, max_selector_length):
	selector_x_list = []
	selector_y_list = []
	selector_p_list = []
	e_x_list = []
	e_y_list = []
	e_p_list = []
	for c_iter in range(5):
		for r_iter in range(5):
			current_node_id = c_iter + r_iter * (n_grid_x+1) 
			select_index = np.zeros(n_matrix_size)

			# x
			selector_x = pick_grid_nodes_2d(n_grid_x, n_grid_y, current_node_id)
			selector_x_resize = selector_x + 0
			selector_x_resize.resize(max_selector_length)
			selector_x_wp = wp.from_numpy(selector_x_resize, dtype=wp.int32)
			selector_x_list.append(selector_x_wp)

			select_index[selector_x] = 1.
			e = wp.array(select_index, dtype=wp.float64)
			e_x_list.append(e)

			# y
			selector_y = selector_x + n_nodes
			selector_y_resize = selector_y + 0
			selector_y_resize.resize(max_selector_length)
			selector_y_wp = wp.from_numpy(selector_y_resize, dtype=wp.int32)
			selector_y_list.append(selector_y_wp)

			select_index = np.zeros(n_matrix_size)
			select_index[selector_y] = 1.
			e = wp.array(select_index, dtype=wp.float64)
			e_y_list.append(e)

			# p
			selector_p = selector_x + 2*n_nodes
			selector_p_resize = selector_p + 0
			selector_p_resize.resize(max_selector_length)
			selector_p_wp = wp.from_numpy(selector_p_resize, dtype=wp.int32)
			selector_p_list.append(selector_p_wp)

			select_index = np.zeros(n_matrix_size)
			select_index[selector_p] = 1.
			e = wp.array(select_index, dtype=wp.float64)
			e_p_list.append(e)


	return selector_x_list, selector_y_list, selector_p_list, e_x_list, e_y_list, e_p_list


def precompute_seed_vectors_coupled_3d(n_grid_x, n_grid_y, n_grid_z, n_nodes, n_matrix_size, max_selector_length):
	selector_x_list = []
	selector_y_list = []
	selector_z_list = []
	selector_p_list = []
	e_x_list = []
	e_y_list = []
	e_z_list = []
	e_p_list = []
	for c_iter in range(5):
		for r_iter in range(5):
			for l_iter in range(5):
				current_node_id = c_iter + r_iter * (n_grid_x+1) + l_iter*((n_grid_x+1)*(n_grid_y+1))
				select_index = np.zeros(n_matrix_size)

				# x
				selector_x = pick_grid_nodes_3d(n_grid_x, n_grid_y, n_grid_z, current_node_id)
				selector_x_resize = selector_x + 0
				selector_x_resize.resize(max_selector_length)
				selector_x_wp = wp.from_numpy(selector_x_resize, dtype=wp.int32)
				selector_x_list.append(selector_x_wp)

				select_index[selector_x] = 1.
				e = wp.array(select_index, dtype=wp.float64)
				e_x_list.append(e)

				# y
				selector_y = selector_x + n_nodes
				selector_y_resize = selector_y + 0
				selector_y_resize.resize(max_selector_length)
				selector_y_wp = wp.from_numpy(selector_y_resize, dtype=wp.int32)
				selector_y_list.append(selector_y_wp)

				select_index = np.zeros(n_matrix_size)
				select_index[selector_y] = 1.
				e = wp.array(select_index, dtype=wp.float64)
				e_y_list.append(e)

				# z
				selector_z = selector_x + 2*n_nodes
				selector_z_resize = selector_z + 0
				selector_z_resize.resize(max_selector_length)
				selector_z_wp = wp.from_numpy(selector_z_resize, dtype=wp.int32)
				selector_z_list.append(selector_z_wp)

				select_index = np.zeros(n_matrix_size)
				select_index[selector_z] = 1.
				e = wp.array(select_index, dtype=wp.float64)
				e_z_list.append(e)

				# p
				selector_p = selector_x + 3*n_nodes
				selector_p_resize = selector_p + 0
				selector_p_resize.resize(max_selector_length)
				selector_p_wp = wp.from_numpy(selector_p_resize, dtype=wp.int32)
				selector_p_list.append(selector_p_wp)

				select_index = np.zeros(n_matrix_size)
				select_index[selector_p] = 1.
				e = wp.array(select_index, dtype=wp.float64)
				e_p_list.append(e)


	return selector_x_list, selector_y_list, selector_z_list, selector_p_list, e_x_list, e_y_list, e_z_list, e_p_list


@wp.kernel
def from_jacobian_to_vector_parallel_2d(jacobian_wp: wp.array(dtype=wp.float64),
										rows: wp.array(dtype=wp.int32),
										cols: wp.array(dtype=wp.int32),
										vals: wp.array(dtype=wp.float64),
										n_grid_x: wp.int32,
										n_grid_y: wp.int32,
										n_nodes: wp.int32,
										n_matrix_size: wp.int32,
										selector_wp: wp.array(dtype=wp.int32),
										boundary_flag_array: wp.array(dtype=wp.bool),
										activate_flag_array: wp.array(dtype=wp.bool)
										):
	
	selector_index = wp.tid()

	row_index = selector_wp[selector_index]

	if row_index>0 or (row_index==0 and selector_index==0):
		# from dof to node_id
		node_idx = wp.int(0)
		node_idy = wp.int(0)

		if row_index<n_nodes: # x-dof
			node_idx = wp.mod(row_index, n_grid_x+1)
			node_idy = wp.int((row_index-node_idx)/(n_grid_x+1)) 
		else:
			node_idx = wp.mod((row_index-n_nodes), n_grid_x+1)
			node_idy = wp.int((row_index-n_nodes)/(n_grid_x+1))


		for i in range(5):
			adj_node_idx = node_idx + (i-2)
			for j in range(5):
				adj_node_idy = node_idy + (j-2)
				
				adj_index_x = adj_node_idx + adj_node_idy*(n_grid_x+1)
				adj_index_y = adj_index_x + n_nodes

				if adj_node_idx>=0 and adj_node_idx<=n_grid_x and adj_node_idy>=0 and adj_node_idy<=n_grid_y: # if adj_node is reasonable
					if boundary_flag_array[row_index]==False and activate_flag_array[row_index]==True:
						if boundary_flag_array[adj_index_x]==False and activate_flag_array[adj_index_x]==True:
							rows[row_index*25 + (i+j*5)] = row_index
							cols[row_index*25 + (i+j*5)] = adj_index_x
							vals[row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_x]

						if boundary_flag_array[adj_index_y]==False and activate_flag_array[adj_index_y]==True:
							rows[25*n_matrix_size + row_index*25 + (i+j*5)] = row_index
							cols[25*n_matrix_size + row_index*25 + (i+j*5)] = adj_index_y
							vals[25*n_matrix_size + row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_y]


@wp.kernel
def from_jacobian_to_vector_parallel_coupled_2d(jacobian_wp: wp.array(dtype=wp.float64),
												rows: wp.array(dtype=wp.int32),
												cols: wp.array(dtype=wp.int32),
												vals: wp.array(dtype=wp.float64),
												n_grid_x: wp.int32,
												n_grid_y: wp.int32,
												n_nodes: wp.int32,
												n_matrix_size: wp.int32,
												selector_wp: wp.array(dtype=wp.int32),
												boundary_flag_array: wp.array(dtype=wp.bool),
												activate_flag_array: wp.array(dtype=wp.bool)
												):
	
	selector_index = wp.tid()

	row_index = selector_wp[selector_index]

	if row_index>0 or (row_index==0 and selector_index==0):
		# from dof to node_id
		node_idx = wp.int(0)
		node_idy = wp.int(0)

		if row_index<n_nodes: # x-dof
			node_idx = wp.mod(row_index, n_grid_x+1)
			node_idy = wp.int((row_index-node_idx)/(n_grid_x+1)) 
		elif row_index>=n_nodes and row_index<2*n_nodes: # y-dof
			node_idx = wp.mod((row_index-n_nodes), n_grid_x+1)
			node_idy = wp.int((row_index-n_nodes)/(n_grid_x+1))
		else:
			node_idx = wp.mod((row_index-2*n_nodes), n_grid_x+1)
			node_idy = wp.int((row_index-2*n_nodes)/(n_grid_x+1))


		for i in range(5):
			adj_node_idx = node_idx + (i-2)
			for j in range(5):
				adj_node_idy = node_idy + (j-2)
				
				adj_index_x = adj_node_idx + adj_node_idy*(n_grid_x+1)
				adj_index_y = adj_index_x + n_nodes
				adj_index_p = adj_index_x + 2*n_nodes

				if adj_node_idx>=0 and adj_node_idx<=n_grid_x and adj_node_idy>=0 and adj_node_idy<=n_grid_y: # if adj_node is reasonable
					if boundary_flag_array[row_index]==False and activate_flag_array[row_index]==True:
						if boundary_flag_array[adj_index_x]==False and activate_flag_array[adj_index_x]==True:
							rows[row_index*25 + (i+j*5)] = row_index
							cols[row_index*25 + (i+j*5)] = adj_index_x
							vals[row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_x]

						if boundary_flag_array[adj_index_y]==False and activate_flag_array[adj_index_y]==True:
							rows[25*n_matrix_size + row_index*25 + (i+j*5)] = row_index
							cols[25*n_matrix_size + row_index*25 + (i+j*5)] = adj_index_y
							vals[25*n_matrix_size + row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_y]

						if boundary_flag_array[adj_index_p]==False and activate_flag_array[adj_index_p]==True:
							rows[50*n_matrix_size + row_index*25 + (i+j*5)] = row_index
							cols[50*n_matrix_size + row_index*25 + (i+j*5)] = adj_index_p
							vals[50*n_matrix_size + row_index*25 + (i+j*5)] = -jacobian_wp[adj_index_p]

@wp.kernel
def from_jacobian_to_vector_parallel_coupled_3d(jacobian_wp: wp.array(dtype=wp.float64),
												rows: wp.array(dtype=wp.int32),
												cols: wp.array(dtype=wp.int32),
												vals: wp.array(dtype=wp.float64),
												n_grid_x: wp.int32,
												n_grid_y: wp.int32,
												n_grid_z: wp.int32,
												n_nodes: wp.int32,
												n_matrix_size: wp.int32,
												selector_wp: wp.array(dtype=wp.int32),
												boundary_flag_array: wp.array(dtype=wp.bool),
												activate_flag_array: wp.array(dtype=wp.bool)
												):
	selector_index = wp.tid()

	row_index = selector_wp[selector_index]

	if row_index>0 or (row_index==0 and selector_index==0):
		# from dof to node_id
		node_idx = wp.int(0)
		node_idy = wp.int(0)
		node_idz = wp.int(0)

		if row_index<n_nodes: # x-dof
			node_idx = wp.mod(row_index, n_grid_x+1)
			node_idy = wp.int(wp.mod(row_index, ((n_grid_x+1)*(n_grid_y+1))) / (n_grid_x+1)) 
			node_idz = wp.int(row_index / ((n_grid_x+1)*(n_grid_y+1)))
		elif row_index>=n_nodes and row_index<2*n_nodes: # y-dof
			node_idx = wp.mod(row_index-n_nodes, n_grid_x+1)
			node_idy = wp.int(wp.mod(row_index-n_nodes, ((n_grid_x+1)*(n_grid_y+1))) / (n_grid_x+1)) 
			node_idz = wp.int((row_index-n_nodes) / ((n_grid_x+1)*(n_grid_y+1)))
		elif row_index>=2*n_nodes and row_index<3*n_nodes: # z-dof
			node_idx = wp.mod(row_index-2*n_nodes, n_grid_x+1)
			node_idy = wp.int(wp.mod(row_index-2*n_nodes, ((n_grid_x+1)*(n_grid_y+1))) / (n_grid_x+1)) 
			node_idz = wp.int((row_index-2*n_nodes) / ((n_grid_x+1)*(n_grid_y+1)))
		else: # pressure dof
			node_idx = wp.mod(row_index-3*n_nodes, n_grid_x+1)
			node_idy = wp.int(wp.mod(row_index-3*n_nodes, ((n_grid_x+1)*(n_grid_y+1))) / (n_grid_x+1)) 
			node_idz = wp.int((row_index-3*n_nodes) / ((n_grid_x+1)*(n_grid_y+1)))

		# Flattened loop
		for flattened_id in range(125):
			i = wp.int(flattened_id/25)
			j = wp.mod(wp.int(flattened_id/5), 5)
			k = wp.mod(flattened_id, 5)

			adj_node_idx = node_idx + (i-2)
			adj_node_idy = node_idy + (j-2)
			adj_node_idz = node_idz + (k-2)

			adj_index_x = adj_node_idx + adj_node_idy*(n_grid_x+1) + adj_node_idz*((n_grid_x+1)*(n_grid_y+1))
			adj_index_y = adj_index_x + n_nodes
			adj_index_z = adj_index_x + 2*n_nodes
			adj_index_p = adj_index_x + 3*n_nodes

			if adj_node_idx>=0 and adj_node_idx<=n_grid_x and adj_node_idy>=0 and adj_node_idy<=n_grid_y and adj_node_idz>=0 and adj_node_idz<=n_grid_z: # adj_node is reasonable
				if boundary_flag_array[row_index]==False and activate_flag_array[row_index]==True:
					if boundary_flag_array[adj_index_x]==False and activate_flag_array[adj_index_x]==True and wp.isnan(jacobian_wp[adj_index_x])==False:
						rows[row_index*125 + (i+j*5+k*25)] = row_index
						cols[row_index*125 + (i+j*5+k*25)] = adj_index_x
						vals[row_index*125 + (i+j*5+k*25)] = -jacobian_wp[adj_index_x]

					if boundary_flag_array[adj_index_y]==False and activate_flag_array[adj_index_y]==True and wp.isnan(jacobian_wp[adj_index_y])==False:
						rows[125*n_matrix_size + row_index*125 + (i+j*5+k*25)] = row_index
						cols[125*n_matrix_size + row_index*125 + (i+j*5+k*25)] = adj_index_y
						vals[125*n_matrix_size + row_index*125 + (i+j*5+k*25)] = -jacobian_wp[adj_index_y]

					if boundary_flag_array[adj_index_z]==False and activate_flag_array[adj_index_z]==True and wp.isnan(jacobian_wp[adj_index_z])==False:
						rows[125*2*n_matrix_size + row_index*125 + (i+j*5+k*25)] = row_index
						cols[125*2*n_matrix_size + row_index*125 + (i+j*5+k*25)] = adj_index_z
						vals[125*2*n_matrix_size + row_index*125 + (i+j*5+k*25)] = -jacobian_wp[adj_index_z]

					if boundary_flag_array[adj_index_p]==False and activate_flag_array[adj_index_p]==True and wp.isnan(jacobian_wp[adj_index_p])==False:
						rows[125*3*n_matrix_size + row_index*125 + (i+j*5+k*25)] = row_index
						cols[125*3*n_matrix_size + row_index*125 + (i+j*5+k*25)] = adj_index_p
						vals[125*3*n_matrix_size + row_index*125 + (i+j*5+k*25)] = -jacobian_wp[adj_index_p]



