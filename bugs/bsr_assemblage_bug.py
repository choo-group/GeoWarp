import warp as wp
import warp.sparse 








# rows_np = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5])
# rows = wp.array(rows_np, dtype=wp.int32)

# cols_np = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])
# cols = wp.array(cols_np, dtype=wp.int32)

# vals_np = np.array([-33653.84615385, -14423.07692308, -14423.07692308, 0, 0, 0, -14423.07692308, -33653.84615385, -14423.07692308, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# vals = wp.array(vals_np, dtype=wp.float64)

# elastic_cto = wps.bsr_zeros(6, 6, block_type=wp.float64)

# bsr_matrix = wps.bsr_zeros(6, 6, block_type=wp.float64)
# rhs = wp.zeros(shape=6, dtype=wp.float64)


rows = wp.zeros(4, dtype=wp.int32)
cols = wp.zeros(4, dtype=wp.int32)
vals = wp.zeros(4, dtype=wp.float64)

bsr_matrix1 = wp.sparse.bsr_zeros(2, 2, block_type=wp.float64)
bsr_matrix2 = wp.sparse.bsr_zeros(2, 2, block_type=wp.float64)


wp.sparse.bsr_set_from_triplets(bsr_matrix1, rows, cols, vals, prune_numerical_zeros=False)
print('bsr_matrix1:', bsr_matrix1)


wp.sparse.bsr_set_from_triplets(bsr_matrix2, rows, cols, vals, prune_numerical_zeros=False)
print('bsr_matrix2:', bsr_matrix2)




