import numpy as np

arr = np.arange(34)
arr = [50, 20, 39, 45, 289, 31905, 248, 59, 2231, 98]
batch_size = 3
selected_indices = []
seq_len = 5
rem = len(arr) % seq_len
if seq_len - rem > 0:
    edge_fill = np.zeros(seq_len - rem)
    arr = np.append(arr, edge_fill)

# now reshape the arr
arr = arr.reshape(rem + 1, seq_len)
# build batches

