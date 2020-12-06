import numpy as np

data = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6]
batch_sizes = [4, 4, 3, 2, 2, 1, 1]
X = []
print(f"len(data): {len(data)}")
print(f"len(batch_sizes): {len(batch_sizes)}")
start = 0
end = 0
for i in range(len(batch_sizes)):
    end = start + batch_sizes[i]
    timestep_data = data[start:end]
    X.append(timestep_data)
    start = end
X
hidden_step_per_batch = [1, 2, 3, 4, 5, 6]
hidden_step_per_sample.append(hidden_state[-1].unsqueeze())
batch_size_sample_count = {x: None for x in set(batch_sizes)}


array_1 = np.random.rand(3, 2, 2)
array_2 = np.random.rand(6, 2, 2)
array_3 = np.random.rand(5, 2, 2)

array_3_2_2 = np.asarray(
    [
        [[0.4320533, 0.95237545], [0.95117146, 0.28072601]],
        [[0.286709, 0.78094501], [0.39720068, 0.19124495]],
        [[0.74266487, 0.04059769], [0.77634308, 0.44699487]],
    ]
)
array_6_2_2 = np.asarray(
    [
        [[0.26123001, 0.43621651], [0.29971344, 0.13970178]],
        [[0.15628298, 0.3657026], [0.96539916, 0.68926097]],
        [[0.58413511, 0.11802923], [0.45866091, 0.65589033]],
        [[0.54983958, 0.37413628], [0.32830311, 0.51556421]],
        [[0.803263, 0.38144133], [0.83290483, 0.59297061]],
        [[0.16416857, 0.34305098], [0.24445541, 0.66263662]],
    ]
)

array_5_2_2 = np.asarray(
    [
        [[0.02210515, 0.47564467], [0.4374821, 0.54370569]],
        [[0.90344924, 0.07239532], [0.41982878, 0.53991954]],
        [[0.39102434, 0.18958706], [0.94818771, 0.07274136]],
        [[0.28359869, 0.10362835], [0.44260859, 0.45276598]],
        [[0.08861049, 0.96498926], [0.83168994, 0.89479731]],
    ]
)

seq = [array_3_2_2, array_6_2_2, array_5_2_2]
dim = 0
concatenated = np.concatenate(seq, axis=dim)
saved_tensors = seq
split_list = [x.shape[dim] for x in saved_tensors]

if len(split_list) > 0:
    counter = 0
    for i in range(len(split_list)):
        counter = counter + split_list[i]
        split_list[i] = counter

grad_output = saved_tensors
split_tensors = np.split(grad_output, split_list, dim)

