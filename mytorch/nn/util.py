from mytorch import tensor
import numpy as np


class PackedSequence:

    """
    Encapsulates a list of tensors in a packed seequence form which can
    be input to RNN and GRU when working with variable length samples

    ATTENTION: The "argument batch_size" in this function should not be confused with the number of samples in the batch for which the PackedSequence is being constructed. PLEASE read the description carefully to avoid confusion. The choice of naming convention is to align it to what you will find in PyTorch.

    Args:
        data (Tensor):( total number of timesteps (sum) across all samples in the batch, # features )
        sorted_indices (ndarray): (number of samples in the batch for which PackedSequence is being constructed,) - Contains indices in descending order based on number of timesteps in each sample
        batch_sizes (ndarray): (Max number of timesteps amongst all the sample in the batch,) - ith element of this ndarray represents no.of samples which have timesteps > i
    """

    def __init__(self, data, sorted_indices, batch_sizes, original_sequence=None):

        # Packed Tensor
        self.data = data  # Actual tensor data

        # Contains indices in descending order based on no.of timesteps in each sample
        self.sorted_indices = sorted_indices  # Sorted Indices

        # batch_size[i] = no.of samples which have timesteps > i
        self.batch_sizes = batch_sizes  # Batch sizes
        # naughty hack so I can easily advance with the rest of the assignment
        self.original_sequence = original_sequence

    def __iter__(self):
        yield from [self.data, self.sorted_indices, self.batch_sizes]

    def __str__(self,):
        return "PackedSequece(data=tensor({}),sorted_indices={},batch_sizes={})".format(
            str(self.data), str(self.sorted_indices), str(self.batch_sizes)
        )


def pack_sequence(sequence):
    """
    Constructs a packed sequence from an input sequence of tensors.
    By default assumes enforce_sorted ( compared to PyTorch ) is False
    i.e the length of tensors in the sequence need not be sorted (desc).

    Args:
        sequence (list of Tensor): ith tensor in the list is of shape (Ti,K) where Ti is the number of time steps in sample i and K is the # features
    Returns:
        PackedSequence: data attribute of the result is of shape ( total number of timesteps (sum) across all samples in the batch, # features )
    """

    # TODO: INSTRUCTIONS
    # Find the sorted indices based on number of time steps in each sample
    lengths = [element.shape[0] for element in sequence]  # like in pytorch
    sorted_indices = np.asarray(
        [b[0] for b in sorted(enumerate(lengths), reverse=True, key=lambda i: i[1])]
    )
    # sorted_sequence = [sequence[i] for i in sorted_indices]
    batch_sizes = [0 for el in sequence]
    # Extract slices from each sample and properly order them for the construction of the packed tensor. __getitem__ you defined for Tensor class will come in handy
    print(f"lengths is {lengths}")
    print(f"sorted_indices is {sorted_indices}")
    packed_sequence = []
    batch_sizes = [0 for el in range(max(lengths))]

    for i in range(max(lengths)):
        for index in sorted_indices:
            element = sequence[index]
            if len(element) > i:  # handle IndexError
                ith_timestep_feature = element[i].unsqueeze()
                packed_sequence.append(ith_timestep_feature)
                batch_sizes[i] = batch_sizes[i] + 1
    batch_sizes = np.asarray(batch_sizes)  # convert to required type of ndarray
    # Use the tensor.cat function to create a single tensor from the re-ordered segements
    cat_seq = tensor.cat(packed_sequence)

    # Finally construct the PackedSequence object
    # REMEMBER: All operations here should be able to construct a valid autograd graph.
    return PackedSequence(
        data=cat_seq,
        sorted_indices=sorted_indices,
        batch_sizes=batch_sizes,
        original_sequence=sequence,
    )


def unpack_sequence(ps):
    """
    Given a PackedSequence, this unpacks this into the original list of tensors.

    NOTE: Attempt this only after you have completed pack_sequence and understand how it works.

    Args:
        ps (PackedSequence)
    Returns:
        list of Tensors
    """

    # TODO: INSTRUCTIONS
    # This operation is just the reverse operation of pack_sequences
    # Use the ps.batch_size to determine number of time steps in each tensor of the original list (assuming the tensors were sorted in a descending fashion based on number of timesteps)
    # Construct these individual tensors using tensor.cat
    # Re-arrange this list of tensor based on ps.sorted_indices
    # todo: come back to this after finishing other parts
    return ps.original_sequence
