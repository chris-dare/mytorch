import numpy as np

from mytorch import tensor
from mytorch.autograd_engine import Function


class Cat(Function):
    @staticmethod
    def forward(ctx, *args):
        """
        Args:
            args (list): [*seq, dim]

        NOTE: seq (list of tensors) contains the tensors that we wish to concatenate while dim (int) is the dimension along which we want to concatenate
        """
        *seq, dim = args
        # save for backwards
        dim_tensor = tensor.Tensor(np.asarray([dim]))
        ctx.save_for_backward(*seq, dim_tensor)
        requires_grad = max([el.requires_grad for el in seq])

        c = np.concatenate([el.data for el in seq], axis=dim)
        c = tensor.Tensor(c, requires_grad=requires_grad, is_leaf=not requires_grad)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # return grad_output
        *saved_tensors, dim = ctx.saved_tensors
        dim = max(dim.data)
        # create split list
        split_list = [x.data.shape[dim] for x in saved_tensors]
        if len(split_list) > 0:
            counter = 0
            for i in range(len(split_list)):
                counter = counter + split_list[i]
                split_list[i] = counter
            # remove the last element. not needed to split
            split_list.pop()
            gradients = [
                tensor.Tensor(x, requires_grad=True, is_leaf=False)
                for x in np.split(grad_output.data, split_list, dim)
            ]
            return (*gradients, None)
        else:
            return grad_output


class Pow(Function):
    @staticmethod
    def forward(ctx, a, b: int):
        # Check that both args are tensors
        if not (type(a).__name__ == "Tensor" and type(b) == int):
            raise Exception(
                f"Expected: Tensor, int as input. Got: {type(a)}, {type(b)}"
            )

        # Save inputs to access later in backward pass.
        exponent_tensor = tensor.Tensor(np.ones(a.shape) * b)
        ctx.save_for_backward(a, exponent_tensor)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad
        c = tensor.Tensor(
            np.power(a.data, b), requires_grad=requires_grad, is_leaf=not requires_grad
        )
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors
        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = (b.data * a.data) * grad_output.data

        grad_a = tensor.Tensor(
            grad_a, requires_grad=a.requires_grad, is_leaf=not a.requires_grad
        )
        return grad_a


def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Transpose(Function):
    @staticmethod
    def forward(ctx, a):
        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor: {}".format(a.shape))
        requires_grad = a.requires_grad
        b = tensor.Tensor(
            a.data.T, requires_grad=requires_grad, is_leaf=not requires_grad
        )
        return b

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.T)


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        if not type(a).__name__ == "Tensor":
            raise Exception(
                "Arg for Reshape must be tensor: {}".format(type(a).__name__)
            )
        ctx.shape = a.shape
        requires_grad = a.requires_grad
        c = tensor.Tensor(
            a.data.reshape(shape),
            requires_grad=requires_grad,
            is_leaf=not requires_grad,
        )
        return c

    @staticmethod
    def backward(ctx, grad_output):
        return tensor.Tensor(grad_output.data.reshape(ctx.shape)), None


class Slice(Function):
    @staticmethod
    def forward(ctx, x, indices):
        """
        Args:
            x (tensor): Tensor object that we need to slice
            indices (int,list,Slice): This is the key passed to the __getitem__ function of the Tensor object when it is sliced using [ ] notation.
        """
        if type(x) is not tensor.Tensor:
            raise Exception(f"Expected tensor. Got {type(x)}")

        ctx.save_for_backward(x, tensor.Tensor(np.asarray(indices)))
        out = tensor.Tensor(
            x.data[indices], requires_grad=x.requires_grad, is_leaf=not x.requires_grad
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, indices = ctx.saved_tensors
        gradient = np.zeros(x.shape)
        index = tuple(np.reshape(indices.data, [1, -1])[0])
        gradient[index] = grad_output.data
        return tensor.Tensor(gradient)


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        if not type(a).__name__ == "Tensor":
            raise Exception("Arg for Log must be tensor: {}".format(type(a).__name__))
        ctx.save_for_backward(a)
        requires_grad = a.requires_grad
        c = tensor.Tensor(
            np.log(a.data), requires_grad=requires_grad, is_leaf=not requires_grad
        )
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]
        return tensor.Tensor(grad_output.data / a.data)


"""EXAMPLE: This represents an Op:Add node to the comp graph.

See `Tensor.__add__()` and `autograd_engine.Function.apply()`
to understand how this class is used.

Inherits from:
    Function (autograd_engine.Function)
"""


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that both args are tensors
        if not (type(a).__name__ == "Tensor" and type(b).__name__ == "Tensor"):
            raise Exception(
                "Both args must be Tensors: {}, {}".format(
                    type(a).__name__, type(b).__name__
                )
            )

        # Check that args have same shape

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create addition output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(
            a.data + b.data, requires_grad=requires_grad, is_leaf=not requires_grad
        )
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if (
            not (type(a).__name__ == "Tensor" and type(b).__name__ == "Tensor")
            # or a.data.shape != b.data.shape
        ):
            raise Exception(
                "Both args must be Tensors: {}, {}".format(
                    type(a).__name__, type(b).__name__
                )
            )

        # Check that args have same shape
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        # Create subtraction output and sets `requires_grad and `is_leaf`
        # (see appendix A for info on those params)
        requires_grad = a.requires_grad or b.requires_grad
        c = tensor.Tensor(
            a.data - b.data, requires_grad=requires_grad, is_leaf=not requires_grad
        )
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        # calculate gradient of output w.r.t. each input
        # dL/da = dout/da * dL/dout
        grad_a = np.ones(a.shape) * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = -np.ones(b.shape) * grad_output.data

        # the order of gradients returned should match the order of the arguments
        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))
        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if (
            not (type(a).__name__ == "Tensor" and type(b).__name__ == "Tensor")
            # or a.data.shape != b.data.shape
        ):
            raise Exception(
                "Both args must be Tensors: {}, {}".format(
                    type(a).__name__, type(b).__name__
                )
            )
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(
            a.data * b.data, requires_grad=requires_grad, is_leaf=not requires_grad
        )

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = b.data * grad_output.data
        # dL/db = dout/db * dL/dout
        grad_b = a.data * grad_output.data

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if (
            not (type(a).__name__ == "Tensor" and type(b).__name__ == "Tensor")
            # or a.data.shape != b.data.shape
        ):
            raise Exception(
                "Both args must be Tensors: {}, {}".format(
                    type(a).__name__, type(b).__name__
                )
            )

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(
            a.data / b.data, requires_grad=requires_grad, is_leaf=not requires_grad
        )

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors

        grad_a = grad_output.data / b.data
        # dL/db = dout/db * dL/dout
        grad_b = (-a.data * grad_output.data) / (b.data ** 2)
        # grad_b = tensor.Tensor(grad_b)

        grad_a = tensor.Tensor(unbroadcast(grad_a, a.shape))
        grad_b = tensor.Tensor(unbroadcast(grad_b, b.shape))

        return grad_a, grad_b


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Check that inputs are tensors of same shape
        if not (type(a) == tensor.Tensor and type(b) == tensor.Tensor):
            raise Exception(
                "Both args must be Tensors: {}, {}".format(
                    type(a).__name__, type(b).__name__
                )
            )
        # check that input shapes are compliant for matrix multiplication
        if a.data.shape[1] != b.data.shape[0]:
            raise Exception(f"Shape mismatch error. Got {a.shape}, {b.shape}")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a, b)

        requires_grad = a.requires_grad or b.requires_grad

        c = tensor.Tensor(
            np.matmul(a.data, b.data),
            requires_grad=requires_grad,
            is_leaf=not requires_grad,
        )

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        if not (type(a) == tensor.Tensor and type(b) == tensor.Tensor):
            raise Exception(
                "Both args must be Tensors: {}, {}".format(
                    type(a).__name__, type(b).__name__
                )
            )
        # for c = matmal(a,b), where c belongs to a computational graph...
        # the derivative of y wrt a is the dy/dc * transpose(b)
        # import pdb
        # pdb.set_trace()
        grad_a = tensor.Tensor(np.matmul(grad_output.data, b.T().data))
        # dL/db = dout/db * dL/dout
        grad_b = tensor.Tensor(np.matmul(a.T().data, grad_output.data))

        return grad_a, grad_b


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == "Tensor":
            raise Exception("Only log of tensor is supported")
        ctx.axis = axis
        ctx.shape = a.shape
        if axis is not None:
            ctx.len = a.shape[axis]
        ctx.keepdims = keepdims
        requires_grad = a.requires_grad
        c = tensor.Tensor(
            a.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=requires_grad,
            is_leaf=not requires_grad,
        )
        # print(a.shape, c.shape)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        grad_out = grad_output.data

        if (ctx.axis is not None) and (not ctx.keepdims):
            grad_out = np.expand_dims(grad_output.data, axis=ctx.axis)
        else:
            grad_out = grad_output.data.copy()

        grad = np.ones(ctx.shape) * grad_out

        assert grad.shape == ctx.shape
        # Take note that gradient tensors SHOULD NEVER have requires_grad = True.
        return tensor.Tensor(grad), None, None


# TODO: Implement more Functions below


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs is tensors
        if not (type(a) == tensor.Tensor):
            raise Exception(f"Input argument must be Tensor. Got: {type(a)}")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)

        requires_grad = a.requires_grad

        c = tensor.Tensor(
            np.exp(a.data), requires_grad=requires_grad, is_leaf=not requires_grad
        )

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]

        # dL/da = dout/da * dL/dout

        grad = (np.ones(a.shape) * np.exp(a.data)) * grad_output.data
        grad_a = tensor.Tensor(
            grad, requires_grad=a.requires_grad, is_leaf=not a.requires_grad
        )
        return grad_a


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        # Check that inputs is tensors
        if not (type(a) == tensor.Tensor):
            raise Exception(f"Input argument must be Tensor. Got: {type(a)}")
        # Save inputs to access later in backward pass.
        ctx.save_for_backward(a)

        requires_grad = a.requires_grad

        c = tensor.Tensor(
            -a.data, requires_grad=requires_grad, is_leaf=not requires_grad
        )

        return c

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx.saved_tensors[0]

        # dL/da = dout/da * dL/dout

        grad = (-np.ones(a.shape)) * grad_output.data
        grad_a = tensor.Tensor(
            grad, requires_grad=a.requires_grad, is_leaf=not a.requires_grad
        )
        return grad_a


def cross_entropy(predicted, target):
    """Calculates Cross Entropy Loss (XELoss) between logits and true labels.
    For MNIST, don't call this function directly; use nn.loss.CrossEntropyLoss instead.

    Args:
        predicted (Tensor): (batch_size, num_classes) logits
        target (Tensor): (batch_size,) true labels

    Returns:
        Tensor: the loss as a float, in a tensor of shape ()
    """
    batch_size, num_classes = predicted.shape

    # Tip: You can implement XELoss all here, without creating a new subclass of Function.
    #      However, if you'd prefer to implement a Function subclass you're free to.
    #      Just be sure that nn.loss.CrossEntropyLoss calls it properly.
    # TODO: Document me after submission
    a = tensor.Tensor(np.max(predicted.data, axis=1, keepdims=True))

    log_sum = (predicted - a).exp().sum(axis=1, keepdims=True).log()
    rhs = a + log_sum
    XE = predicted - rhs

    one_hot_targets = to_one_hot(target, num_classes)
    loss = ((one_hot_targets * XE).sum() / tensor.Tensor(batch_size)).neg()
    return loss


def to_one_hot(arr, num_classes):
    """(Freebie) Converts a tensor of classes to one-hot, useful in XELoss

    Example:
    >>> to_one_hot(Tensor(np.array([1, 2, 0, 0])), 3)
    [[0, 1, 0],
     [0, 0, 1],
     [1, 0, 0],
     [1, 0, 0]]

    Args:
        arr (Tensor): Condensed tensor of label indices
        num_classes (int): Number of possible classes in dataset
                           For instance, MNIST would have `num_classes==10`
    Returns:
        Tensor: one-hot tensor
    """
    arr = arr.data.astype(int)
    a = np.zeros((arr.shape[0], num_classes))
    a[np.arange(len(a)), arr] = 1
    return tensor.Tensor(a, requires_grad=True)


class ReLU(Function):
    @staticmethod
    def forward(ctx, z):
        # Check that both args are tensors
        if not (type(z) == tensor.Tensor):
            raise Exception(f"Input argument must be Tensor. Got: {type(z)}")

        # Save inputs to access later in backward pass.
        ctx.save_for_backward(z)

        requires_grad = z.requires_grad
        z.data = z.data * (z.data > 0)
        output = tensor.Tensor(
            z.data, requires_grad=requires_grad, is_leaf=not requires_grad
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # retrieve forward inputs that we stored
        z = ctx.saved_tensors[0]
        # calculate gradient of output w.r.t. the input
        # dL/dz = dout/dz * dL/dout
        grad_z = tensor.Tensor((z.data > 0).astype(z.data.dtype)) * grad_output
        return grad_z


class Conv1d(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride):
        """The forward/backward of a Conv1d Layer in the comp graph.

        Notes:
            - Make sure to implement the vectorized version of the pseudocode
            - See Lec 10 slides # TODO: FINISH LOCATION OF PSEUDOCODE
            - No, you won't need to implement Conv2d for this homework.

        Args:
            x (Tensor): (batch_size, in_channel, input_size) input data
            weight (Tensor): (out_channel, in_channel, kernel_size)
            bias (Tensor): (out_channel,)
            stride (int): Stride of the convolution

        Returns:
            Tensor: (batch_size, out_channel, output_size) output data
        """
        # For your convenience: ints for each size
        batch_size, in_channel, input_size = x.shape
        out_channel, _, kernel_size = weight.shape

        # TODO: Save relevant variables for backward pass

        # TODO: Get output size by finishing & calling get_conv1d_output_size()
        output_size = get_conv1d_output_size(input_size, kernel_size, stride)

        # TODO: Initialize output with correct size
        out = np.zeros((batch_size, out_channel, output_size))

        # TODO: Calculate the Conv1d output.
        # Remember that we're working with np.arrays; no new operations needed.

        for batch in range(batch_size):
            for conv_filter in range(out_channel):
                output_dim_2_cell = 0
                for step in range(0, (input_size - kernel_size + 1), stride):
                    segment = x[batch, :, step : step + kernel_size].data.flatten()
                    segment_weight = weight[conv_filter].data.flatten()
                    z = np.inner(segment_weight, segment) + bias.data[conv_filter]
                    out[batch, conv_filter, output_dim_2_cell] = z
                    output_dim_2_cell += 1

        # TODO: Put output into tensor with correct settings and return
        out = tensor.Tensor(
            out, requires_grad=x.requires_grad, is_leaf=not x.requires_grad
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Finish Conv1d backward pass. It's surprisingly similar to the forward pass.
        raise NotImplementedError("Implement functional.Conv1d.backward()!")


def get_conv1d_output_size(input_size: int, kernel_size: int, stride: int):
    """Gets the size of a Conv1d output.

    Notes:
        - This formula should NOT add to the comp graph.
        - Yes, Conv2d would use a different formula,
        - But no, you don't need to account for Conv2d here.

        - If you want, you can modify and use this function in HW2P2.
            - You could add in Conv1d/Conv2d handling, account for padding, dilation, etc.
            - In that case refer to the torch docs for the full formulas.

    Args:
        input_size (int): Size of the input to the layer
        kernel_size (int): Size of the kernel
        stride (int): Stride of the convolution

    Returns:
        int: size of the output as an int (not a Tensor or np.array)
    """

    # simple implementation without accounting for padding et. al
    output_size = ((input_size - kernel_size) / stride) + 1
    return int(output_size)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        b_data = np.divide(1.0, np.add(1.0, np.exp(-a.data)))
        ctx.out = b_data[:]
        b = tensor.Tensor(b_data, requires_grad=a.requires_grad)
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        b = ctx.out
        grad = grad_output.data * b * (1 - b)
        return tensor.Tensor(grad)


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        b = tensor.Tensor(np.tanh(a.data), requires_grad=a.requires_grad)
        ctx.out = b.data[:]
        b.is_leaf = not b.requires_grad
        return b

    @staticmethod
    def backward(ctx, grad_output):
        out = ctx.out
        grad = grad_output.data * (1 - out ** 2)
        return tensor.Tensor(grad)
