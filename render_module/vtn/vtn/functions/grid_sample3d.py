from torch.autograd import Function
from torch.autograd.function import once_differentiable
from .._ext import vtn_lib
from cffi import FFI
ffi = FFI()

typename_to_func_infix = {
    'torch.FloatTensor': 'VTN_Float_',
    'torch.DoubleTensor': 'VTN_Double_',
    'torch.cuda.FloatTensor': 'VTN_Cuda_',
    'torch.cuda.DoubleTensor': 'VTN_CudaDouble_',
}


def function_by_type(name_, typename):
    assert typename in typename_to_func_infix, 'GridSampler3D only support data type: %s, got: %s' % (str(list(typename_to_func_infix.keys())), typename)
    return typename_to_func_infix[typename] + name_


class GridSampler3D(Function):

    @staticmethod
    def forward(ctx, input, grid):
        assert input.dim() == 5
        assert grid.dim() == 5 and grid.size(4) == 3
        assert grid.size(0) == input.size(0)
        assert input.is_cuda == grid.is_cuda
        assert input.type() == grid.type(), 'sampler input and grid must have same DataType. Types got: %s, %s' % (input.type(), grid.type())
        ctx.save_for_backward(input, grid)
        ctx.is_cuda = input.is_cuda
        grid_sz = grid.size()
        output = input.new_zeros([grid_sz[0], input.size(1), grid_sz[1], grid_sz[2], grid_sz[3]])

        func_name = function_by_type('BilinearSampler3DChannelFirst_updateOutput', input.type())
        getattr(vtn_lib, func_name)(input, grid, output)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        assert ctx.is_cuda == grad_output.is_cuda
        input, grid = ctx.saved_tensors
        assert input.type() == grad_output.type(), 'sampler input and grad_output must have same DataType. Types got: %s, %s' % (input.type(), grad_output.type())
        grad_input = input.new(input.size())
        grad_grid = grid.new(grid.size())

        func_name = function_by_type('BilinearSampler3DChannelFirst_updateGradInput', input.type())
        getattr(vtn_lib, func_name)(input, grid, grad_input, grad_grid, grad_output)

        return grad_input, grad_grid


def grid_sample3d(input, grid):
    """
    Perform trilinear interpolation on 3D matrices
    input: batch * channel * x * y * z
    grid: batch * gridx * gridy * gridz * 3
    output: batch * channel * gridx * gridy * gridz
    The interpolation is performed on each channel independently
    """
    return GridSampler3D.apply(input, grid)
