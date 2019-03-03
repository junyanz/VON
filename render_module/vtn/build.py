import os
import sys
import torch
from torch.utils.ffi import create_extension

sources = ['vtn/src/vtn.c']
headers = ['vtn/src/vtn.h']
defines = []
with_cuda = False
this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)

extra_compile_args = list()
extra_compile_args.append('-std=c++11')
if sys.platform == 'linux':
    extra_compile_args.append('-fopenmp')   # -fopenmp not supported on MacOS
else:
    assert sys.platform == 'darwin'

extra_objects = list()
if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['vtn/src/vtn_cuda_generic.c']
    headers += ['vtn/src/vtn_cuda_generic.h']
    defines += [('WITH_CUDA', None)]
    with_cuda = True

    extra_objects = ['vtn/src/vtn_cuda_kernel_generic.cu.o']
    extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi_params = {
    'headers': headers,
    'sources': sources,
    'define_macros': defines,
    'relative_to': __file__,
    'with_cuda': with_cuda,
    'extra_objects': extra_objects,
    'include_dirs': [os.path.join(this_file, 'vtn/src')],
    'extra_compile_args': extra_compile_args,
}

ffi = create_extension(
    'vtn._ext.vtn_lib',
    package=True,
    **ffi_params
)

if __name__ == '__main__':
    ffi = create_extension(
        'vtn._ext.vtn_lib',
        package=False,
        **ffi_params)
    ffi.build()
