import os
from setuptools import setup, find_packages


this_file = os.path.dirname(__file__)

setup(
    name="pytorch_vtn",
    version="0.1.0",
    description="Pytorch extension of 3D transformer network",
    url="https://github.mit.edu/ckzhang/pytorch-vtn.git",
    author="Chengkai Zhang",
    author_email="ckzhang@mit.edu",
    # Require cffi.
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build", "test"]),
    # Package where to put the extensions. Has to be a prefix of build.py.
    ext_package="",
    # Extensions to compile.
    cffi_modules=[
        os.path.join(this_file, "build.py:ffi")
    ],
)
