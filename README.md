# cuda_toys
This repo is an accumulation of some fun HPC toys with CUDA/Thrust and OpenMP.

Each project in examples has its own README.md for specific instructions and notes about the project.

## Build
These projects use CMake. There's a top level ``CMakeLists.txt``, and each example project also has a CMakeLists.txt that is used to build each individually. 
```bash
mkdir build
cd build
cmake ../
make
```

## Running
Programs are built in ``build/examples/projectname``. For example, to run ``thrust_median``, you would do:
```bash
.build/examples/thrust_median/thrust_median
`````

