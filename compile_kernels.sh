cd src/main/resources/kernels/
nvcc -m64 -ptx -Xcompiler=-Wall gradient.cu -o gradient.ptx
nvcc -m64 -ptx -Xcompiler=-Wall mandelbrotSet.cu -o mandelbrotSet.ptx
nvcc -m64 -ptx -Xcompiler=-Wall juliaSet.cu -o juliaSet.ptx