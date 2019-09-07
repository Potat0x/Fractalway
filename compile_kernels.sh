cd src/main/resources/kernels/
nvcc -m64 -ptx -Xcompiler=-Wall mandelbrotSet.cu -o mandelbrotSet.ptx
nvcc -m64 -ptx -Xcompiler=-Wall juliaSet.cu -o juliaSet.ptx
nvcc -m64 -ptx -Xcompiler=-Wall burningShip.cu -o burningShip.ptx
