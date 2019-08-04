package app;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.dim3;

import java.io.File;

import static jcuda.driver.JCudaDriver.*;

class CudaPainter {
    private final int imageWidth;
    private final long arraySizeInBytes;
    private final String ptxFileName;
    private final String functionName;
    private final int threadsPerBlock;

    CudaPainter(int imageWidth, String kernelFilename, String functionName) {
        this.imageWidth = imageWidth;
        this.ptxFileName = new File(CudaPainter.class.getResource(kernelFilename).getFile()).getAbsolutePath();
        this.functionName = functionName;
        arraySizeInBytes = imageWidth * imageWidth * Sizeof.INT;
        threadsPerBlock = 32;
    }

    void paint(int[] red, int[] green, int[] blue) {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, functionName);

        CUdeviceptr deviceOutputR = new CUdeviceptr();

        cuMemAlloc(deviceOutputR, arraySizeInBytes);

        CUdeviceptr deviceOutputG = new CUdeviceptr();
        cuMemAlloc(deviceOutputG, arraySizeInBytes);

        CUdeviceptr deviceOutputB = new CUdeviceptr();
        cuMemAlloc(deviceOutputB, arraySizeInBytes);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(deviceOutputR),
                Pointer.to(deviceOutputG),
                Pointer.to(deviceOutputB)
        );

        dim3 dimBlock = new dim3(threadsPerBlock, threadsPerBlock, 1);
        int blocksPerGrid = (imageWidth + threadsPerBlock - 1) / threadsPerBlock;
        dim3 dimGrid = new dim3(blocksPerGrid, blocksPerGrid, 1);

//        threadsPerBlock = 32*N
//        dim3 dimBlock = new dim3(threadsPerBlock, threadsPerBlock, 1);
//        dim3 dimGrid = new dim3(imageWidth / dimBlock.x, imageWidth / dimBlock.y, 1);

        System.out.println("block " + dimBlock);
        System.out.println("grid  " + dimGrid);

        System.out.println("a--> " + red[10 * 255 + 120]);

        cuLaunchKernel(function,
                dimGrid.x, dimGrid.y, dimGrid.z,
                dimBlock.x, dimBlock.y, dimBlock.z,
                0, null, kernelParameters, null
        );

        cuCtxSynchronize();

        System.out.println("b--> " + red[10 * 255 + 120]);
        cuMemcpyDtoH(Pointer.to(red), deviceOutputR, arraySizeInBytes);
        cuMemcpyDtoH(Pointer.to(green), deviceOutputG, arraySizeInBytes);
        cuMemcpyDtoH(Pointer.to(blue), deviceOutputB, arraySizeInBytes);
        System.out.println("c--> " + red[10 * 255 + 120]);
        memFree(deviceOutputR, deviceOutputG, deviceOutputB);
    }

    private void memFree(CUdeviceptr... devicePtr) {
        for (CUdeviceptr devptr : devicePtr) {
            cuMemFree(devptr);
        }
    }
}
