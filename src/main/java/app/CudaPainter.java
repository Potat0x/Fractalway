package app;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.runtime.dim3;

import java.io.File;

import static jcuda.driver.JCudaDriver.*;

class CudaPainter {
    private final int imageWidth;
    private final int imageHeight;
    private final long arraySizeInBytes;
    private final String ptxFileName;
    private final String functionName;
    private final int threadsPerBlock;
    private CUdeviceptr deviceOutputR;
    private CUdeviceptr deviceOutputG;
    private CUdeviceptr deviceOutputB;
    private CUfunction function;

    CudaPainter(int imageWidth, int imageHeight, String kernelFilename, String functionName) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.ptxFileName = new File(CudaPainter.class.getResource(kernelFilename).getFile()).getAbsolutePath();
        this.functionName = functionName;
        arraySizeInBytes = imageWidth * imageHeight * Sizeof.INT;
        threadsPerBlock = 32;
        prepareCuda();
    }

    void paint(int[] red, int[] green, int[] blue, double zoom, double posX, double posY, int maxIter) {

        Pointer kernelParameters = Pointer.to(
                Pointer.to(new double[]{-0.8}),
                Pointer.to(new double[]{0.156}),
                Pointer.to(new double[]{zoom}),
                Pointer.to(new double[]{posX}),
                Pointer.to(new double[]{posY}),
                Pointer.to(new int[]{maxIter}),
                Pointer.to(new int[]{imageWidth}),
                Pointer.to(new int[]{imageHeight}),
                Pointer.to(deviceOutputR),
                Pointer.to(deviceOutputG),
                Pointer.to(deviceOutputB)
        );

        dim3 dimBlock = new dim3(threadsPerBlock, threadsPerBlock, 1);
        int blocksPerGridX = (imageWidth + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGridY = (imageHeight + threadsPerBlock - 1) / threadsPerBlock;
        dim3 dimGrid = new dim3(blocksPerGridX, blocksPerGridY, 1);

        System.out.println("block " + dimBlock + "\ngrid  " + dimGrid);

        cuLaunchKernel(function,
                dimGrid.x, dimGrid.y, dimGrid.z,
                dimBlock.x, dimBlock.y, dimBlock.z,
                0, null, kernelParameters, null
        );

        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(red), deviceOutputR, arraySizeInBytes);
        cuMemcpyDtoH(Pointer.to(green), deviceOutputG, arraySizeInBytes);
        cuMemcpyDtoH(Pointer.to(blue), deviceOutputB, arraySizeInBytes);
    }

    private void prepareCuda() {
        JCudaDriver.setExceptionsEnabled(true);
        cuInit(0);

        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);

        function = new CUfunction();
        cuModuleGetFunction(function, module, functionName);

        deviceOutputR = new CUdeviceptr();
        cuMemAlloc(deviceOutputR, arraySizeInBytes);
        deviceOutputG = new CUdeviceptr();
        cuMemAlloc(deviceOutputG, arraySizeInBytes);
        deviceOutputB = new CUdeviceptr();
        cuMemAlloc(deviceOutputB, arraySizeInBytes);
    }

    private void memFree(CUdeviceptr... devicePtr) {
        for (CUdeviceptr devptr : devicePtr) {
            cuMemFree(devptr);
        }
    }

    @Override //todo: finalize is deprecated since Java 9
    protected void finalize() throws Throwable {
        memFree(deviceOutputR, deviceOutputG, deviceOutputB);
        super.finalize();
    }
}
