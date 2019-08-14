package pl.potat0x.fractalway;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.dim3;
import pl.potat0x.fractalway.fractal.Fractal;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
    private CUmodule module;
    private CUcontext context;

    CudaPainter(int imageWidth, int imageHeight, String kernelFilename, String functionName) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.ptxFileName = new File(CudaPainter.class.getResource(kernelFilename).getFile()).getAbsolutePath();
        this.functionName = functionName;
        arraySizeInBytes = imageWidth * imageHeight * Sizeof.INT;
        threadsPerBlock = 32;
        prepareCuda();
    }

    void paint(int[] red, int[] green, int[] blue, Fractal fractal, double... fractalSpecificParams) {
        Pointer kernelParameters = prepareKernelParams(fractal, fractalSpecificParams);

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
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);

        context = new CUcontext();
        cuCtxCreate(context, 0, device);

        module = new CUmodule();
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

    void destroy() {
        memFree(deviceOutputR, deviceOutputG, deviceOutputB);
        cuModuleUnload(module);
        cuCtxDestroy(context);
    }

    private void memFree(CUdeviceptr... devicePtr) {
        for (CUdeviceptr devptr : devicePtr) {
            cuMemFree(devptr);
        }
    }

    private Pointer prepareKernelParams(Fractal fractal, double... fractalSpecificParams) {
        List<Pointer> paramPointers = new ArrayList<>(Arrays.asList(
                Pointer.to(new double[]{fractal.zoom}),
                Pointer.to(new double[]{fractal.posX}),
                Pointer.to(new double[]{fractal.posY}),
                Pointer.to(new int[]{fractal.iterations}),
                Pointer.to(new int[]{imageWidth}),
                Pointer.to(new int[]{imageHeight}),
                Pointer.to(deviceOutputR),
                Pointer.to(deviceOutputG),
                Pointer.to(deviceOutputB)
        ));

        for (double param : fractalSpecificParams) {
            paramPointers.add(Pointer.to(new double[]{param}));
        }
        return Pointer.to(paramPointers.toArray(new Pointer[0]));
    }
}
