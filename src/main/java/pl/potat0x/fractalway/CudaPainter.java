package pl.potat0x.fractalway;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
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
    private CUdeviceptr deviceOutputArgb;
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

    Tuple2<Float, Float> paint(int[] argb, Fractal fractal, double... fractalSpecificParams) {
        Pointer kernelParameters = prepareKernelParams(fractal, fractalSpecificParams);

        dim3 dimBlock = new dim3(threadsPerBlock, threadsPerBlock, 1);
        int blocksPerGridX = (imageWidth + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGridY = (imageHeight + threadsPerBlock - 1) / threadsPerBlock;
        dim3 dimGrid = new dim3(blocksPerGridX, blocksPerGridY, 1);
        System.out.println("block " + dimBlock + "\ngrid  " + dimGrid);

        CudaEventClock cudaClock = new CudaEventClock();

        cudaClock.start();
        cuLaunchKernel(function,
                dimGrid.x, dimGrid.y, dimGrid.z,
                dimBlock.x, dimBlock.y, dimBlock.z,
                0, null, kernelParameters, null
        );
        float kernelTime = cudaClock.stop();

        cudaClock.start();
        cuMemcpyDtoH(Pointer.to(argb), deviceOutputArgb, arraySizeInBytes);
        float memcpyTime = cudaClock.stop();

        cudaClock.destroy();
        return Tuple.of(kernelTime, memcpyTime);
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

        deviceOutputArgb = new CUdeviceptr();
        cuMemAlloc(deviceOutputArgb, arraySizeInBytes);
    }

    void destroy() {
        cuMemFree(deviceOutputArgb);
        cuModuleUnload(module);
        cuCtxDestroy(context);
    }

    private Pointer prepareKernelParams(Fractal fractal, double... fractalSpecificParams) {
        List<Pointer> paramPointers = new ArrayList<>(Arrays.asList(
                Pointer.to(new double[]{fractal.zoom}),
                Pointer.to(new double[]{fractal.posX}),
                Pointer.to(new double[]{fractal.posY}),
                Pointer.to(new int[]{fractal.iterations}),
                Pointer.to(new int[]{imageWidth}),
                Pointer.to(new int[]{imageHeight}),
                Pointer.to(deviceOutputArgb)
        ));

        for (double param : fractalSpecificParams) {
            paramPointers.add(Pointer.to(new double[]{param}));
        }
        return Pointer.to(paramPointers.toArray(new Pointer[0]));
    }
}
