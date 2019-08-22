package pl.potat0x.fractalway.fractalpainter;

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
import pl.potat0x.fractalway.utils.Config;
import pl.potat0x.fractalway.clock.CudaEventClock;
import pl.potat0x.fractalway.fractal.Fractal;
import pl.potat0x.fractalway.fractal.FractalType;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static io.vavr.API.*;
import static io.vavr.Predicates.is;
import static io.vavr.Predicates.isIn;
import static jcuda.driver.JCudaDriver.*;

public class CudaPainter implements FractalPainter {
    private final int imageWidth;
    private final int imageHeight;
    private final long arraySizeInBytes;
    private final String ptxFileName;
    private final String functionName;
    private final int threadsPerBlock;
    private final boolean usePinnedMemory;
    private CUdeviceptr deviceOutputArgb;
    private CUfunction function;
    private CUmodule module;
    private CUcontext context;

    public CudaPainter(int imageWidth, int imageHeight, String kernelFilename, String functionName, boolean usePinnedMemory) {
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.ptxFileName = new File(CudaPainter.class.getResource(kernelFilename).getFile()).getAbsolutePath();
        this.functionName = functionName;
        this.usePinnedMemory = usePinnedMemory;
        arraySizeInBytes = imageWidth * imageHeight * Sizeof.INT;
        threadsPerBlock = Config.getInt("cuda-threads-per-block");
        prepareCuda();
    }

    private double[] getFractalSpecificParams(Fractal fractal) {
        return Match(fractal.type).of(
                Case($(is(FractalType.JULIA_SET)), new double[]{fractal.complexParamRe, fractal.complexParamIm}),
                Case($(isIn(FractalType.MANDELBROT_SET, FractalType.BURNING_SHIP)), new double[0])
        );
    }

    @Override
    public Tuple2<Float, Float> paint(int[] argb, Fractal fractal) {
        System.out.println("CudaPainter.paint, threadsPerBlock=" + threadsPerBlock);

        Pointer kernelParameters = prepareKernelParams(fractal, getFractalSpecificParams(fractal));

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

        if (usePinnedMemory) {
            cuMemAllocHost(deviceOutputArgb, arraySizeInBytes);
        } else {
            cuMemAlloc(deviceOutputArgb, arraySizeInBytes);
        }
    }

    @Override
    public void destroy() {
        if (usePinnedMemory) {
            cuMemFreeHost(deviceOutputArgb);
        } else {
            cuMemFree(deviceOutputArgb);
        }
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
