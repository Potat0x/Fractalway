package pl.potat0x.fractalway.clock;

import jcuda.driver.CUevent;

import static jcuda.driver.CUevent_flags.CU_EVENT_DEFAULT;
import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventDestroy;
import static jcuda.driver.JCudaDriver.cuEventElapsedTime;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuEventSynchronize;

public class CudaEventClock {
    private final CUevent start;
    private final CUevent stop;

    public CudaEventClock() {
        start = new CUevent();
        cuEventCreate(start, CU_EVENT_DEFAULT);
        stop = new CUevent();
        cuEventCreate(stop, CU_EVENT_DEFAULT);
    }

    public void start() {
        cuEventRecord(start, null);
    }

    public float stop() {
        cuEventRecord(stop, null);
        cuEventSynchronize(stop);
        float[] kernelTime = {0};
        cuEventElapsedTime(kernelTime, start, stop);
        return kernelTime[0];
    }

    public void destroy() {
        cuEventDestroy(start);
        cuEventDestroy(stop);
    }
}
