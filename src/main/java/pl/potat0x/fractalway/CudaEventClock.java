package pl.potat0x.fractalway;

import jcuda.driver.CUevent;

import static jcuda.driver.CUevent_flags.CU_EVENT_DEFAULT;
import static jcuda.driver.JCudaDriver.cuEventCreate;
import static jcuda.driver.JCudaDriver.cuEventDestroy;
import static jcuda.driver.JCudaDriver.cuEventElapsedTime;
import static jcuda.driver.JCudaDriver.cuEventRecord;
import static jcuda.driver.JCudaDriver.cuEventSynchronize;

class CudaEventClock {
    private final CUevent start;
    private final CUevent stop;

    CudaEventClock() {
        start = new CUevent();
        cuEventCreate(start, CU_EVENT_DEFAULT);
        stop = new CUevent();
        cuEventCreate(stop, CU_EVENT_DEFAULT);
    }

    void start() {
        cuEventRecord(start, null);
    }

    float stop() {
        cuEventRecord(stop, null);
        cuEventSynchronize(stop);
        float[] kernelTime = {0};
        cuEventElapsedTime(kernelTime, start, stop);
        return kernelTime[0];
    }

    void destroy() {
        cuEventDestroy(start);
        cuEventDestroy(stop);
    }
}
