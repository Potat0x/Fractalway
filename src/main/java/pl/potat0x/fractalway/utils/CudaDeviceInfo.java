package pl.potat0x.fractalway.utils;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import jcuda.driver.CUdevice;

import java.nio.charset.StandardCharsets;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuDeviceGetName;
import static jcuda.driver.JCudaDriver.cuDriverGetVersion;
import static jcuda.driver.JCudaDriver.cuMemGetInfo;

public class CudaDeviceInfo {
    private CUdevice device;

    public CudaDeviceInfo(int deviceOrdinal) {
        device = new CUdevice();
        cuDeviceGet(device, deviceOrdinal);
    }

    public String name() {
        byte[] name = new byte[512];
        cuDeviceGetName(name, name.length, device);
        return new String(name, StandardCharsets.US_ASCII);
    }

    public String computeCapability() {
        int[] major = {0};
        int[] minor = {0};
        cuDeviceGetAttribute(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        return "" + major[0] + "." + minor[0];
    }

    public Tuple2<Long, Long> freeAndTotalMemoryInBytes() {
        long[] free = {0};
        long[] total = {0};
        cuMemGetInfo(free, total);
        return Tuple.of(total[0] - free[0], total[0]);
    }

    public String cudaVersion() {
        int[] cudaVersion = {0};
        cuDriverGetVersion(cudaVersion);
        return decodeCudaVersion(cudaVersion[0]);
    }

    private String decodeCudaVersion(int ver) {
        int major = ver / 1000;
        int minor = (ver - major * 1000) / 10;
        return "" + major + "." + minor;
    }
}
