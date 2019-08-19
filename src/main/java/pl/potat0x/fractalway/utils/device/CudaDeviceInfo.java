package pl.potat0x.fractalway.utils.device;

import jcuda.driver.CUdevice;

import java.nio.charset.StandardCharsets;

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuDeviceGetAttribute;
import static jcuda.driver.JCudaDriver.cuDeviceGetCount;
import static jcuda.driver.JCudaDriver.cuDeviceGetName;
import static jcuda.driver.JCudaDriver.cuDriverGetVersion;

public class CudaDeviceInfo {
    private CUdevice device;

    public static int getDeviceCount() {
        int[] count = {0};
        cuDeviceGetCount(count);
        return count[0];
    }

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
        return major[0] + "." + minor[0];
    }

    public String cudaVersion() {
        int[] cudaVersion = {0};
        cuDriverGetVersion(cudaVersion);
        return decodeCudaVersion(cudaVersion[0]);
    }

    private String decodeCudaVersion(int ver) {
        int major = ver / 1000;
        int minor = (ver - major * 1000) / 10;
        return major + "." + minor;
    }
}
