package pl.potat0x.fractalway.utils.device;

import oshi.SystemInfo;
import oshi.hardware.CentralProcessor;
import oshi.hardware.HardwareAbstractionLayer;

public class CpuInfo {
    public final CentralProcessor cpu;

    public CpuInfo() {
        SystemInfo systemInfo = new SystemInfo();
        HardwareAbstractionLayer hal = systemInfo.getHardware();
        cpu = hal.getProcessor();
    }
}
