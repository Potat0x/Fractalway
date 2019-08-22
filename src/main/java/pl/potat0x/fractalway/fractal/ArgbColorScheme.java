package pl.potat0x.fractalway.fractal;

public class ArgbColorScheme {
    public int rLeftShift, rRightShift;
    public int gLeftShift, gRightShift;
    public int bLeftShift, bRightShift;

    public ArgbColorScheme() {
        setToDefault();
    }

    void setToDefault() {
        rLeftShift = 8;
        rRightShift = 24;

        gLeftShift = 16;
        gRightShift = 24;

        bLeftShift = 24;
        bRightShift = 24;
    }
}
