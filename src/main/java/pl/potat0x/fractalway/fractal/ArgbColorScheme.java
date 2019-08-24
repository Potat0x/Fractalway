package pl.potat0x.fractalway.fractal;

public class ArgbColorScheme {
    public int redLeftShift, redRightShift;
    public int greenLeftShift, greenRightShift;
    public int blueLeftShift, blueRightShift;
    public boolean redLeftMultiplication, redRightMultiplication;
    public boolean greenLeftMultiplication, greenRightMultiplication;
    public boolean blueLeftMultiplication, blueRightMultiplication;

    public ArgbColorScheme() {
        setToDefault();
    }

    void setToDefault() {
        redLeftShift = 8;
        redRightShift = 24;

        greenLeftShift = 16;
        greenRightShift = 24;

        blueLeftShift = 24;
        blueRightShift = 24;
    }
}
