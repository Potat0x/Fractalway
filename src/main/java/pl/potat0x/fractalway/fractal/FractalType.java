package pl.potat0x.fractalway.fractal;

public enum FractalType {
    MANDELBROT_SET("Mandelbrot set"),
    JULIA_SET("Julia set");

    private final String text;

    FractalType(String text) {
        this.text = text;
    }

    @Override
    public String toString() {
        return text;
    }
}
