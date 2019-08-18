package pl.potat0x.fractalway.fractalpainter;

import io.vavr.Tuple2;
import pl.potat0x.fractalway.fractal.Fractal;

public interface FractalPainter {
    Tuple2<Float, Float> paint(int[] argb, Fractal fractal);

    default void destroy() {
    }
}
