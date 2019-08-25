package pl.potat0x.fractalway.fractal;

import org.junit.Test;

import static org.junit.Assert.*;

public class FractalTest {
    @Test
    public void copyingTest() {
        Fractal fr = new Fractal(FractalType.MANDELBROT_SET, 0);
        fr.iterations = 1;
        fr.posX = 2;
        fr.posY = 3;
        fr.positionStep = 4;
        fr.zoom = 5;
        fr.zoomMultiplier = 6;
        fr.complexParamRe = 7;
        fr.complexParamIm = 8;

        Fractal frCopy = fr.copy();

        assertNotSame(fr, frCopy);
        assertEquals(fr, frCopy);
    }
}
