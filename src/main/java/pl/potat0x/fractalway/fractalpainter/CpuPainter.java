package pl.potat0x.fractalway.fractalpainter;

import io.vavr.Function4;
import io.vavr.Tuple;
import io.vavr.Tuple2;
import pl.potat0x.fractalway.fractal.Fractal;
import pl.potat0x.fractalway.fractal.FractalType;

import static io.vavr.API.*;
import static io.vavr.Predicates.is;
import static java.lang.Math.abs;

public class CpuPainter implements FractalPainter {
    private final int width;
    private final int height;

    public CpuPainter(int width, int height) {
        this.width = width;
        this.height = height;
    }

    @Override
    public Tuple2<Float, Float> paint(int[] argb, Fractal fractal) {
        System.out.println("CpuPainter.paint");
        Function4<int[], Fractal, Integer, Integer, Void> fractalFunction = Match(fractal.type).of(
                Case($(is(FractalType.MANDELBROT_SET)), Function4.of(this::mandelbrotSet)),
                Case($(is(FractalType.JULIA_SET)), Function4.of(this::juliaSet)),
                Case($(is(FractalType.BURNING_SHIP)), Function4.of(this::burningShip))
        );

        long start = System.currentTimeMillis();
        makeCalculation(argb, fractal, fractalFunction);
        long end = System.currentTimeMillis() - start;
        return Tuple.of(end * 1f, 0f);
    }

    private void makeCalculation(int[] argb, Fractal fractal, Function4<int[], Fractal, Integer, Integer, Void> fractalFunction) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                fractalFunction.apply(argb, fractal, x, y);
            }
        }
    }

    private Void mandelbrotSet(int[] argb, Fractal fractal, int x, int y) {
        double zPrevRe = 0;
        double zPrevIm = 0;
        double zNextRe;
        double zNextIm;

        double pRe = (x - width / 2.0) * fractal.zoom + fractal.posX;
        double pIm = (y - height / 2.0) * fractal.zoom + fractal.posY;

        int i = 0;
        while (i++ < fractal.iterations - 1) {
            // zNext = zPrev*zPrev + p
            zNextRe = zPrevRe * zPrevRe - zPrevIm * zPrevIm + pRe;
            zNextIm = 2.0 * zPrevRe * zPrevIm + pIm;

            // |zPrev| > 4.0
            if ((zNextRe * zNextRe + zNextIm * zNextIm) > 4.0) {
                break;
            }

            zPrevRe = zNextRe;
            zPrevIm = zNextIm;
        }

        createArgbColor(argb, fractal, x, y, i);
        return null;
    }

    private Void juliaSet(int[] argb, Fractal fractal, int x, int y) {
        double zNextRe;
        double zNextIm;

        double pRe = (x - width / 2.0) * fractal.zoom + fractal.posX;
        double pIm = (y - height / 2.0) * fractal.zoom + fractal.posY;

        double zPrevRe = pRe;
        double zPrevIm = pIm;

        int i = 0;
        while (i++ < fractal.iterations - 1) {
            // zNext = zPrev*zPrev + c
            zNextRe = zPrevRe * zPrevRe - zPrevIm * zPrevIm + fractal.complexParamRe;
            zNextIm = 2.0 * zPrevRe * zPrevIm + fractal.complexParamIm;

            // |zPrev| > 4.0
            if ((zNextRe * zNextRe + zNextIm * zNextIm) > 4.0) {
                break;
            }

            zPrevRe = zNextRe;
            zPrevIm = zNextIm;
        }

        createArgbColor(argb, fractal, x, y, i);
        return null;
    }

    private Void burningShip(int[] argb, Fractal fractal, int x, int y) {
        double zPrevRe = 0;
        double zPrevIm = 0;
        double zNextRe;
        double zNextIm;

        double pRe = (x - width / 2.0) * fractal.zoom + fractal.posX;
        double pIm = (y - height / 2.0) * fractal.zoom + fractal.posY;

        int i = 0;
        while (i++ < fractal.iterations - 1) {
            // zNext = zPrev*zPrev + p
            zNextRe = zPrevRe * zPrevRe - zPrevIm * zPrevIm + pRe;
            zNextIm = 2.0 * zPrevRe * zPrevIm + pIm;

            // |zPrev| > 4.0
            if ((zNextRe * zNextRe + zNextIm * zNextIm) > 4.0) {
                break;
            }

            zPrevRe = Math.abs(zNextRe);
            zPrevIm = Math.abs(zNextIm);
        }

        createArgbColor(argb, fractal, x, y, i);
        return null;
    }

    private void createArgbColor(int[] argb, Fractal fractal, int x, int y, int i) {
        double color = (255.0 * i) / (1.0 * fractal.iterations);
        int r = (int) (17.0 * (abs(255 - color) / 255.0));
        int g = (int) (255.0 * (color / 255.0));
        int b = (int) (33.0 * (abs(255 - color) / 255.0));
        int idx = y * width + x;
        argb[idx] = (255 << 24) | (r << 16) | (g << 8) | b;
    }
}
