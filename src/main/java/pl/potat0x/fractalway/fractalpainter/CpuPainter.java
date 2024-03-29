package pl.potat0x.fractalway.fractalpainter;

import io.vavr.Function4;
import io.vavr.Tuple;
import io.vavr.Tuple2;
import pl.potat0x.fractalway.clock.Clock;
import pl.potat0x.fractalway.fractal.Fractal;
import pl.potat0x.fractalway.fractal.FractalType;
import pl.potat0x.fractalway.utils.Config;
import pl.potat0x.fractalway.utils.math.IntervalDistributor;

import java.util.List;
import java.util.stream.Collectors;

import static io.vavr.API.*;
import static io.vavr.Predicates.is;

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

        Clock clock = new Clock();
        makeCalculations(Config.getInt("cpu-threads"), argb, fractal, fractalFunction);
        return Tuple.of(clock.getElapsedTime() * 1f, 0f);
    }

    private void makeCalculations(int numberOfThreads, int[] argb, Fractal fractal, Function4<int[], Fractal, Integer, Integer, Void> fractalFunction) {
        //noinspection SuspiciousNameCombination
        List<Tuple2<Integer, Integer>> sections = IntervalDistributor.cutIntervalToSimilarSections(numberOfThreads, height);

        List<Thread> threads = sections.stream()
                .map(section -> {
                    Thread thread = new Thread(() -> calculateFractalArea(section, argb, fractal, fractalFunction));
                    thread.setDaemon(true);
                    thread.start();
                    return thread;
                }).collect(Collectors.toList());

        threads.forEach(t -> {
            try {
                t.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
    }

    private void calculateFractalArea(Tuple2<Integer, Integer> rangeY, int[] argb, Fractal fractal, Function4<int[], Fractal, Integer, Integer, Void> fractalFunction) {
        for (int y = rangeY._1; y < rangeY._2; y++) {
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

        Tuple2<Double, Double> p = calculatePositionParam(fractal, x, y);
        double pRe = p._1;
        double pIm = p._2;

        int i = 0;
        while (i++ < fractal.iterations - 1) {
            // zNext = zPrev*zPrev + p
            zNextRe = zPrevRe * zPrevRe - zPrevIm * zPrevIm + pRe;
            zNextIm = 2.0 * zPrevRe * zPrevIm + pIm;

            // |zNext| > 4.0
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

        Tuple2<Double, Double> p = calculatePositionParam(fractal, x, y);
        double pRe = p._1;
        double pIm = p._2;

        double zPrevRe = pRe;
        double zPrevIm = pIm;

        int i = 0;
        while (i++ < fractal.iterations - 1) {
            // zNext = zPrev*zPrev + c
            zNextRe = zPrevRe * zPrevRe - zPrevIm * zPrevIm + fractal.complexParamRe;
            zNextIm = 2.0 * zPrevRe * zPrevIm + fractal.complexParamIm;

            // |zNext| > 4.0
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

        Tuple2<Double, Double> p = calculatePositionParam(fractal, x, y);
        double pRe = p._1;
        double pIm = p._2;

        int i = 0;
        while (i++ < fractal.iterations - 1) {
            // zNext = zPrev*zPrev + p
            zNextRe = zPrevRe * zPrevRe - zPrevIm * zPrevIm + pRe;
            zNextIm = 2.0 * zPrevRe * zPrevIm + pIm;

            // |zNext| > 4.0
            if ((zNextRe * zNextRe + zNextIm * zNextIm) > 4.0) {
                break;
            }

            zPrevRe = Math.abs(zNextRe);
            zPrevIm = Math.abs(zNextIm);
        }

        createArgbColor(argb, fractal, x, y, i);
        return null;
    }

    private Tuple2<Double, Double> calculatePositionParam(Fractal fractal, int x, int y) {
        double pRe = (x - width / 2.0) * fractal.zoom + fractal.posX;
        double pIm = (y - height / 2.0) * fractal.zoom + fractal.posY;
        return Tuple.of(pRe, pIm);
    }

    private void createArgbColor(int[] argb, Fractal fractal, int x, int y, int i) {
        int color = (int) ((255.0 * i) / (1.0 * fractal.iterations));
        int idx = y * width + x;
        argb[idx] = (255 << 24) | (color << 16) | (color << 8) | color;
    }
}
