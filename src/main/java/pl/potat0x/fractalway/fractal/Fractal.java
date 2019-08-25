package pl.potat0x.fractalway.fractal;

import pl.potat0x.fractalway.utils.Config;

import java.util.Objects;

public class Fractal {
    private final int upperIterLimit;

    public final FractalType type;
    public int iterations = 160;

    public double posX = 0;
    public double posY = 0;
    public double positionStep = 32;

    public double zoom = 0.0055;
    public double zoomMultiplier = 1.2;

    public double complexParamRe = -0.8;
    public double complexParamIm = 0.156;

    public Fractal(FractalType type, int upperIterLimit) {
        this.type = type;
        this.upperIterLimit = upperIterLimit;
    }

    public void moveFractalPointToImageCenter(double imageWidth, double imageHeight, double x, double y) {
        double diffCenterX = imageWidth / 2.0 - x;
        double diffCenterY = imageHeight / 2.0 - y;
        posX -= diffCenterX * zoom;
        posY -= diffCenterY * zoom;
    }

    public void increaseIterations(int increaseBy) {
        iterations = Math.min(iterations + increaseBy, upperIterLimit);
    }

    public void decreaseIterations(int decreaseBy) {
        int lowerIterLimit = 1;
        iterations = Math.max(iterations - decreaseBy, lowerIterLimit);
    }

    public void zoomIn() {
        zoom /= zoomMultiplier;
    }

    public void zoomOut() {
        zoom *= zoomMultiplier;
    }

    public void moveUp() {
        posY -= positionStep * zoom;
    }

    public void moveDown() {
        posY += positionStep * zoom;
    }

    public void moveLeft() {
        posX -= positionStep * zoom;
    }

    public void moveRight() {
        posX += positionStep * zoom;
    }

    public String getViewAsString() {
        return "zoom = " + zoom + "; posX = " + posX + "; posY = " + posY + ";";
    }

    public Fractal copy() {
        Fractal newFractal = new Fractal(type, upperIterLimit);
        newFractal.iterations = iterations;
        newFractal.posX = posX;
        newFractal.posY = posY;
        newFractal.positionStep = positionStep;
        newFractal.zoom = zoom;
        newFractal.zoomMultiplier = zoomMultiplier;
        newFractal.complexParamRe = complexParamRe;
        newFractal.complexParamIm = complexParamIm;
        return newFractal;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Fractal fractal = (Fractal) o;
        return upperIterLimit == fractal.upperIterLimit &&
                iterations == fractal.iterations &&
                Double.compare(fractal.posX, posX) == 0 &&
                Double.compare(fractal.posY, posY) == 0 &&
                Double.compare(fractal.positionStep, positionStep) == 0 &&
                Double.compare(fractal.zoom, zoom) == 0 &&
                Double.compare(fractal.zoomMultiplier, zoomMultiplier) == 0 &&
                Double.compare(fractal.complexParamRe, complexParamRe) == 0 &&
                Double.compare(fractal.complexParamIm, complexParamIm) == 0 &&
                type == fractal.type;
    }

    @Override
    public int hashCode() {
        return Objects.hash(upperIterLimit, type, iterations, posX, posY, positionStep, zoom, zoomMultiplier, complexParamRe, complexParamIm);
    }
}
