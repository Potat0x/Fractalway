package pl.potat0x.fractalway.fractal;

public class Fractal {
    public final FractalType type;
    public int maxIter = 200;

    public double posX = 0;
    public double posY = 0;
    public double positionStep = 32;

    public double zoom = 0.0055;
    public double zoomMultiplier = 1.2;

    public double complexParamRe = -0.8;
    public double complexParamIm = 0.156;

    public Fractal(FractalType type) {
        this.type = type;
    }

    public void moveFractalPointToImageCenter(double imageWidth, double imageHeight, double x, double y) {
        double diffCenterX = imageWidth / 2.0 - x;
        double diffCenterY = imageHeight / 2.0 - y;
        posX -= diffCenterX * zoom;
        posY -= diffCenterY * zoom;
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
}
