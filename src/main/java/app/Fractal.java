package app;

public class Fractal {
    final FractalType type;
    public int maxIter = 200;
    public double zoom = 0.0055;
    public double posX = 0;
    public double posY = 0;
    public double zoomStep = 1.2;
    public double moveStep = 32;

    Fractal(FractalType type) {
        this.type = type;
    }

    void moveFractalPointToImageCenter(double imageWidth, double imageHeight, double x, double y) {
        double diffCenterX = imageWidth / 2.0 - x;
        double diffCenterY = imageHeight / 2.0 - y;
        posX -= diffCenterX * zoom;
        posY -= diffCenterY * zoom;
    }

    void zoomIn() {
        zoom /= zoomStep;
    }

    void zoomOut() {
        zoom *= zoomStep;
    }

    void moveUp() {
        posY -= moveStep * zoom;
    }

    void moveDown() {
        posY += moveStep * zoom;
    }

    void moveLeft() {
        posX -= moveStep * zoom;
    }

    void moveRight() {
        posX += moveStep * zoom;
    }

    String getViewAsString() {
        return "zoom = " + zoom + "; posX = " + posX + "; posY = " + posY + ";";
    }
}
