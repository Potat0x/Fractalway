package app;

class Fractal {
    final FractalType type;
    int maxIter = 200;
    double zoom = 0.0055;
    double posX = 0;
    double posY = 0;
    double zoomStep = 1.2;
    double moveStep = 32;

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
        zoom *= zoomStep;
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
