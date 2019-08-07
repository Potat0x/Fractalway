package app;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.image.PixelWriter;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;

public class MainWindowController {
    private final int CANVAS_WIDTH = 512;
    private final int[] red;
    private final int[] green;
    private final int[] blue;

    private final PatternPainter patternPainter;
    private final CudaPainter painter;

    private int maxIter = 250;

    private double zoom = 0.0055;
    private double zoomStep = 1.5;
    private double posX = 0;
    private double posY = 0;
    private double moveStep = 32;

    @FXML
    Canvas canvas;

    public MainWindowController() {
        int arraySize = CANVAS_WIDTH * CANVAS_WIDTH;
        this.red = new int[arraySize];
        this.green = new int[arraySize];
        this.blue = new int[arraySize];
        patternPainter = new PatternPainter(CANVAS_WIDTH, red, green, blue);
        painter = new CudaPainter(CANVAS_WIDTH, "/kernels/mandelbrotSet.ptx", "mandelbrotSet");
    }

    @FXML
    public void initialize() {
        canvas.setHeight(CANVAS_WIDTH);
        canvas.setWidth(CANVAS_WIDTH);
        loadPattern();
//        paintImageOnCanvas();
        cudaPaint();
    }

    private void cudaPaint() {
        long cudaStart = System.currentTimeMillis();
        painter.paint(red, green, blue, zoom, posX, posY, maxIter);
        System.out.println("cudaPaint (" + (System.currentTimeMillis() - cudaStart) + "ms): zoom = " + zoom + "; posX = " + posX + "; posY = " + posY + ";");
        long paintStart = System.currentTimeMillis();
        paintImageOnCanvas();
        System.out.println("paintImageOnCanvas: " + (System.currentTimeMillis() - paintStart) + "ms");
    }

    private void paintImageOnCanvas() {
        PixelWriter pixelWriter = canvas.getGraphicsContext2D().getPixelWriter();
        for (int y = 0; y < CANVAS_WIDTH; y++) {
            for (int x = 0; x < CANVAS_WIDTH; x++) {
                pixelWriter.setColor(x, y, createColor(x, y));
            }
        }
    }

    private void loadPattern() {
        for (int y = 0; y < canvas.getHeight(); y++) {
            for (int x = 0; x < canvas.getWidth(); x++) {
                patternPainter.gradient(x, y);
            }
        }
    }

    private Color createColor(int x, int y) {
        int index = calculateIndex(x, y);
        double r = red[index] / 255.0;
        double g = green[index] / 255.0;
        double b = blue[index] / 255.0;
        return new Color(r, g, b, 1);
    }

    private int calculateIndex(int x, int y) {
        return CANVAS_WIDTH * y + x;
    }

    public void areaStop(MouseEvent mouseEvent) {
        System.out.println("areaStop: " + mouseEvent.getX() + ", " + mouseEvent.getY());
        cudaPaint();
    }

    public void moveDown(ActionEvent actionEvent) {
        posY -= moveStep * zoom;
        cudaPaint();
    }

    public void moveUp(ActionEvent actionEvent) {
        posY += moveStep * zoom;
        cudaPaint();
    }

    public void moveLeft(ActionEvent actionEvent) {
        posX -= moveStep * zoom;
        cudaPaint();
    }

    public void moveRight(ActionEvent actionEvent) {
        posX += moveStep * zoom;
        cudaPaint();
    }

    public void decreaseZoom(ActionEvent actionEvent) {
        System.out.println("decreaseZoom");
        zoom *= zoomStep;
        cudaPaint();
    }

    public void increaseZoom(ActionEvent actionEvent) {
        System.out.println("increaseZoom");
        zoom /= zoomStep;
        cudaPaint();
    }

    public void setPosition(MouseEvent mouseEvent) {
        double diffCenterX = CANVAS_WIDTH / 2.0 - mouseEvent.getX();
        double diffCenterY = CANVAS_WIDTH / 2.0 - mouseEvent.getY();
        posX -= diffCenterX * zoom;
        posY -= diffCenterY * zoom;
        cudaPaint();
    }
}
