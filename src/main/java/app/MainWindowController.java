package app;

import javafx.fxml.FXML;
import javafx.geometry.Point2D;
import javafx.scene.canvas.Canvas;
import javafx.scene.image.PixelWriter;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.input.ScrollEvent;
import javafx.scene.paint.Color;

import java.awt.*;

import static io.vavr.API.*;
import static io.vavr.Predicates.is;

public class MainWindowController {
    private final int CANVAS_WIDTH = 800;
    private final int CANVAS_HEIGHT = 600;
    private final int[] red;
    private final int[] green;
    private final int[] blue;

    private final PatternPainter patternPainter;
    private final CudaPainter painter;

    private int maxIter = 200;

    private double zoom = 0.0055;
    private double zoomStep = 1.2;
    private double posX = 0;
    private double posY = 0;
    private double moveStep = 32;

    @FXML
    Canvas canvas;

    public MainWindowController() {
        int arraySize = CANVAS_WIDTH * CANVAS_HEIGHT;
        this.red = new int[arraySize];
        this.green = new int[arraySize];
        this.blue = new int[arraySize];
        patternPainter = new PatternPainter(CANVAS_WIDTH, red, green, blue);
        painter = new CudaPainter(CANVAS_WIDTH, CANVAS_HEIGHT, "/kernels/mandelbrotSet.ptx", "mandelbrotSet");
    }

    @FXML
    public void initialize() {
        canvas.setWidth(CANVAS_WIDTH);
        canvas.setHeight(CANVAS_HEIGHT);
        canvas.setFocusTraversable(true);
//        loadPattern();
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
        for (int y = 0; y < CANVAS_HEIGHT; y++) {
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

    public void updateFractalPosition(MouseEvent mouseEvent) {
        moveClickedFractalPointToCanvasCenter(mouseEvent.getX(), mouseEvent.getY());
        cudaPaint();
        if (mouseEvent.isControlDown()) {
            moveMouseToCanvasCenter();
        }
    }

    public void updateFractalZoom(ScrollEvent scrollEvent) {
        if (scrollEvent.isControlDown()) {
            moveMouseToCanvasCenter();
            moveClickedFractalPointToCanvasCenter(scrollEvent.getX(), scrollEvent.getY());
        }

        if (scrollEvent.getDeltaY() > 0) {
            zoom /= zoomStep;
        } else {
            zoom *= zoomStep;
        }

        cudaPaint();
    }

    public void handleKeyPressed(KeyEvent keyEvent) {
        Match(keyEvent.getCode()).option(
                Case($(is(KeyCode.UP)), o -> run(() -> posY -= moveStep * zoom)),
                Case($(is(KeyCode.DOWN)), o -> run(() -> posY += moveStep * zoom)),
                Case($(is(KeyCode.LEFT)), o -> run(() -> posX -= moveStep * zoom)),
                Case($(is(KeyCode.RIGHT)), o -> run(() -> posX += moveStep * zoom)),
                Case($(is(KeyCode.D)), o -> run(() -> zoom /= zoomStep)),
                Case($(is(KeyCode.A)), o -> run(() -> zoom *= zoomStep))
        ).peek(x -> cudaPaint());
    }

    private void moveClickedFractalPointToCanvasCenter(double eventX, double eventY) {
        double diffCenterX = CANVAS_WIDTH / 2.0 - eventX;
        double diffCenterY = CANVAS_HEIGHT / 2.0 - eventY;
        posX -= diffCenterX * zoom;
        posY -= diffCenterY * zoom;
    }

    private void moveMouseToCanvasCenter() {
        Point2D point2D = canvas.localToScreen(CANVAS_WIDTH / 2.0, CANVAS_HEIGHT / 2.0);
        setMousePos(Math.round(point2D.getX()), Math.round(point2D.getY()));
    }

    private void setMousePos(long x, long y) {
        try {
            new Robot().mouseMove((int) x, (int) y);
        } catch (AWTException e) {
            e.printStackTrace();
        }
    }
}
