package app;

import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.image.PixelWriter;
import javafx.scene.paint.Color;

public class MainWindowController {
    private final static int CANVAS_WIDTH = 555;
    private final int[] red;
    private final int[] green;
    private final int[] blue;
    private final PatternPainter patternPainter;

    @FXML
    Canvas canvas;

    public MainWindowController() {
        int arraySize = CANVAS_WIDTH * CANVAS_WIDTH;
        this.red = new int[arraySize];
        this.green = new int[arraySize];
        this.blue = new int[arraySize];
        patternPainter = new PatternPainter(CANVAS_WIDTH, red, green, blue);
    }

    @FXML
    public void initialize() {
        canvas.setHeight(CANVAS_WIDTH);
        canvas.setWidth(CANVAS_WIDTH);
        loadPattern();
        paintImageOnCanvas();
    }

    public void cudaPaint() {
        CudaPainter painter = new CudaPainter(CANVAS_WIDTH, "/kernels/gradient.ptx", "gradient");
        long cudaStart = System.currentTimeMillis();
        painter.paint(red, green, blue);
        System.out.println("cudaPaint: " + (System.currentTimeMillis() - cudaStart) + "ms");
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
}
