package pl.potat0x.fractalway;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.Point2D;
import javafx.scene.Cursor;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.*;
import javafx.scene.image.PixelWriter;
import javafx.scene.input.*;
import javafx.scene.paint.Color;
import pl.potat0x.fractalway.fractal.Fractal;
import pl.potat0x.fractalway.fractal.FractalType;
import pl.potat0x.fractalway.settings.FractalSettingsController;
import pl.potat0x.fractalway.settings.NavigationSettingsController;
import pl.potat0x.fractalway.utils.StringCapitalizer;
import pl.potat0x.fractalway.utils.WindowBuilder;

import java.awt.AWTException;
import java.awt.Robot;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static io.vavr.API.*;
import static io.vavr.Predicates.is;
import static io.vavr.Predicates.isIn;

public class MainController {
    private final int CANVAS_WIDTH = 800;
    private final int CANVAS_HEIGHT = 600;
    private final int[] red;
    private final int[] green;
    private final int[] blue;

    private final PatternPainter patternPainter;
    private Fractal fractal;
    private CudaPainter painter;

    @FXML
    private Canvas canvas;
    @FXML
    private Menu fractalMenu;
    @FXML
    private Menu settingsMenu;
    @FXML
    private Menu canvasCursorMenu;
    @FXML
    private Label deviceInfoLabel;
    @FXML
    private CheckMenuItem deviceInfoMenuItem;

    public MainController() {
        int arraySize = CANVAS_WIDTH * CANVAS_HEIGHT;
        this.red = new int[arraySize];
        this.green = new int[arraySize];
        this.blue = new int[arraySize];
        patternPainter = new PatternPainter(CANVAS_WIDTH, red, green, blue);
    }

    @FXML
    private void initialize() {
        initCanvas();
        initFractalMenu();
        drawPattern();
        initCursorMenu();
        initDeviceInfoLabel();
        initDeviceInfoMenuItem();
        drawFractal();
    }

    @FXML
    private void updateFractalPosition(MouseEvent mouseEvent) {
        moveClickedFractalPointToCanvasCenter(mouseEvent.getX(), mouseEvent.getY());
        drawFractal();
        if (mouseEvent.isControlDown()) {
            moveMouseToCanvasCenter();
        }
    }

    @FXML
    private void updateFractalZoom(ScrollEvent scrollEvent) {
        if (scrollEvent.isControlDown()) {
            moveMouseToCanvasCenter();
            moveClickedFractalPointToCanvasCenter(scrollEvent.getX(), scrollEvent.getY());
        }

        if (scrollEvent.getDeltaY() > 0) {
            fractal.zoomIn();
        } else {
            fractal.zoomOut();
        }
        drawFractal();
    }

    @FXML
    private void handleKeyPressed(KeyEvent keyEvent) {
        Match(keyEvent.getCode()).option(
                Case($(is(KeyCode.UP)), o -> run(() -> fractal.moveUp())),
                Case($(is(KeyCode.DOWN)), o -> run(() -> fractal.moveDown())),
                Case($(is(KeyCode.LEFT)), o -> run(() -> fractal.moveLeft())),
                Case($(is(KeyCode.RIGHT)), o -> run(() -> fractal.moveRight())),
                Case($(is(KeyCode.D)), o -> run(() -> fractal.zoomIn())),
                Case($(is(KeyCode.A)), o -> run(() -> fractal.zoomOut())),
                Case($(is(KeyCode.C).or(is(KeyCode.ADD))), o -> run(() -> fractal.increaseIterations())),
                Case($(is(KeyCode.Z).or(is(KeyCode.SUBTRACT))), o -> run(() -> fractal.decreaseIterations()))
        ).peek(x -> drawFractal());
    }

    @FXML
    private void showNavigationSettingsWindow(ActionEvent actionEvent) throws IOException {
        openWindow("/fxml/navigation_settings.fxml", new NavigationSettingsController(fractal, this::drawFractal), "Navigation settings");
    }

    @FXML
    private void showFractalSettingsWindow(ActionEvent actionEvent) throws IOException {
        openWindow("/fxml/fractal_settings.fxml", new FractalSettingsController(fractal, this::drawFractal), "Fractal settings");
    }

    private void initCanvas() {
        canvas.setWidth(CANVAS_WIDTH);
        canvas.setHeight(CANVAS_HEIGHT);
        canvas.setFocusTraversable(true);
    }

    private void initCursorMenu() {
        ToggleGroup cursorGroup = new ToggleGroup();
        List<MenuItem> menuItems = addItemsToToggleGroup(cursorGroup, Cursor.DEFAULT, Cursor.CROSSHAIR, Cursor.NONE);
        canvasCursorMenu.getItems().addAll(menuItems);

        cursorGroup.selectedToggleProperty().addListener((observable, oldValue, newValue) -> {
            Cursor cursor = (Cursor) newValue.getUserData();
            Match(cursor).of(
                    Case($(isIn(Cursor.CROSSHAIR, Cursor.NONE)), o -> run(() -> canvas.setCursor(cursor))),
                    Case($(), o -> run(() -> canvas.setCursor(Cursor.DEFAULT)))
            );
        });
    }

    private void initFractalMenu() {
        ToggleGroup fractalGroup = new ToggleGroup();
        List<MenuItem> menuItems = addItemsToToggleGroup(fractalGroup, (Object[]) FractalType.values());
        fractalMenu.getItems().addAll(menuItems);

        fractalGroup.selectedToggleProperty().addListener((observable, oldValue, newValue) -> {
            FractalType newType = (FractalType) newValue.getUserData();
            fractal = new Fractal(newType);
            releaseFractalPainter();
            painter = createFractalPainter();
            drawFractal();
        });

        if (fractalGroup.getToggles().size() > 0) {
            fractalGroup.getToggles().get(0).setSelected(true);
        }
    }

    private void initDeviceInfoLabel() {
        CudaDeviceInfo devInfo = new CudaDeviceInfo(0);
        String text = devInfo.name() +
                "\nCUDA version: " + devInfo.cudaVersion() +
                "\nCompute capability: " + devInfo.computeCapability() +
                "\nMemory: " + devInfo.freeAndTotalMemoryInBytes()._2 / (1024 * 1024) + " MB";
        Platform.runLater(() -> deviceInfoLabel.setText(text));
    }

    private void initDeviceInfoMenuItem() {
        deviceInfoMenuItem.selectedProperty().addListener((observable, oldValue, newValue) -> deviceInfoLabel.setVisible(newValue));
        deviceInfoMenuItem.selectedProperty().set(true);
    }

    private List<MenuItem> addItemsToToggleGroup(ToggleGroup group, Object... items) {
        List<MenuItem> menuItems = new ArrayList<>();
        for (Object type : items) {
            RadioMenuItem menuItem = new RadioMenuItem(StringCapitalizer.capitalizeFirstLetter(type.toString().toLowerCase()));
            menuItem.setToggleGroup(group);
            menuItem.setUserData(type);
            menuItems.add(menuItem);
        }
        return menuItems;
    }

    private void cudaPaint(double... fractalSpecificParams) {
        long cudaStart = System.currentTimeMillis();
        painter.paint(red, green, blue, fractal, fractalSpecificParams);
        System.out.println("cudaPaint (" + (System.currentTimeMillis() - cudaStart) + "ms): " + fractal.getViewAsString());
        long paintStart = System.currentTimeMillis();
        paintImageOnCanvas();
        System.out.println("paintImageOnCanvas: " + (System.currentTimeMillis() - paintStart) + "ms");
    }

    private void releaseFractalPainter() {
        if (painter != null) {
            painter.destroy();
        }
    }

    private void drawFractal() {
        cudaPaint(getFractalParams());
    }

    private double[] getFractalParams() {
        return Match(fractal.type).of(
                Case($(is(FractalType.JULIA_SET)), new double[]{fractal.complexParamRe, fractal.complexParamIm}),
                Case($(is(FractalType.MANDELBROT_SET)), new double[0])
        );
    }

    private CudaPainter createFractalPainter() {
        Tuple2<String, String> kernelFileAndName = Match(fractal.type).of(
                Case($(is(FractalType.MANDELBROT_SET)), Tuple.of("/kernels/mandelbrotSet.ptx", "mandelbrotSet")),
                Case($(is(FractalType.JULIA_SET)), Tuple.of("/kernels/juliaSet.ptx", "juliaSet"))
        );
        return new CudaPainter(CANVAS_WIDTH, CANVAS_HEIGHT, kernelFileAndName._1, kernelFileAndName._2);
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
                patternPainter.diagonalStripes(x, y);
            }
        }
    }

    private void drawPattern() {
        loadPattern();
        paintImageOnCanvas();
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

    private void moveClickedFractalPointToCanvasCenter(double eventX, double eventY) {
        fractal.moveFractalPointToImageCenter(CANVAS_WIDTH, CANVAS_HEIGHT, eventX, eventY);
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

    private void openWindow(String filename, Object controller, String title) throws IOException {
        new WindowBuilder(filename)
                .withController(controller)
                .withTitle(title)
                .closedByEscapeKey(true)
                .withOnHiddenEventHandler(event -> drawFractal())
                .build().show();
    }
}
