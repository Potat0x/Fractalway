package pl.potat0x.fractalway;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import javafx.application.Platform;
import javafx.beans.value.ChangeListener;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.Point2D;
import javafx.scene.Cursor;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelWriter;
import javafx.scene.input.*;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundImage;
import javafx.scene.layout.GridPane;
import pl.potat0x.fractalway.clock.Clock;
import pl.potat0x.fractalway.fractal.Fractal;
import pl.potat0x.fractalway.fractal.FractalType;
import pl.potat0x.fractalway.fractalpainter.CpuPainter;
import pl.potat0x.fractalway.fractalpainter.CudaPainter;
import pl.potat0x.fractalway.fractalpainter.FractalPainter;
import pl.potat0x.fractalway.fractalpainter.FractalPainterDevice;
import pl.potat0x.fractalway.settings.FractalSettingsController;
import pl.potat0x.fractalway.settings.NavigationSettingsController;
import pl.potat0x.fractalway.utils.CudaDeviceInfo;
import pl.potat0x.fractalway.utils.PatternPainter;
import pl.potat0x.fractalway.utils.StringCapitalizer;
import pl.potat0x.fractalway.utils.WindowBuilder;

import java.awt.AWTException;
import java.awt.Robot;
import java.io.IOException;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import static io.vavr.API.*;
import static io.vavr.Predicates.is;
import static io.vavr.Predicates.isIn;
import static javafx.scene.layout.BackgroundPosition.CENTER;
import static javafx.scene.layout.BackgroundRepeat.REPEAT;
import static javafx.scene.layout.BackgroundSize.DEFAULT;

public class MainController {
    private int canvasWidth;
    private int canvasHeight;
    private int[] argb;

    private boolean windowResized;
    private int newCanvasWidth;
    private int newCanvasHeight;

    private PatternPainter patternPainter;
    private Fractal fractal;
    private FractalPainter painter;
    private boolean invertFractalColors;
    private DecimalFormat decimalFormat;

    private ToggleGroup deviceGroup;

    @FXML
    private GridPane mainPane;
    @FXML
    private Canvas canvas;
    @FXML
    private Menu fractalMenu;
    @FXML
    private Menu deviceMenu;
    @FXML
    private Menu settingsMenu;
    @FXML
    private Menu canvasCursorMenu;
    @FXML
    private Label deviceInfoLabel;
    @FXML
    private Label eventInfoLabel;
    @FXML
    private CheckMenuItem deviceInfoMenuItem;
    @FXML
    private CheckMenuItem invertColorsMenuItem;
    @FXML
    private CheckMenuItem eventInfoMenuItem;

    public MainController() {
        canvasWidth = 820;
        canvasHeight = 620;
        initImageArray();
        initDecimalFormatter();
        fractal = new Fractal(FractalType.MANDELBROT_SET);
        patternPainter = new PatternPainter(canvasWidth, argb);
        invertFractalColors = false;
    }

    @FXML
    private void initialize() {
        setBackgroundImage();
        initCanvas();
        initDeviceMenu();
        drawPattern();
        initFractalMenu();
        initCursorMenu();
        initDeviceInfoLabel();
        initDeviceInfoMenuItem();
        initEventInfoMenuItem();
        initInvertColorsMenuItem();
        initResizeEventHandler();
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

    private void initImageArray() {
        argb = new int[canvasWidth * canvasHeight];
    }

    private void setBackgroundImage() {
        Image image = new Image("file:src/main/resources/images/background.png");
        mainPane.setBackground(new Background(new BackgroundImage(image, REPEAT, REPEAT, CENTER, DEFAULT)));
    }

    private void initCanvas() {
        canvas.setWidth(canvasWidth);
        canvas.setHeight(canvasHeight);
        canvas.setFocusTraversable(true);
    }

    private void initCursorMenu() {
        ToggleGroup cursorGroup = new ToggleGroup();
        List<MenuItem> menuItems = createRadioItemsInToggleGroup(cursorGroup, Cursor.DEFAULT, Cursor.CROSSHAIR, Cursor.NONE);
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
        List<MenuItem> menuItems = createRadioItemsInToggleGroup(fractalGroup, (Object[]) FractalType.values());
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

    private void initDeviceMenu() {
        deviceGroup = new ToggleGroup();
        deviceMenu.getItems().addAll(
                createRadioItemInToggleGroup(deviceGroup, "CPU", FractalPainterDevice.CPU),
                createRadioItemInToggleGroup(deviceGroup, "CUDA GPU", FractalPainterDevice.CUDA_GPU)
        );
        deviceGroup.selectedToggleProperty().addListener((observable, oldValue, newValue) -> {
            painter = createFractalPainter();
            drawFractal();
        });
        deviceGroup.getToggles().get(1).setSelected(true);
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

    private void initEventInfoMenuItem() {
        eventInfoMenuItem.selectedProperty().addListener((observable, oldValue, newValue) -> eventInfoLabel.setVisible(newValue));
        eventInfoMenuItem.selectedProperty().set(true);
    }

    private void initInvertColorsMenuItem() {
        invertColorsMenuItem.selectedProperty().addListener((observable, oldValue, newValue) -> {
            invertFractalColors = newValue;
            drawFractal();
        });
    }

    private void initResizeEventHandler() {
        ChangeListener<Number> resizeListener = (observable, oldValue, newValue) -> {
            newCanvasWidth = new Double(canvas.getScene().getWidth()).intValue();
            newCanvasHeight = new Double(canvas.getScene().getHeight()).intValue();
            windowResized = true;
        };

        mainPane.widthProperty().addListener(resizeListener);
        mainPane.heightProperty().addListener(resizeListener);
        mainPane.setOnMouseEntered(event -> resize());
    }

    private void resize() {
        if (windowResized) {
            canvasWidth = newCanvasWidth;
            canvasHeight = newCanvasHeight;
            initImageArray();
            initCanvas();
            patternPainter = new PatternPainter(canvasWidth, argb);
            releaseFractalPainter();
            painter = createFractalPainter();
            drawFractal();
            windowResized = false;
        }
    }

    private void initDecimalFormatter() {
        decimalFormat = new DecimalFormat("#.##");
        decimalFormat.setRoundingMode(RoundingMode.HALF_UP);
        decimalFormat.setMinimumFractionDigits(2);
    }

    private List<MenuItem> createRadioItemsInToggleGroup(ToggleGroup group, Object... items) {
        List<MenuItem> menuItems = new ArrayList<>();
        for (Object type : items) {
            RadioMenuItem item = createRadioItemInToggleGroup(group, StringCapitalizer.capitalizeFirstLetter(type.toString().toLowerCase()), type);
            menuItems.add(item);
        }
        return menuItems;
    }

    private RadioMenuItem createRadioItemInToggleGroup(ToggleGroup group, String text, Object userData) {
        RadioMenuItem item = new RadioMenuItem(text);
        item.setToggleGroup(group);
        item.setUserData(userData);
        return item;
    }

    private Tuple2<Float, Float> paintFractal() {
        Tuple2<Float, Float> eventInfo = painter.paint(argb, fractal);
        System.out.println("paintFractal (calc: " + eventInfo._1 + " ms, memcpy: " + eventInfo._2 + " ms, total: " + (eventInfo._1 + eventInfo._2) + " ms"
                + "\n\t" + fractal.getViewAsString());
        return Tuple.of(eventInfo._1, eventInfo._2);
    }

    private void refreshEventLabel(Tuple2<Float, Float> paintTimeInfo) {
        String text = "Kernel time: " + decimalFormat.format(paintTimeInfo._1) + " ms" +
                "\nMemcpy time: " + decimalFormat.format(paintTimeInfo._2) + " ms" +
                "\nTotal: " + decimalFormat.format(paintTimeInfo._1 + paintTimeInfo._2) + " ms";
        eventInfoLabel.setText(text);
    }

    private void releaseFractalPainter() {
        if (painter != null) {
            painter.destroy();
        }
    }

    private void drawFractal() {
        Tuple2<Float, Float> timeInfo = paintFractal();
        Clock clock = new Clock();
        paintImageOnCanvas();
        System.out.println("paintImageOnCanvas: " + clock.getElapsedTime() + " ms");
        refreshEventLabel(timeInfo);
    }

    private FractalPainterDevice getCurrentDeviceType() {
        return (FractalPainterDevice) deviceGroup.getSelectedToggle().getUserData();
    }

    private FractalPainter createFractalPainter() {
        if (getCurrentDeviceType().equals(FractalPainterDevice.CUDA_GPU)) {
            return createCudaFractalPainter();
        }
        return createCpuFractalPainter();
    }

    private FractalPainter createCpuFractalPainter() {
        return new CpuPainter(canvasWidth, canvasHeight);
    }

    private CudaPainter createCudaFractalPainter() {
        Tuple2<String, String> kernelFileAndName = Match(fractal.type).of(
                Case($(is(FractalType.MANDELBROT_SET)), Tuple.of("/kernels/mandelbrotSet.ptx", "mandelbrotSet")),
                Case($(is(FractalType.JULIA_SET)), Tuple.of("/kernels/juliaSet.ptx", "juliaSet")),
                Case($(is(FractalType.BURNING_SHIP)), Tuple.of("/kernels/burningShip.ptx", "burningShip"))
        );
        return new CudaPainter(canvasWidth, canvasHeight, kernelFileAndName._1, kernelFileAndName._2);
    }

    private void paintImageOnCanvas() {
        for (int y = 0; y < canvasHeight; y++) {
            for (int x = 0; x < canvasWidth; x++) {
                argb[y * canvasWidth + x] = createColorArgb(x, y);
            }
        }
        PixelWriter pixelWriter = canvas.getGraphicsContext2D().getPixelWriter();
        pixelWriter.setPixels(0, 0, canvasWidth, canvasHeight, PixelFormat.getIntArgbPreInstance(), argb, 0, canvasWidth);
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

    private int createColorArgb(int x, int y) {
        int index = calculateIndex(x, y);
        int r = (argb[index]) >>> 16;
        int g = (argb[index] << 8) >>> 16;
        int b = (argb[index] << 16) >>> 16;

        if (invertFractalColors) {
            r = 255 - r;
            g = 255 - g;
            b = 255 - b;
        }

        return (255 << 24) | (r << 16) | (g << 8) | b;
    }

    private int calculateIndex(int x, int y) {
        return canvasWidth * y + x;
    }

    private void moveClickedFractalPointToCanvasCenter(double eventX, double eventY) {
        fractal.moveFractalPointToImageCenter(canvasWidth, canvasHeight, eventX, eventY);
    }

    private void moveMouseToCanvasCenter() {
        Point2D point2D = canvas.localToScreen(canvasWidth / 2.0, canvasHeight / 2.0);
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
