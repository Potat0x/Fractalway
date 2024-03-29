package pl.potat0x.fractalway;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import io.vavr.control.Either;
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
import javafx.stage.FileChooser;
import javafx.stage.FileChooser.ExtensionFilter;
import pl.potat0x.fractalway.clock.Clock;
import pl.potat0x.fractalway.colorscheme.ArgbColorScheme;
import pl.potat0x.fractalway.colorscheme.ColorSchemeHistory;
import pl.potat0x.fractalway.fractal.Fractal;
import pl.potat0x.fractalway.fractal.FractalType;
import pl.potat0x.fractalway.fractalpainter.CpuPainter;
import pl.potat0x.fractalway.fractalpainter.CudaPainter;
import pl.potat0x.fractalway.fractalpainter.FractalPainter;
import pl.potat0x.fractalway.fractalpainter.FractalPainterDevice;
import pl.potat0x.fractalway.settings.ColorSchemeSettingsController;
import pl.potat0x.fractalway.settings.FractalSettingsController;
import pl.potat0x.fractalway.settings.NavigationSettingsController;
import pl.potat0x.fractalway.utils.Action;
import pl.potat0x.fractalway.utils.ArrayToImageWriter;
import pl.potat0x.fractalway.utils.Config;
import pl.potat0x.fractalway.utils.StringCapitalizer;
import pl.potat0x.fractalway.utils.WindowBuilder;
import pl.potat0x.fractalway.utils.device.CpuInfo;
import pl.potat0x.fractalway.utils.device.CudaDeviceInfo;
import pl.potat0x.fractalway.utils.math.ParabolicScaleConverter;

import java.awt.AWTException;
import java.awt.Robot;
import java.io.File;
import java.io.IOException;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

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

    private Fractal prevFractal;
    private ArgbColorScheme prevColorScheme;

    private Fractal fractal;
    private FractalPainter painter;
    private DecimalFormat decimalFormat;
    private final CpuInfo cpuInfo;

    private ToggleGroup deviceGroup;
    private String imageDialogLastDirectory;
    private final ArgbColorScheme colorScheme;
    private final ColorSchemeHistory colorSchemeHistory;
    private final ParabolicScaleConverter sliderValueConverter;

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
    private Label timeInfoLabel;
    @FXML
    private Label canvasSizeInfoLabel;
    @FXML
    private Label iterationsInfoLabel;
    @FXML
    private Slider iterationsSlider;
    @FXML
    private CheckMenuItem iterationsSliderMenuItem;
    @FXML
    private CheckMenuItem deviceInfoMenuItem;
    @FXML
    private CheckMenuItem timeInfoMenuItem;

    public MainController() {
        canvasWidth = Config.getInt("canvas-width");
        canvasHeight = Config.getInt("canvas-height");
        initImageArray();
        initDecimalFormatter();
        fractal = new Fractal(FractalType.MANDELBROT_SET, Config.getInt("iterations-upper-limit"));
        cpuInfo = new CpuInfo();
        colorScheme = new ArgbColorScheme();
        colorSchemeHistory = new ColorSchemeHistory(colorScheme);
        colorSchemeHistory.addToHistory(colorScheme);
        sliderValueConverter = new ParabolicScaleConverter(Config.getInt("iterations-upper-limit"), Config.getDouble("main-iterations-slider-scale-exp"));
    }

    @FXML
    private void initialize() {
        setBackgroundImage();
        initCanvas();
        initDeviceMenu();
        initFractalMenu();
        initCursorMenu();
        initDeviceInfoMenuItem();
        initTimeInfoMenuItem();
        initResizeEventHandler();
        refreshCanvasSizeInfoLabel();
        initIterationsSliderAndLabel();
        initIterationsSliderMenuItem();
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
    private void handleCanvasScrollEvent(ScrollEvent scrollEvent) {
        if (scrollEvent.isShiftDown()) {
            int repetitions = scrollEvent.isControlDown() ? Config.getInt("iterations-fast-change-step") : 1;
            changeFractalIterations(scrollEvent.getDeltaX() > 0, repetitions);
            return;
        }

        if (scrollEvent.isControlDown()) {
            moveMouseToCanvasCenter();
            moveClickedFractalPointToCanvasCenter(scrollEvent.getX(), scrollEvent.getY());
        }

        changeFractalZoom(scrollEvent.getDeltaY() > 0);
        drawFractal();
    }

    private void changeFractalIterations(boolean increase, int changeBy) {
        if (increase) {
            fractal.increaseIterations(changeBy);
        } else {
            fractal.decreaseIterations(changeBy);
        }
        updateIterationsSlider();
    }

    private void changeFractalZoom(boolean increase) {
        if (increase) {
            fractal.zoomIn();
        } else {
            fractal.zoomOut();
        }
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
                Case($(is(KeyCode.C).or(is(KeyCode.ADD))), o -> run(() -> changeFractalIterations(true, 1))),
                Case($(is(KeyCode.Z).or(is(KeyCode.SUBTRACT))), o -> run(() -> changeFractalIterations(false, 1))),
                Case($(is(KeyCode.S)), o -> run(() -> {
                    if (keyEvent.isControlDown()) {
                        saveAsImage();
                    }
                }))
        ).peek(x -> drawFractal());
    }

    @FXML
    private void showNavigationSettingsWindow(ActionEvent actionEvent) throws IOException {
        openWindow("/fxml/navigation_settings.fxml", new NavigationSettingsController(fractal, this::drawFractal), "Navigation settings");
    }

    @FXML
    private void showFractalSettingsWindow(ActionEvent actionEvent) throws IOException {
        Action drawFractal = () -> {
            updateIterationsSlider();
            drawFractal();
        };
        openWindow("/fxml/fractal_settings.fxml", new FractalSettingsController(fractal, drawFractal), "Fractal settings");
    }

    @FXML
    private void showColorSchemeSettingsWindow(ActionEvent actionEvent) throws IOException {
        openWindow("/fxml/color_settings.fxml", new ColorSchemeSettingsController(colorScheme, colorSchemeHistory, this::drawFractal), "Color scheme");
    }

    @FXML
    private void saveAsImage() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save as image");
        fileChooser.getExtensionFilters().addAll(createExtensionFilters());

        if (imageDialogLastDirectory != null) {
            fileChooser.setInitialDirectory(new File(imageDialogLastDirectory));
        }

        File file = fileChooser.showSaveDialog(canvas.getScene().getWindow());
        if (file != null) {
            String fileExtension = fileChooser.getSelectedExtensionFilter().getDescription();
            Either<String, Void> savingResult = new ArrayToImageWriter().saveImage(argb, canvasWidth, canvasHeight, file, fileExtension);
            if (savingResult.isRight()) {
                imageDialogLastDirectory = file.getParent();
            }
            showSavingResultAlert(savingResult, file.getPath());
        }
    }

    private void initIterationsSliderMenuItem() {
        iterationsSliderMenuItem.selectedProperty().addListener((observable, oldValue, newValue) -> {
            iterationsSlider.setVisible(newValue);
            iterationsInfoLabel.setVisible(newValue);
        });
        iterationsSliderMenuItem.setSelected(true);
    }

    private void initIterationsSliderAndLabel() {
        int iterLimit = Config.getInt("iterations-upper-limit");
        iterationsSlider.setMax(iterLimit);
        iterationsSlider.setMajorTickUnit(iterLimit);
        updateIterationsSlider();
        updateIterationsInfoLabel();

        iterationsSlider.valueProperty().addListener((observable, oldValue, newValue) -> {
            if (sliderValueConverter.checkIfParabolicValuesAreDifferentAsIntegers(oldValue.doubleValue(), newValue.doubleValue())) {
                updateFractalIterations(newValue.doubleValue());
                updateIterationsInfoLabel();
                drawFractal();
            }
            canvas.requestFocus();
        });
    }

    private void updateIterationsInfoLabel() {
        iterationsInfoLabel.setText(String.valueOf(fractal.iterations));
    }

    private void updateFractalIterations(double x) {
        double iterations = Math.round(sliderValueConverter.linearToParabolic(x));
        fractal.iterations = (int) Math.max(iterations, 1);
    }

    private void updateIterationsSlider() {
        iterationsSlider.setValue(sliderValueConverter.parabolicToLinear(fractal.iterations));
    }

    private void showSavingResultAlert(Either<String, Void> savingResult, String filePath) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION, "Image saved to\n" + filePath, ButtonType.OK);
        if (savingResult.isLeft()) {
            alert.setAlertType(Alert.AlertType.ERROR);
            alert.setContentText("Saving image to\n" + filePath + "\nfailed.\n" + savingResult.getLeft() + ".");
        }
        alert.setTitle("Fractalway");
        alert.setHeaderText(null);
        alert.showAndWait();
    }

    private List<ExtensionFilter> createExtensionFilters() {
        List<String> extensions = Arrays.asList("png", "gif", "jpeg", "bmp");
        return extensions.stream().map(e ->
                new ExtensionFilter(e.toUpperCase(), "*." + e.toLowerCase())
        ).collect(Collectors.toList());
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
            fractal = new Fractal(newType, Config.getInt("iterations-upper-limit"));
            releaseFractalPainter();
            painter = createFractalPainter();
            drawFractal();
        });

        selectFirstItemInToggleGroup(fractalGroup);
    }

    private void selectFirstItemInToggleGroup(ToggleGroup group) {
        group.getToggles().get(0).setSelected(true);
    }

    private void initDeviceMenu() {
        deviceGroup = new ToggleGroup();
        deviceMenu.getItems().add(createRadioItemInToggleGroup(deviceGroup, "CPU", FractalPainterDevice.CPU));

        int cudaDeviceCount = CudaDeviceInfo.getDeviceCount();
        System.out.println("Found CUDA devices: " + cudaDeviceCount);
        if (cudaDeviceCount > 0) {
            deviceMenu.getItems().add(createRadioItemInToggleGroup(deviceGroup, "CUDA GPU", FractalPainterDevice.CUDA_GPU));
        }

        deviceGroup.selectedToggleProperty().addListener((observable, oldValue, newValue) -> {
            initDeviceInfoLabel(getCurrentDeviceType());
            releaseFractalPainter();
            painter = createFractalPainter();
            drawFractal(true);
        });

        selectLastItemInToggleGroup(deviceGroup);
    }

    private void selectLastItemInToggleGroup(ToggleGroup group) {
        group.getToggles().get(group.getToggles().size() - 1).setSelected(true);
    }

    private void initDeviceInfoLabel(FractalPainterDevice deviceType) {
        String labelText = createDeviceInfoText(deviceType);
        deviceInfoLabel.setText(labelText);
    }

    private String createDeviceInfoText(FractalPainterDevice deviceType) {
        if (deviceType == FractalPainterDevice.CUDA_GPU) {
            return cudaDeviceInfoText();
        }
        return cpuDeviceInfoText();
    }

    private String cudaDeviceInfoText() {
        CudaDeviceInfo devInfo = new CudaDeviceInfo(0);
        return devInfo.name() +
                "\nCUDA " + devInfo.cudaVersion() +
                "\nCompute capability " + devInfo.computeCapability();
    }

    private String cpuDeviceInfoText() {
        return cpuInfo.cpu.getName() + "\n" +
                cpuInfo.cpu.getPhysicalProcessorCount() + " cores, " +
                cpuInfo.cpu.getLogicalProcessorCount() + " threads";
    }

    private void initDeviceInfoMenuItem() {
        deviceInfoMenuItem.selectedProperty().addListener((observable, oldValue, newValue) -> deviceInfoLabel.setVisible(newValue));
        deviceInfoMenuItem.selectedProperty().set(true);
    }

    private void initTimeInfoMenuItem() {
        timeInfoMenuItem.selectedProperty().addListener((observable, oldValue, newValue) -> {
            timeInfoLabel.setVisible(newValue);
            canvasSizeInfoLabel.setVisible(newValue);
        });
        timeInfoMenuItem.selectedProperty().set(true);
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
            refreshCanvasSizeInfoLabel();
            initImageArray();
            initCanvas();
            releaseFractalPainter();
            painter = createFractalPainter();
            drawFractal(true);
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
        Tuple2<Float, Float> timeInfo = painter.paint(argb, fractal);
        System.out.println("paintFractal (calc: " + timeInfo._1 + " ms, memcpy: " + timeInfo._2 + " ms, total: " + (timeInfo._1 + timeInfo._2) + " ms)"
                + "\n\t" + fractal.getViewAsString());
        return Tuple.of(timeInfo._1, timeInfo._2);
    }

    private void refreshTimeInfoLabel(Tuple2<Float, Float> timeInfo) {
        String text = createTimeInfoText(timeInfo);
        timeInfoLabel.setText(text);
    }

    private void refreshCanvasSizeInfoLabel() {
        canvasSizeInfoLabel.setText(canvasWidth + "x" + canvasHeight + " px");
    }

    private String createTimeInfoText(Tuple2<Float, Float> timeInfo) {
        if (getCurrentDeviceType() == FractalPainterDevice.CUDA_GPU) {
            return gpuTimeInfoText(timeInfo);
        }
        return cpuTimeInfoText(timeInfo);
    }

    private String gpuTimeInfoText(Tuple2<Float, Float> timeInfo) {
        return "Kernel: " + decimalFormat.format(timeInfo._1) + " ms" +
                "\nMemcpy: " + decimalFormat.format(timeInfo._2) + " ms" +
                "\nTotal: " + decimalFormat.format(timeInfo._1 + timeInfo._2) + " ms";
    }

    private String cpuTimeInfoText(Tuple2<Float, Float> timeInfo) {
        return decimalFormat.format(timeInfo._1) + " ms";
    }

    private void releaseFractalPainter() {
        if (painter != null) {
            painter.destroy();
        }
    }

    private void drawFractal(boolean force) {
        if (force || isFractalAndColorSchemeDifferentThanPrevious()) {
            Tuple2<Float, Float> timeInfo = paintFractal();
            Clock clock = new Clock();
            paintImageOnCanvas();
            System.out.println("paintImageOnCanvas: " + clock.getElapsedTime() + " ms (" + canvasWidth + "x" + canvasHeight + " px)");
            refreshTimeInfoLabel(timeInfo);
            saveCurrentFractalAndColorSchemeAsPrevious();
            System.out.println("FRACTAL PAINTED:");
            System.out.println(colorScheme);
        }
    }

    private void drawFractal() {
        drawFractal(false);
    }

    private boolean isFractalAndColorSchemeDifferentThanPrevious() {
        return !(fractal.equals(prevFractal) && colorScheme.equals(prevColorScheme));
    }

    private void saveCurrentFractalAndColorSchemeAsPrevious() {
        prevFractal = fractal.copy();
        prevColorScheme = colorScheme.copy();
    }

    private FractalPainterDevice getCurrentDeviceType() {
        return (FractalPainterDevice) deviceGroup.getSelectedToggle().getUserData();
    }

    private FractalPainter createFractalPainter() {
        if (getCurrentDeviceType() == FractalPainterDevice.CUDA_GPU) {
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
        return new CudaPainter(canvasWidth, canvasHeight, kernelFileAndName._1, kernelFileAndName._2, Config.getBoolean("cuda-use-pinned-memory"));
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

    private int createColorArgb(int x, int y) {
        int index = calculateIndex(x, y);

        ArgbColorScheme cs = colorScheme;
        int r = (argb[index] << cs.redLeftShift) >>> cs.redRightShift;
        int g = (argb[index] << cs.greenLeftShift) >>> cs.greenRightShift;
        int b = (argb[index] << cs.blueLeftShift) >>> cs.blueRightShift;

        r = cs.redLeftMultiplication ? r * cs.redLeftShift : r;
        g = cs.greenLeftMultiplication ? g * cs.greenLeftShift : g;
        b = cs.blueLeftMultiplication ? b * cs.blueLeftShift : b;

        r = cs.redRightMultiplication ? r * cs.redRightShift : r;
        g = cs.greenRightMultiplication ? g * cs.greenRightShift : g;
        b = cs.blueRightMultiplication ? b * cs.blueRightShift : b;

        if (colorScheme.invertColors) {
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
