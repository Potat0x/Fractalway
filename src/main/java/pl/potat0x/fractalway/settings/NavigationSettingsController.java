package pl.potat0x.fractalway.settings;

import javafx.fxml.FXML;
import javafx.scene.control.TextField;
import pl.potat0x.fractalway.fractal.Fractal;
import pl.potat0x.fractalway.utils.Action;
import pl.potat0x.fractalway.validation.DoubleValidator;

public class NavigationSettingsController extends BaseController {
    private final Fractal fractal;

    @FXML
    private TextField zoomField;
    @FXML
    private TextField positionXField;
    @FXML
    private TextField positionYField;
    @FXML
    private TextField zoomMultiplierField;
    @FXML
    private TextField positionStepField;

    public NavigationSettingsController(Fractal fractal, Action submitFormAction) {
        super(submitFormAction);
        this.fractal = fractal;
    }

    @FXML
    private void initialize() {
        super.initialize(zoomField);
        setCorrectnessWatchers();
        fillForm();
    }

    @FXML
    private void resetZoom() {
        fractal.resetZoom();
        fillForm();
        onFormSubmitted.execute();
    }

    @Override
    protected void setCorrectnessWatchers() {
        createFormCorrectnessWatcher().registerFields(new DoubleValidator(), zoomField, positionXField, positionYField, zoomMultiplierField, positionStepField);
    }

    @Override
    protected void readForm() {
        fractal.zoom = readDouble(zoomField);
        fractal.posX = readDouble(positionXField);
        fractal.posY = readDouble(positionYField);
        fractal.zoomMultiplier = readDouble(zoomMultiplierField);
        fractal.positionStep = readDouble(positionStepField);
    }

    @Override
    protected void fillForm() {
        setText(zoomField, fractal.zoom);
        setText(positionXField, fractal.posX);
        setText(positionYField, fractal.posY);
        setText(zoomMultiplierField, fractal.zoomMultiplier);
        setText(positionStepField, fractal.positionStep);
    }
}
