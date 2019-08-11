package pl.potat0x.fractalway.settings;

import pl.potat0x.fractalway.fractal.Fractal;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.TextField;
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

    public NavigationSettingsController(Fractal fractal) {
        this.fractal = fractal;
    }

    @FXML
    void initialize() {
        super.initialize(zoomField);
        setCorrectnessWatchers();
        fillForm();
    }

    @FXML
    private void applyAndClose(ActionEvent actionEvent) {
        if (isSubmittingUnblocked()) {
            readForm();
            closeWindow();
        }
    }

    private void setCorrectnessWatchers() {
        createFormCorrectnessWatcher().registerFields(new DoubleValidator(), zoomField, positionXField, positionYField, zoomMultiplierField, positionStepField);
    }

    private void readForm() {
        fractal.zoom = readDouble(zoomField);
        fractal.posX = readDouble(positionXField);
        fractal.posY = readDouble(positionYField);
        fractal.zoomMultiplier = readDouble(zoomMultiplierField);
        fractal.positionStep = readDouble(positionStepField);
    }

    private void fillForm() {
        setText(zoomField, fractal.zoom);
        setText(positionXField, fractal.posX);
        setText(positionYField, fractal.posY);
        setText(zoomMultiplierField, fractal.zoomMultiplier);
        setText(positionStepField, fractal.positionStep);
    }
}
