package app.settings;

import app.Fractal;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.TextField;

public class NavigationSettingsController extends BaseController {
    private final Fractal fractal;

    @FXML
    private TextField zoomField;
    @FXML
    private TextField positionXField;
    @FXML
    private TextField positionYField;
    @FXML
    private TextField zoomStepField;
    @FXML
    private TextField positionStepField;

    public NavigationSettingsController(Fractal fractal) {
        this.fractal = fractal;
    }

    @FXML
    void initialize() {
        super.initialize(zoomField);
        fillForm();
    }

    @FXML
    private void handleOkButton(ActionEvent actionEvent) {
        readForm();
        closeWindow();
    }

    private void readForm() {
        fractal.zoom = readDouble(zoomField);
        fractal.posX = readDouble(positionXField);
        fractal.posY = readDouble(positionYField);
        fractal.zoomStep = readDouble(zoomStepField);
        fractal.moveStep = readDouble(positionStepField);
    }

    private void fillForm() {
        assignDoubleToField(zoomField, fractal.zoom);
        assignDoubleToField(positionXField, fractal.posX);
        assignDoubleToField(positionYField, fractal.posY);
        assignDoubleToField(zoomStepField, fractal.zoomStep);
        assignDoubleToField(positionStepField, fractal.moveStep);
    }

    private void assignDoubleToField(TextField textField, double value) {
        textField.setText(Double.toString(value));
    }

    private double readDouble(TextField textField) {
        return Double.parseDouble(textField.getText());
    }
}
