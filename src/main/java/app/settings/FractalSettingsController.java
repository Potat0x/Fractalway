package app.settings;

import app.Fractal;
import javafx.fxml.FXML;
import javafx.scene.control.TextField;

public class FractalSettingsController extends BaseController {
    private final Fractal fractal;

    @FXML
    private TextField iterationsField;

    public FractalSettingsController(Fractal fractal) {
        this.fractal = fractal;
    }

    @FXML
    private void initialize() {
        super.initialize(iterationsField);
        iterationsField.setText(Integer.toString(fractal.maxIter));
    }

    @FXML
    private void handleOkButton() {
        fractal.maxIter = Integer.parseInt(iterationsField.getText());
        closeWindow();
    }
}
