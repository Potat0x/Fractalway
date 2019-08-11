package app.settings;

import app.Fractal;
import app.FractalType;
import javafx.fxml.FXML;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;

public class FractalSettingsController extends BaseController {
    private final Fractal fractal;

    @FXML
    private TextField realPartField;
    @FXML
    private TextField imaginaryPartField;
    @FXML
    private TextField iterationsField;
    @FXML
    private HBox complexParamContainer;

    public FractalSettingsController(Fractal fractal) {
        this.fractal = fractal;
    }

    @FXML
    private void initialize() {
        super.initialize(iterationsField);
        configureForm();
        fillForm();
    }

    @FXML
    private void applyAndClose() {
        readForm();
        closeWindow();
    }

    private void configureForm() {
        if (fractal.type != FractalType.JULIA_SET) {
            complexParamContainer.setDisable(true);
            complexParamContainer.setVisible(false);
            complexParamContainer.setManaged(false);
        }
    }

    private void readForm() {
        fractal.maxIter = readInteger(iterationsField);
        fractal.complexParamRe = readDouble(realPartField);
        fractal.complexParamIm = readDouble(imaginaryPartField);
    }

    private void fillForm() {
        setText(realPartField, fractal.complexParamRe);
        setText(imaginaryPartField, fractal.complexParamIm);
        setText(iterationsField, fractal.maxIter);
    }
}
