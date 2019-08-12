package pl.potat0x.fractalway.settings;

import pl.potat0x.fractalway.fractal.Fractal;
import pl.potat0x.fractalway.fractal.FractalType;
import javafx.fxml.FXML;
import javafx.scene.control.TextField;
import javafx.scene.layout.HBox;
import pl.potat0x.fractalway.utils.Action;
import pl.potat0x.fractalway.validation.DoubleValidator;
import pl.potat0x.fractalway.validation.IntegerValidator;

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

    public FractalSettingsController(Fractal fractal, Action setOnFormSubmitted) {
        super(setOnFormSubmitted);
        this.fractal = fractal;
    }

    @FXML
    private void initialize() {
        super.initialize(iterationsField);
        setCorrectnessWatchers();
        configureForm();
        fillForm();
    }

    @Override
    protected void setCorrectnessWatchers() {
        createFormCorrectnessWatcher()
                .registerFields(new DoubleValidator(), realPartField, imaginaryPartField)
                .registerFields(new IntegerValidator(), iterationsField);
    }

    private void configureForm() {
        if (fractal.type != FractalType.JULIA_SET) {
            complexParamContainer.setDisable(true);
            complexParamContainer.setVisible(false);
            complexParamContainer.setManaged(false);
        }
    }

    @Override
    protected void readForm() {
        fractal.maxIter = readInteger(iterationsField);
        fractal.complexParamRe = readDouble(realPartField);
        fractal.complexParamIm = readDouble(imaginaryPartField);
    }

    @Override
    protected void fillForm() {
        setText(realPartField, fractal.complexParamRe);
        setText(imaginaryPartField, fractal.complexParamIm);
        setText(iterationsField, fractal.maxIter);
    }
}
