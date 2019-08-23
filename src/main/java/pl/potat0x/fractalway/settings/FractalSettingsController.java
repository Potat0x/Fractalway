package pl.potat0x.fractalway.settings;

import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.layout.HBox;
import pl.potat0x.fractalway.fractal.Fractal;
import pl.potat0x.fractalway.fractal.FractalType;
import pl.potat0x.fractalway.utils.Action;
import pl.potat0x.fractalway.utils.Config;
import pl.potat0x.fractalway.utils.math.ParabolicScaleConverter;
import pl.potat0x.fractalway.validation.DoubleValidator;
import pl.potat0x.fractalway.validation.IntegerValidator;

public class FractalSettingsController extends BaseController {
    private final Fractal fractal;
    private final ParabolicScaleConverter sliderValueConverter;

    @FXML
    private TextField realPartField;
    @FXML
    private TextField imaginaryPartField;
    @FXML
    private TextField iterationsField;
    @FXML
    private HBox complexParamContainer;
    @FXML
    private Slider iterationsSlider;

    public FractalSettingsController(Fractal fractal, Action onFormSubmitted) {
        super(onFormSubmitted);
        this.sliderValueConverter = new ParabolicScaleConverter(Config.getInt("iterations-upper-limit"), Config.getDouble("iterations-slider-scale-exponent"));
        this.fractal = fractal;
    }

    @FXML
    private void initialize() {
        super.initialize(iterationsField);
        iterationsSlider.setMax(Config.getInt("iterations-upper-limit"));
        setCorrectnessWatchers();
        configureForm();
        fillForm();

        iterationsSlider.valueProperty().addListener((observable, oldValue, newValue) -> {
            updateFractalIterations(newValue.doubleValue());
            fillForm();
            onFormSubmitted.execute();
        });
    }

    @Override
    protected void setCorrectnessWatchers() {
        int lowerIterLimit = 1;
        createFormCorrectnessWatcher()
                .registerFields(new DoubleValidator(), realPartField, imaginaryPartField)
                .registerFields(new IntegerValidator(lowerIterLimit, Config.getInt("iterations-upper-limit")), iterationsField);
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
        fractal.iterations = readInteger(iterationsField);
        fractal.complexParamRe = readDouble(realPartField);
        fractal.complexParamIm = readDouble(imaginaryPartField);
        updateIterationsSliderPosition();
    }

    @Override
    protected void fillForm() {
        setText(iterationsField, fractal.iterations);
        setText(realPartField, fractal.complexParamRe);
        setText(imaginaryPartField, fractal.complexParamIm);
        updateIterationsSliderPosition();
    }

    private void updateFractalIterations(double x) {
        double iterations = Math.round(sliderValueConverter.linearToParabolic(x));
        fractal.iterations = (int) Math.max(iterations, 1);
    }

    private void updateIterationsSliderPosition() {
        iterationsSlider.setValue(sliderValueConverter.parabolicToLinear(fractal.iterations));
    }
}
