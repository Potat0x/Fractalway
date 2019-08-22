package pl.potat0x.fractalway.settings;

import javafx.fxml.FXML;
import javafx.scene.control.*;
import pl.potat0x.fractalway.fractal.ArgbColorScheme;
import pl.potat0x.fractalway.utils.Action;

import java.util.function.Consumer;

public class ColorSchemeSettingsController extends BaseController {

    private final ArgbColorScheme colorScheme;

    @FXML
    private Slider redLeft;
    @FXML
    private Slider redRight;
    @FXML
    private Slider greenLeft;
    @FXML
    private Slider greenRight;
    @FXML
    private Slider blueLeft;
    @FXML
    private Slider blueRight;

    public ColorSchemeSettingsController(ArgbColorScheme colorScheme, Action onFormSubmitted) {
        super(onFormSubmitted);
        this.colorScheme = colorScheme;
    }

    @FXML
    private void initialize() {
        fillForm();
        initValueListeners();
    }

    @Override
    protected void fillForm() {
        setSliderValue(redLeft, colorScheme.rLeftShift);
        setSliderValue(redRight, colorScheme.rRightShift);
        setSliderValue(greenLeft, colorScheme.gLeftShift);
        setSliderValue(greenRight, colorScheme.gRightShift);
        setSliderValue(blueLeft, colorScheme.bLeftShift);
        setSliderValue(blueRight, colorScheme.bRightShift);
    }

    @Override
    protected void readForm() {
    }

    @Override
    protected void setCorrectnessWatchers() {
    }

    private void initValueListeners() {
        addListenerToSlider(redLeft, newVal -> colorScheme.rLeftShift = newVal);
        addListenerToSlider(redRight, newVal -> colorScheme.rRightShift = newVal);
        addListenerToSlider(greenLeft, newVal -> colorScheme.gLeftShift = newVal);
        addListenerToSlider(greenRight, newVal -> colorScheme.gRightShift = newVal);
        addListenerToSlider(blueLeft, newVal -> colorScheme.bLeftShift = newVal);
        addListenerToSlider(blueRight, newVal -> colorScheme.bRightShift = newVal);
    }

    private void addListenerToSlider(Slider slider, Consumer<Integer> consumer) {
        slider.valueProperty().addListener((observable, oldValue, newValue) -> {
            consumer.accept(newValue.intValue());
            onFormSubmitted.execute();
        });
    }

    private void setSliderValue(Slider slider, int value) {
        slider.setValue(value);
    }
}
