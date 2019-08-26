package pl.potat0x.fractalway.settings;

import javafx.fxml.FXML;
import javafx.scene.control.*;
import pl.potat0x.fractalway.fractal.ArgbColorScheme;
import pl.potat0x.fractalway.utils.Action;

import java.util.LinkedList;
import java.util.List;
import java.util.function.Consumer;

public class ColorSchemeSettingsController extends BaseController {

    private final ArgbColorScheme colorScheme;
    private final List<ArgbColorScheme> colorSchemeHistory;

    @FXML
    private CheckBox randomBitshift;

    @FXML
    private Slider redLeft, redRight;
    @FXML
    private Slider greenLeft, greenRight;
    @FXML
    private Slider blueLeft, blueRight;

    @FXML
    private Pagination colorSchemeHistoryPagin;
    @FXML
    private CheckBox randomLeftMultiplication, randomRightMultiplication;

    @FXML
    private CheckBox redLeftMultiplication, redRightMultiplication;
    @FXML
    private CheckBox greenLeftMultiplication, greenRightMultiplication;
    @FXML
    private CheckBox blueLeftMultiplication, blueRightMultiplication;

    /*
    Do not repaint fractal multiple times while updating multiple controls
    */
    private boolean listenersUnlocked = true;

    public ColorSchemeSettingsController(ArgbColorScheme colorScheme, Action onFormSubmitted) {
        super(onFormSubmitted);
        this.colorScheme = colorScheme;
        colorSchemeHistory = new LinkedList<>();
        colorSchemeHistory.add(colorScheme.copy());
    }

    @FXML
    private void initialize() {
        fillForm();
        initValueListeners();
        colorSchemeHistoryPagin.currentPageIndexProperty().addListener((observable, oldValue, newValue) -> {
            restoreColorSchemeFromHistory(newValue.intValue());
            fillForm();
            onFormSubmitted.execute();
        });
    }

    @Override
    protected void fillForm() {
        setSliderValue(redLeft, colorScheme.redLeftShift);
        setSliderValue(redRight, colorScheme.redRightShift);
        setSliderValue(greenLeft, colorScheme.greenLeftShift);
        setSliderValue(greenRight, colorScheme.greenRightShift);
        setSliderValue(blueLeft, colorScheme.blueLeftShift);
        setSliderValue(blueRight, colorScheme.blueRightShift);

        setCheckBoxValue(redLeftMultiplication, colorScheme.redLeftMultiplication);
        setCheckBoxValue(redRightMultiplication, colorScheme.redRightMultiplication);
        setCheckBoxValue(greenLeftMultiplication, colorScheme.greenLeftMultiplication);
        setCheckBoxValue(greenRightMultiplication, colorScheme.greenRightMultiplication);
        setCheckBoxValue(blueLeftMultiplication, colorScheme.blueLeftMultiplication);
        setCheckBoxValue(blueRightMultiplication, colorScheme.blueRightMultiplication);
    }

    @FXML
    private void createRandomColorScheme() {
        updateCurrentItemInHistory();
        addNewRandomColorSchemeToHistory();
        listenersUnlocked = false;
        updateHistoryPaginPageCount();
        selectLastPaginItem();
        listenersUnlocked = true;
        onFormSubmitted.execute();
    }

    @FXML
    private void resetCurrentColorSchemeToDefault() {
        colorScheme.assignValues(new ArgbColorScheme());
        updateCurrentItemInHistory();
        listenersUnlocked = false;
        fillForm();
        listenersUnlocked = true;
        onFormSubmitted.execute();
    }

    private void updateHistoryPaginPageCount() {
        colorSchemeHistoryPagin.setPageCount(colorSchemeHistory.size());
    }

    private void addNewRandomColorSchemeToHistory() {
        ArgbColorScheme newColorScheme = new ArgbColorScheme();
        newColorScheme.random(randomBitshift.isSelected(), randomLeftMultiplication.isSelected(), randomRightMultiplication.isSelected());
        colorSchemeHistory.add(newColorScheme);
    }

    private void updateCurrentItemInHistory() {
        colorSchemeHistory.get(colorSchemeHistoryPagin.getCurrentPageIndex()).assignValues(colorScheme);
    }

    private void restoreColorSchemeFromHistory(int historyItemId) {
        ArgbColorScheme selectedScheme = colorSchemeHistory.get(historyItemId);
        colorScheme.assignValues(selectedScheme);
    }

    private void selectLastPaginItem() {
        int pageCount = colorSchemeHistoryPagin.getPageCount();
        if (pageCount > 0) {
            colorSchemeHistoryPagin.setCurrentPageIndex(pageCount - 1);
        }
    }

    @Override
    protected void readForm() {
    }

    @Override
    protected void setCorrectnessWatchers() {
    }

    private void initValueListeners() {
        addListenerToSlider(redLeft, newVal -> colorScheme.redLeftShift = newVal);
        addListenerToSlider(redRight, newVal -> colorScheme.redRightShift = newVal);
        addListenerToSlider(greenLeft, newVal -> colorScheme.greenLeftShift = newVal);
        addListenerToSlider(greenRight, newVal -> colorScheme.greenRightShift = newVal);
        addListenerToSlider(blueLeft, newVal -> colorScheme.blueLeftShift = newVal);
        addListenerToSlider(blueRight, newVal -> colorScheme.blueRightShift = newVal);
        addListenerToSlider(blueRight, newVal -> colorScheme.blueRightShift = newVal);

        addListenerToCheckBox(redLeftMultiplication, newVal -> colorScheme.redLeftMultiplication = newVal);
        addListenerToCheckBox(redRightMultiplication, newVal -> colorScheme.redRightMultiplication = newVal);
        addListenerToCheckBox(greenLeftMultiplication, newVal -> colorScheme.greenLeftMultiplication = newVal);
        addListenerToCheckBox(greenRightMultiplication, newVal -> colorScheme.greenRightMultiplication = newVal);
        addListenerToCheckBox(blueLeftMultiplication, newVal -> colorScheme.blueLeftMultiplication = newVal);
        addListenerToCheckBox(blueRightMultiplication, newVal -> colorScheme.blueRightMultiplication = newVal);
    }

    private void addListenerToCheckBox(CheckBox checkBox, Consumer<Boolean> consumer) {
        checkBox.selectedProperty().addListener((observable, oldValue, newValue) -> {
            if (listenersUnlocked) {
                consumer.accept(newValue);
                onFormSubmitted.execute();
            }
        });
    }

    private void addListenerToSlider(Slider slider, Consumer<Integer> consumer) {
        slider.valueProperty().addListener((observable, oldValue, newValue) -> {
            if (listenersUnlocked) {
                consumer.accept(newValue.intValue());
                onFormSubmitted.execute();
            }
        });
    }

    private void setSliderValue(Slider slider, int value) {
        slider.setValue(value);
    }

    private void setCheckBoxValue(CheckBox checkBox, boolean value) {
        checkBox.setSelected(value);
    }
}
