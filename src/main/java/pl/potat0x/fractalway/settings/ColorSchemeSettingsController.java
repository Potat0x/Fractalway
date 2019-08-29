package pl.potat0x.fractalway.settings;

import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.input.*;
import pl.potat0x.fractalway.colorscheme.ArgbColorScheme;
import pl.potat0x.fractalway.colorscheme.ColorSchemeHistory;
import pl.potat0x.fractalway.colorscheme.PredefinedColorSchemes;
import pl.potat0x.fractalway.utils.Action;

import java.util.function.Consumer;

public class ColorSchemeSettingsController extends BaseController {

    private final ArgbColorScheme colorScheme;
    private final ColorSchemeHistory history;

    @FXML
    private CheckBox randomBitshift;

    @FXML
    private Slider redLeft, redRight;
    @FXML
    private Slider greenLeft, greenRight;
    @FXML
    private Slider blueLeft, blueRight;

    @FXML
    private ToggleButton invertColorsButton;
    @FXML
    private MenuButton colorSchemeMenuButton;
    @FXML
    private SplitMenuButton deleteHistoryButton;
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

    public ColorSchemeSettingsController(ArgbColorScheme colorScheme, ColorSchemeHistory history, Action onFormSubmitted) {
        super(onFormSubmitted);
        this.colorScheme = colorScheme;
        this.history = history;
    }

    @FXML
    private void initialize() {
        super.initialize(invertColorsButton);
        fillForm();
        initValueListeners();
        initPredefinedColorSchemeMenuButton();
        initColorSchemeHistoryPagin();
        updatePagin(history.indexOf(colorScheme));
        initInvertColorsButton();
        addEscKeyEventHandlerToColorSchemeMenu();
    }

    @FXML
    private void createRandomColorScheme() {
        ArgbColorScheme newColorScheme = new ArgbColorScheme();
        newColorScheme.random(randomBitshift.isSelected(), randomLeftMultiplication.isSelected(), randomRightMultiplication.isSelected());
        history.addToHistory(newColorScheme);
        updatePagin(history.size() - 1);
        fillAndSubmitForm();
    }

    @FXML
    private void deleteCurrentColorSchemeFromHistory() {
        int currentPageIndex = getPaginCurrentIndex();
        history.delete(currentPageIndex);
        int newIndex = history.getIndexIfValidElseGetLastIndex(currentPageIndex);
        history.restoreFromHistory(newIndex);
        updatePagin(newIndex);
        fillAndSubmitForm();
    }

    @FXML
    private void clearColorSchemeHistory() {
        history.clear();
        history.addToHistory(colorScheme);
        updatePagin(history.size() - 1);
        fillForm();
    }

    @Override
    protected void fillForm() {
        listenersUnlocked = false;
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

        invertColorsButton.setSelected(colorScheme.invertColors);
        deleteHistoryButton.setDisable(history.size() < 2);
        listenersUnlocked = true;
    }

    private void fillAndSubmitForm() {
        fillForm();
        onFormSubmitted.execute();
    }

    private int getPaginCurrentIndex() {
        return colorSchemeHistoryPagin.getCurrentPageIndex();
    }

    private void updatePagin(int currentId) {
        colorSchemeHistoryPagin.setPageCount(history.size());
        colorSchemeHistoryPagin.setCurrentPageIndex(currentId);
    }

    private void initPredefinedColorSchemeMenuButton() {
        for (PredefinedColorSchemes predefinedScheme : PredefinedColorSchemes.values()) {
            MenuItem menuItem = new MenuItem();
            menuItem.setText(predefinedScheme.name().replaceAll("_", "-"));
            menuItem.setOnAction(event -> {
                colorScheme.assignValues(predefinedScheme.get());
                updateCurrentItemInHistory();
                fillAndSubmitForm();
            });
            colorSchemeMenuButton.getItems().add(menuItem);
        }
    }

    private void initInvertColorsButton() {
        invertColorsButton.selectedProperty().addListener((observable, oldValue, newValue) -> {
            colorScheme.invertColors = newValue;
            updateCurrentItemInHistory();
            fillAndSubmitForm();
        });
    }

    private void updateCurrentItemInHistory() {
        history.updateHistory(history.getIndexIfValidElseGetLastIndex(getPaginCurrentIndex()));
    }

    private void initColorSchemeHistoryPagin() {
        colorSchemeHistoryPagin.currentPageIndexProperty().addListener((observable, oldValue, newValue) -> {
            history.restoreFromHistory(newValue.intValue());
            fillAndSubmitForm();
        });
    }

    private void addEscKeyEventHandlerToColorSchemeMenu() {
        //when focused, MenuButton dont allow scene to receive key event
        colorSchemeMenuButton.setOnKeyPressed(event -> {
            if (event.getCode() == KeyCode.ESCAPE) {
                closeWindow();
            }
        });
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
                updateCurrentItemInHistoryAndSubmitForm();
            }
        });
    }

    private void addListenerToSlider(Slider slider, Consumer<Integer> consumer) {
        slider.valueProperty().addListener((observable, oldValue, newValue) -> {
            if (listenersUnlocked) {
                consumer.accept(newValue.intValue());
                updateCurrentItemInHistoryAndSubmitForm();
            }
        });
    }

    private void updateCurrentItemInHistoryAndSubmitForm() {
        updateCurrentItemInHistory();
        onFormSubmitted.execute();
    }

    private void setSliderValue(Slider slider, int value) {
        slider.setValue(value);
    }

    private void setCheckBoxValue(CheckBox checkBox, boolean value) {
        checkBox.setSelected(value);
    }
}
