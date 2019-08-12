package pl.potat0x.fractalway.settings;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.scene.input.KeyCode;
import javafx.scene.input.KeyEvent;
import javafx.stage.Stage;
import pl.potat0x.fractalway.utils.Action;
import pl.potat0x.fractalway.validation.TextFieldGroupCorrectnessWatcher;

abstract class BaseController {
    private Node windowNode;
    private final Action setOnFormSubmitted;
    private boolean isSubmittingBlocked;

    @FXML
    private Button okButton;

    protected abstract void readForm();

    protected abstract void fillForm();

    protected abstract void setCorrectnessWatchers();

    BaseController(Action setOnFormSubmitted) {
        this.setOnFormSubmitted = setOnFormSubmitted != null ? setOnFormSubmitted : Action.EMPTY;
    }

    void initialize(Node windowNode) {
        this.windowNode = windowNode;
        isSubmittingBlocked = false;
    }

    @FXML
    void closeWindow() {
        Stage window = (Stage) windowNode.getScene().getWindow();
        window.close();
    }

    @FXML
    protected void handleKeyEvent(KeyEvent keyEvent) {
        if (keyEvent.getCode() == KeyCode.ENTER && isSubmittingUnblocked()) {
            readForm();
            setOnFormSubmitted.execute();
            if (keyEvent.isControlDown()) {
                closeWindow();
            }
        }
    }

    @FXML
    protected void handleOkButton(ActionEvent actionEvent) {
        if (isSubmittingUnblocked()) {
            readForm();
            closeWindow();
        }
    }

    TextFieldGroupCorrectnessWatcher createFormCorrectnessWatcher() {
        return new TextFieldGroupCorrectnessWatcher(() -> setSubmittingBlocked(false), () -> setSubmittingBlocked(true));
    }

    void setText(TextField textField, double value) {
        textField.setText(Double.toString(value));
    }

    void setText(TextField textField, int value) {
        textField.setText(Integer.toString(value));
    }

    double readDouble(TextField textField) {
        return Double.parseDouble(textField.getText());
    }

    int readInteger(TextField textField) {
        return Integer.parseInt(textField.getText());
    }

    private boolean isSubmittingUnblocked() {
        return !isSubmittingBlocked;
    }

    private void setSubmittingBlocked(boolean blocked) {
        okButton.setDisable(blocked);
        isSubmittingBlocked = blocked;
    }
}
