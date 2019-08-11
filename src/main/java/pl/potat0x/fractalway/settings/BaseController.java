package pl.potat0x.fractalway.settings;

import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.control.TextField;
import javafx.stage.Stage;
import pl.potat0x.fractalway.validation.TextFieldGroupCorrectnessWatcher;

class BaseController {
    private Node windowNode;
    private boolean isSubmittingBlocked;

    @FXML
    private Button okButton;

    void initialize(Node windowNode) {
        this.windowNode = windowNode;
        isSubmittingBlocked = false;
    }

    @FXML
    void closeWindow() {
        Stage window = (Stage) windowNode.getScene().getWindow();
        window.close();
    }

    boolean isSubmittingUnblocked() {
        return !isSubmittingBlocked;
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

    private void setSubmittingBlocked(boolean blocked) {
        okButton.setDisable(blocked);
        isSubmittingBlocked = blocked;
    }
}
