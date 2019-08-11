package app.settings;

import javafx.fxml.FXML;
import javafx.scene.Node;
import javafx.scene.control.TextField;
import javafx.stage.Stage;

class BaseController {
    private Node windowNode;

    void initialize(Node windowNode) {
        this.windowNode = windowNode;
    }

    @FXML
    void closeWindow() {
        Stage window = (Stage) windowNode.getScene().getWindow();
        window.close();
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
}
