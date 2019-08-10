package app.settings;

import javafx.fxml.FXML;
import javafx.scene.Node;
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
}
