package app.utils;

import javafx.event.EventHandler;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.input.KeyCode;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;

import java.io.IOException;

public class WindowBuilder {
    private String filename;
    private Object controller;
    private String title;
    private boolean closedByEscKey;
    private EventHandler<WindowEvent> onHiddenEventHandler;

    public WindowBuilder(String filename) {
        this.filename = filename;
        closedByEscKey = false;
    }

    public WindowBuilder withController(Object controller) {
        this.controller = controller;
        return this;
    }

    public WindowBuilder withOnHiddenEventHandler(EventHandler<WindowEvent> handler) {
        this.onHiddenEventHandler = handler;
        return this;
    }

    public WindowBuilder withTitle(String title) {
        this.title = title;
        return this;
    }

    public WindowBuilder closedByEscapeKey(boolean closeOnEscapeKeyPressed) {
        this.closedByEscKey = closeOnEscapeKeyPressed;
        return this;
    }

    public Stage build() throws IOException {
        FXMLLoader loader = new FXMLLoader(getClass().getResource(filename));
        if (controller != null) {
            loader.setController(controller);
        }
        Stage stage = new Stage();
        Scene scene = new Scene(loader.load());
        if (closedByEscKey) {
            scene.setOnKeyPressed(event -> {
                if (event.getCode() == KeyCode.ESCAPE) {
                    stage.close();
                }
            });
        }
        stage.setScene(scene);
        stage.initModality(Modality.APPLICATION_MODAL);
        stage.setOnHidden(onHiddenEventHandler);
        stage.sizeToScene();
        stage.setResizable(false);
        stage.setTitle(title);
        return stage;
    }
}
