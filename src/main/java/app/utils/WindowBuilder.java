package app.utils;

import javafx.event.EventHandler;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Modality;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;

import java.io.IOException;

public class WindowBuilder {
    private String filename;
    private Object controller;
    private String title;
    private EventHandler<WindowEvent> windowEventEventHandler;

    public WindowBuilder(String filename) {
        this.filename = filename;
    }

    public WindowBuilder withController(Object controller) {
        this.controller = controller;
        return this;
    }

    public WindowBuilder withOnHiddenEventHandler(EventHandler<WindowEvent> handler) {
        this.windowEventEventHandler = handler;
        return this;
    }

    public WindowBuilder withTitle(String title) {
        this.title = title;
        return this;
    }

    public Stage build() throws IOException {
        FXMLLoader loader = new FXMLLoader(getClass().getResource(filename));
        if (controller != null) {
            loader.setController(controller);
        }
        Stage stage = new Stage();
        stage.setScene(new Scene(loader.load()));
        stage.initModality(Modality.APPLICATION_MODAL);
        stage.setOnHidden(windowEventEventHandler);
        stage.setResizable(false);
        stage.setTitle(title);
        return stage;
    }
}
