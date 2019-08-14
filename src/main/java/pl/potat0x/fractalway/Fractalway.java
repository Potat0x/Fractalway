package pl.potat0x.fractalway;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.nvrtc.JNvrtc.setExceptionsEnabled;

public class Fractalway extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        Parent root = FXMLLoader.load(getClass().getResource("/fxml/main.fxml"));
        primaryStage.setTitle("Fractalway");
        primaryStage.setScene(new Scene(root, 820, 620));
        primaryStage.show();
    }

    public static void main(String[] args) {
        initCuda();
        launch(args);
    }

    private static void initCuda() {
        setExceptionsEnabled(true);
        cuInit(0);
    }
}
