package pl.potat0x.fractalway.validation;

import javafx.beans.property.StringProperty;
import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.scene.control.TextField;

class TextFieldCorrectnessWatcher implements ChangeListener<String> {
    private final static String textFieldErrorStyle = "-fx-text-box-border: rgb(255,117,0); -fx-focus-color: rgb(255,117,0);";
    private final Validator validator;
    private final Action onValidAction;
    private final Action onInvalidAction;
    private boolean isFieldValid;

    TextFieldCorrectnessWatcher(Validator validator, Action onValidAction, Action onInvalidAction) {
        this.validator = validator;
        this.onValidAction = onValidAction;
        this.onInvalidAction = onInvalidAction;
        isFieldValid = true;
    }

    @Override
    public void changed(ObservableValue<? extends String> observable, String oldValue, String newValue) {
        TextField textField = (TextField) ((StringProperty) observable).getBean();
        isFieldValid = validator.check(textField.getText());

        if (isFieldValid) {
            textField.setStyle(null);
            onValidAction.execute();
        } else {
            textField.setStyle(textFieldErrorStyle);
            onInvalidAction.execute();
        }
    }

    boolean isFieldValid() {
        return isFieldValid;
    }
}
