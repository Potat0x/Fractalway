package pl.potat0x.fractalway.validation;

import javafx.scene.control.TextField;
import pl.potat0x.fractalway.utils.Action;

import java.util.ArrayList;
import java.util.List;

public class TextFieldGroupCorrectnessWatcher {
    private final List<TextFieldCorrectnessWatcher> watchers;
    private final Action onValidAction;
    private final Action onInvalidAction;

    public TextFieldGroupCorrectnessWatcher(Action onGroupValidAction, Action onGroupInvalidAction) {
        watchers = new ArrayList<>(5);
        this.onValidAction = () -> {
            if (isFormValid()) {
                onGroupValidAction.execute();
            }
        };
        this.onInvalidAction = () -> {
            if (!isFormValid()) {
                onGroupInvalidAction.execute();
            }
        };
    }

    public TextFieldGroupCorrectnessWatcher registerFields(Validator validator, TextField... textFields) {
        for (TextField field : textFields) {
            TextFieldCorrectnessWatcher watcher = new TextFieldCorrectnessWatcher(validator, onValidAction, onInvalidAction);
            field.textProperty().addListener(watcher);
            watchers.add(watcher);
        }
        return this;
    }

    private boolean isFormValid() {
        return watchers.stream().allMatch(TextFieldCorrectnessWatcher::isFieldValid);
    }
}
