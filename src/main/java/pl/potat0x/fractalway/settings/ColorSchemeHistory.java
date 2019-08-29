package pl.potat0x.fractalway.settings;

import pl.potat0x.fractalway.fractal.ArgbColorScheme;

import java.util.LinkedList;
import java.util.List;

class ColorSchemeHistory {
    private final List<ArgbColorScheme> history;
    private final ArgbColorScheme originalObject;

    ColorSchemeHistory(ArgbColorScheme originalObject) {
        history = new LinkedList<>();
        this.originalObject = originalObject;
    }

    void addToHistory(ArgbColorScheme obj) {
        history.add(obj.copy());
    }

    ArgbColorScheme readFromHistory(int id) {
        return history.get(id);
    }

    void updateHistory(int id) {
        history.get(id).assignValues(originalObject);
    }

    void restoreFromHistory(int id) {
        originalObject.assignValues(readFromHistory(id));
    }

    void delete(int currentId) {
        history.remove(currentId);
    }

    void clear() {
        history.clear();
    }

    int size() {
        return history.size();
    }

    int getIndexIfValidElseGetLastIndex(int id) {
        if (id < history.size()) {
            return id;
        } else {
            return history.size() - 1;
        }
    }
}
