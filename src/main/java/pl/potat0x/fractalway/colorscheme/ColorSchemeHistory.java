package pl.potat0x.fractalway.colorscheme;

import java.util.LinkedList;
import java.util.List;

public class ColorSchemeHistory {
    private final List<ArgbColorScheme> history;
    private final ArgbColorScheme originalObject;

    public ColorSchemeHistory(ArgbColorScheme originalObject) {
        history = new LinkedList<>();
        this.originalObject = originalObject;
    }

    public void addToHistory(ArgbColorScheme obj) {
        history.add(obj.copy());
    }

    public ArgbColorScheme readFromHistory(int id) {
        return history.get(id);
    }

    public void updateHistory(int id) {
        history.get(id).assignValues(originalObject);
    }

    public void restoreFromHistory(int id) {
        originalObject.assignValues(readFromHistory(id));
    }

    public void delete(int currentId) {
        history.remove(currentId);
    }

    public void clear() {
        history.clear();
    }

    public int size() {
        return history.size();
    }

    public int indexOf(ArgbColorScheme obj) {
        return history.indexOf(obj);
    }

    public int getIndexIfValidElseGetLastIndex(int id) {
        if (id < history.size()) {
            return id;
        } else {
            return history.size() - 1;
        }
    }
}
