package pl.potat0x.fractalway.utils;

@FunctionalInterface
public interface Action {
    void execute();

    Action EMPTY = () -> {
    };
}
