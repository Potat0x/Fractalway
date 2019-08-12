package pl.potat0x.fractalway.validation;

@FunctionalInterface
public interface Validator {
    boolean check(String value);
}
