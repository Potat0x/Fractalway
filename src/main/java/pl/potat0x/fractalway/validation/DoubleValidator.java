package pl.potat0x.fractalway.validation;

public class DoubleValidator implements Validator {
    @Override
    public boolean check(String value) {
        try {
            Double.parseDouble(value);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}
