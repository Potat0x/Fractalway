package pl.potat0x.fractalway.validation;

public class IntegerValidator implements Validator {
    @Override
    public boolean check(String value) {
        try {
            Integer.parseInt(value);
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}
