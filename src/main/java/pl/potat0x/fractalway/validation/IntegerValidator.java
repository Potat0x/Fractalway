package pl.potat0x.fractalway.validation;

public class IntegerValidator implements Validator {
    private final Integer minVal;
    private final Integer maxVal;

    public IntegerValidator(Integer minVal, Integer maxVal) {
        this.minVal = minVal;
        this.maxVal = maxVal;
    }

    @Override
    public boolean check(String value) {
        try {
            int val = Integer.parseInt(value);
            return (minVal == null || val >= minVal) && (maxVal == null || val <= maxVal);
        } catch (NumberFormatException e) {
            return false;
        }
    }
}
