package pl.potat0x.fractalway.utils.math;

/*
    Util for increasing slider precision for low values.
    Example (maxVal=500, exp=2.5):

    Value from simple linear slider
    ┌──────┬──────┬──────┬──────┬──────┐
    0     100    200    300    400    500   <-- f(x) = x

    can be converted to parabolic
    ┌──────┬──────┬──────┬──────┬──────┐
    0      9      51    139    286    500   <-- f(x) = (ax)^exp
 */

public class ParabolicScaleConverter {
    private final double exp;
    private final double a;

    public ParabolicScaleConverter(double maxVal, double exp) {
        this.exp = exp;
        this.a = Math.pow(maxVal, 1.0 / exp) / maxVal;
    }

    public double linearToParabolic(double linearValue) {
        return Math.pow(linearValue * a, exp);
    }

    public double parabolicToLinear(double parabolicValue) {
        return Math.pow(parabolicValue, 1.0 / exp) / a;
    }

    public boolean checkIfParabolicValuesAreDifferentAsIntegers(double a, double b) {
        int parabolicValA = (int) Math.round(linearToParabolic(a));
        int parabolicValB = (int) Math.round(linearToParabolic(b));
        return parabolicValA != parabolicValB;
    }
}
