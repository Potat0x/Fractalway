package pl.potat0x.fractalway.utils.math;

import io.vavr.Tuple;
import io.vavr.Tuple2;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class ParabolicScaleConverterTest {
    /*
     * https://www.desmos.com/calculator
     * f(x)=(ax)^2.7, a=0.012915495
     * */

    @Test
    public void conversionTest() {
        double eps = 0.3;
        double maxValue = 1000.0;

        List<Tuple2<Double, Double>> testCases = Arrays.asList(
                Tuple.of(0.0, 0.0),
                Tuple.of(50.0, 0.31),
                Tuple.of(100.0, 1.99),
                Tuple.of(200.0, 12.97),
                Tuple.of(300.0, 38.75),
                Tuple.of(400.0, 84.25),
                Tuple.of(500.0, 153.89),
                Tuple.of(600.0, 251.77),
                Tuple.of(700.0, 381.74),
                Tuple.of(800.0, 547.49),
                Tuple.of(900.0, 752.41),
                Tuple.of(950.0, 870.67),
                Tuple.of(maxValue, maxValue)
        );

        ParabolicScaleConverter converter = new ParabolicScaleConverter(maxValue, 2.7);

        for (Tuple2<Double, Double> testCase : testCases) {
            double givenLinear = testCase._1;
            double givenParabolic = testCase._2;

            double computedParabolic = converter.linearToParabolic(givenLinear);
            double computedLinear = converter.parabolicToLinear(givenParabolic);

            assertEquals(givenParabolic, computedParabolic, eps);
            assertEquals(givenLinear, computedLinear, eps);
        }
    }
}
