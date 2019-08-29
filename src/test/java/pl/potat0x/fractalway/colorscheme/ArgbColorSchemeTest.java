package pl.potat0x.fractalway.colorscheme;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotSame;

public class ArgbColorSchemeTest {
    @Test
    public void copyTest() {
        ArgbColorScheme cs = new ArgbColorScheme();
        cs.invertColors = true;
        cs.redLeftShift = 0;
        cs.redRightShift = 1;
        cs.greenLeftShift = 2;
        cs.greenRightShift = 3;
        cs.blueLeftShift = 4;
        cs.blueRightShift = 5;
        cs.redLeftMultiplication = true;
        cs.redRightMultiplication = true;
        cs.greenLeftMultiplication = true;
        cs.greenRightMultiplication = true;
        cs.blueLeftMultiplication = true;
        cs.blueRightMultiplication = true;

        ArgbColorScheme csCopy = cs.copy();

        assertNotSame(cs, csCopy);
        assertEquals(cs, csCopy);
    }
}
