package pl.potat0x.fractalway.fractal;

import java.util.Random;

public class ArgbColorScheme {
    public int redLeftShift, redRightShift;
    public int greenLeftShift, greenRightShift;
    public int blueLeftShift, blueRightShift;
    public boolean redLeftMultiplication, redRightMultiplication;
    public boolean greenLeftMultiplication, greenRightMultiplication;
    public boolean blueLeftMultiplication, blueRightMultiplication;

    public ArgbColorScheme() {
        setToDefault();
    }

    private void setToDefault() {
        redLeftShift = 8;
        redRightShift = 24;

        greenLeftShift = 16;
        greenRightShift = 24;

        blueLeftShift = 24;
        blueRightShift = 24;
    }

    public void random(boolean randomShift, boolean randomLeftMultiplication, boolean randomRightMultiplication) {
        int maxShift = 24;
        if (randomShift) {
            redLeftShift = randomInt(maxShift);
            redRightShift = randomInt(maxShift);
            greenLeftShift = randomInt(maxShift);
            greenRightShift = randomInt(maxShift);
            blueLeftShift = randomInt(maxShift);
            blueRightShift = randomInt(maxShift);
        }
        if (randomLeftMultiplication) {
            redLeftMultiplication = randomBoolean();
            greenLeftMultiplication = randomBoolean();
            blueLeftMultiplication = randomBoolean();
        }
        if (randomRightMultiplication) {
            redRightMultiplication = randomBoolean();
            greenRightMultiplication = randomBoolean();
            blueRightMultiplication = randomBoolean();
        }
    }

    private int randomInt(int max) {
        return new Random().nextInt(max + 1);
    }

    private boolean randomBoolean() {
        return new Random().nextBoolean();
    }
}
