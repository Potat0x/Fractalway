package pl.potat0x.fractalway.colorscheme;

import java.util.Objects;
import java.util.Random;

public class ArgbColorScheme {
    public boolean invertColors;
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

    public ArgbColorScheme copy() {
        ArgbColorScheme newCs = new ArgbColorScheme();
        newCs.assignValues(this);
        return newCs;
    }

    public void assignValues(ArgbColorScheme src) {
        invertColors = src.invertColors;
        redLeftShift = src.redLeftShift;
        redRightShift = src.redRightShift;
        greenLeftShift = src.greenLeftShift;
        greenRightShift = src.greenRightShift;
        blueLeftShift = src.blueLeftShift;
        blueRightShift = src.blueRightShift;
        redLeftMultiplication = src.redLeftMultiplication;
        redRightMultiplication = src.redRightMultiplication;
        greenLeftMultiplication = src.greenLeftMultiplication;
        greenRightMultiplication = src.greenRightMultiplication;
        blueLeftMultiplication = src.blueLeftMultiplication;
        blueRightMultiplication = src.blueRightMultiplication;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ArgbColorScheme that = (ArgbColorScheme) o;
        return invertColors == that.invertColors &&
                redLeftShift == that.redLeftShift &&
                redRightShift == that.redRightShift &&
                greenLeftShift == that.greenLeftShift &&
                greenRightShift == that.greenRightShift &&
                blueLeftShift == that.blueLeftShift &&
                blueRightShift == that.blueRightShift &&
                redLeftMultiplication == that.redLeftMultiplication &&
                redRightMultiplication == that.redRightMultiplication &&
                greenLeftMultiplication == that.greenLeftMultiplication &&
                greenRightMultiplication == that.greenRightMultiplication &&
                blueLeftMultiplication == that.blueLeftMultiplication &&
                blueRightMultiplication == that.blueRightMultiplication;
    }

    @Override
    public int hashCode() {
        return Objects.hash(invertColors, redLeftShift, redRightShift, greenLeftShift, greenRightShift, blueLeftShift, blueRightShift, redLeftMultiplication, redRightMultiplication, greenLeftMultiplication, greenRightMultiplication, blueLeftMultiplication, blueRightMultiplication);
    }

    @Override
    public String toString() {
        return "ArgbColorScheme{" +
                "invertColors=" + invertColors +
                ", redLeftShift=" + redLeftShift +
                ", redRightShift=" + redRightShift +
                ", greenLeftShift=" + greenLeftShift +
                ", greenRightShift=" + greenRightShift +
                ", blueLeftShift=" + blueLeftShift +
                ", blueRightShift=" + blueRightShift +
                ", redLeftMultiplication=" + redLeftMultiplication +
                ", redRightMultiplication=" + redRightMultiplication +
                ", greenLeftMultiplication=" + greenLeftMultiplication +
                ", greenRightMultiplication=" + greenRightMultiplication +
                ", blueLeftMultiplication=" + blueLeftMultiplication +
                ", blueRightMultiplication=" + blueRightMultiplication +
                '}';
    }
}
