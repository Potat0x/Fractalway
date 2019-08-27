package pl.potat0x.fractalway.settings;

import pl.potat0x.fractalway.fractal.ArgbColorScheme;

enum PredefinedColorSchemes {
    DEFAULT(new ArgbColorScheme()),
    BLUE(9, 8, 6, 5, 11, 23);

    private final ArgbColorScheme colorScheme;

    PredefinedColorSchemes(int rLeftSh, int rRightSh, int gLeftSh, int gRightSh, int bLeftSh, int bRightSh) {
        this(rLeftSh, rRightSh, gLeftSh, gRightSh, bLeftSh, bRightSh, false, false, false, false, false, false);
    }

    PredefinedColorSchemes(int rLeftSh, int rRightSh, int gLeftSh, int gRightSh, int bLeftSh, int bRightSh,
                           boolean rLeftMul, boolean rRightMul, boolean gLeftMul, boolean gRightMul, boolean bLeftMul, boolean bRightMul) {
        colorScheme = new ArgbColorScheme();
        colorScheme.redLeftShift = rLeftSh;
        colorScheme.redRightShift = rRightSh;
        colorScheme.greenLeftShift = gLeftSh;
        colorScheme.greenRightShift = gRightSh;
        colorScheme.blueLeftShift = bLeftSh;
        colorScheme.blueRightShift = bRightSh;

        colorScheme.redLeftMultiplication = rLeftMul;
        colorScheme.redRightMultiplication = rRightMul;
        colorScheme.greenLeftMultiplication = gLeftMul;
        colorScheme.greenRightMultiplication = gRightMul;
        colorScheme.blueLeftMultiplication = bLeftMul;
        colorScheme.blueRightMultiplication = bRightMul;
    }

    PredefinedColorSchemes(ArgbColorScheme colorScheme) {
        this.colorScheme = colorScheme;
    }

    public ArgbColorScheme get() {
        return colorScheme;
    }
}
