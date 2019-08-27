package pl.potat0x.fractalway.settings;

import pl.potat0x.fractalway.fractal.ArgbColorScheme;

enum PredefinedColorSchemes {
    DEFAULT(new ArgbColorScheme()),
    MIX_1(12, 22, 8, 11, 2, 22, true, true, true, true, false, true),
    MIX_2(5, 18, 19, 23, 10, 17),
    MIX_3(18, 14, 11, 20, 17, 23, false, true, false, true, false, false),
    MIX_4(11, 19, 21, 18, 7, 19),
    BLACK_WHITE(2, 16, 8, 22, 8, 22),
    BLUE_YELLOW(9, 8, 6, 5, 11, 23),
    WHITE_PINK_BLUE(11, 21, 7, 12, 2, 22),
    GREEN_RED_YELLOW(8, 24, 24, 22, 24, 0),
    PINK_RED_BLUE(17, 9, 24, 14, 8, 21),
    INV_1(16, 16, 7, 13, 11, 24),
    GREEN_ONLY(0, 0, 21, 21, 0, 0, true, true, false, false, true, true);

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
