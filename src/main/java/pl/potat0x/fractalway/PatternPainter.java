package pl.potat0x.fractalway;

class PatternPainter {
    private final int[] argb;
    private final int canvasWidth;

    PatternPainter(int canvasWidth, int[] argb) {
        this.argb = argb;
        this.canvasWidth = canvasWidth;
    }

    int scaleTo255(int valToScale) {
        return (int) (((1.0 * valToScale) / (1.0 * canvasWidth)) * 255.0);
    }

    void gradient(int x, int y) {
        int index = calculateIndex(x, y);
        argb[index] = argb(scaleTo255(x), scaleTo255(y), 0);
    }

    void diagonalStripes(int x, int y) {
        int index = calculateIndex(x, y);
        float mn = (float) ((x + y) % 33 < 15 ? 0.5 : 0.9);
        argb[index] = argb((int) (44 * mn));
    }

    void colorfulStripesAndCircles(int x, int y) {
        int index = calculateIndex(x, y);
        argb[index] = argb(
                ((x * y) % 50) * 4,
                ((x + y) % 30) * 7,
                ((x + y) % 13) * 17);
    }

    private int argb(int color) {
        return (255 << 24) | (color << 16) | (color << 8) | color;
    }

    private int argb(int r, int g, int b) {
        return (255 << 24) | (r << 16) | (g << 8) | b;
    }

    private int calculateIndex(int x, int y) {
        return canvasWidth * y + x;
    }
}
