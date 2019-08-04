package app;

class PatternPainter {
    private final int[] red;
    private final int[] green;
    private final int[] blue;
    private final int canvasWidth;

    PatternPainter(int canvasWidth, int[] red, int[] green, int[] blue) {
        this.red = red;
        this.green = green;
        this.blue = blue;
        this.canvasWidth = canvasWidth;
    }

    int scaleTo255(int valToScale) {
        return (int) (((1.0 * valToScale) / (1.0 * canvasWidth)) * 255.0);
    }

    void gradient(int x, int y) {
        int index = calculateIndex(x, y);
        red[index] = scaleTo255(x);
        green[index] = scaleTo255(y);
        blue[index] = 0;
    }

    void diagonalStripes(int x, int y) {
        int index = calculateIndex(x, y);
        float mn = (float) ((x + y) % 33 < 15 ? 0.5 : 0.9);
        red[index] = (int) (44 * mn);
        green[index] = (int) (44 * mn);
        blue[index] = (int) (44 * mn);
    }

    void colorfulStripesAndCircles(int x, int y) {
        int index = calculateIndex(x, y);
        red[index] = ((x * y) % 50) * 4;
        green[index] = ((x + y) % 30) * 7;
        blue[index] = ((x + y) % 13) * 17;
    }

    private int calculateIndex(int x, int y) {
        return canvasWidth * y + x;
    }
}
