package pl.potat0x.fractalway.colorscheme;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class ColorSchemeHistoryTest {
    @Test
    public void historyTest() {
        final ArgbColorScheme originalObject = new ArgbColorScheme();
        originalObject.redLeftShift = 1;

        ColorSchemeHistory history = new ColorSchemeHistory(originalObject);
        history.addToHistory(originalObject);

        final ArgbColorScheme originalObjectBackup = originalObject.copy();
        assertEquals(originalObjectBackup, originalObject);

        originalObject.redLeftShift = 2;
        assertNotEquals(originalObjectBackup, originalObject);

        history.restoreFromHistory(0);
        assertEquals(originalObjectBackup, originalObject);

        originalObject.redLeftShift = 3;
        assertNotEquals(originalObject.redLeftShift, history.readFromHistory(0).redLeftShift);

        history.updateHistory(0);
        assertEquals(originalObject.redLeftShift, history.readFromHistory(0).redLeftShift);
    }
}
