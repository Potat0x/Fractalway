package pl.potat0x.fractalway.utils;

public class StringCapitalizer {
    public static String capitalizeFirstLetter(String string) {
        if (string == null || string.length() == 0) {
            return string;
        } else {
            String firstChar = "" + string.charAt(0);
            return firstChar.toUpperCase() + string.substring(1);
        }
    }
}
