package pl.potat0x.fractalway.utils;

import io.vavr.Function0;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public final class Config {

    public static Integer getInt(String propertyName) {
        return Integer.parseInt(properties.get().get(propertyName).toString());
    }

    private static final Function0<Properties> properties = Function0.of(Config::loadProperties).memoized();

    private static Properties loadProperties() {
        Properties properties = new Properties();
        try (InputStream is = Config.class.getResourceAsStream("/application.properties")) {
            properties.load(is);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return properties;
    }
}
