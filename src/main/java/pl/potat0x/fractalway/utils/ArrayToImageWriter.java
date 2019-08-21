package pl.potat0x.fractalway.utils;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import static io.vavr.API.*;
import static io.vavr.Predicates.isIn;

public class ArrayToImageWriter {
    public boolean saveImage(int[] argbSource, int imageWidth, int imageHeight, File file, String extension) {

        int imageType = Match(extension.toLowerCase()).of(
                Case($(isIn("jpeg", "bmp")), BufferedImage.TYPE_INT_RGB),
                Case($(), BufferedImage.TYPE_INT_ARGB)
        );

        BufferedImage bufferedImage = new BufferedImage(imageWidth, imageHeight, imageType);
        bufferedImage.setRGB(0, 0, imageWidth, imageHeight, argbSource, 0, imageWidth);

        try {
            return ImageIO.write(bufferedImage, extension, file);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }
}
