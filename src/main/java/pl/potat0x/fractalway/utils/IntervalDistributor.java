package pl.potat0x.fractalway.utils;

import io.vavr.Tuple;
import io.vavr.Tuple2;

import java.util.ArrayList;
import java.util.List;

public class IntervalDistributor {
    public static List<Tuple2<Integer, Integer>> cutIntervalToSimilarSections(int numberOfSections, int intervalWidth) {

        List<Tuple2<Integer, Integer>> sections = new ArrayList<>();
        for (int seectionNumber = 0; seectionNumber < numberOfSections; seectionNumber++) {
            sections.add(calculateNthSection(seectionNumber, numberOfSections, intervalWidth));
        }

        int sectionShift = 0;
        if (intervalWidth % numberOfSections != 0) {
            for (int i = 0; i < sections.size(); i++) {
                Tuple2<Integer, Integer> currentSection = sections.get(i);
                if (i < (intervalWidth % numberOfSections)) {
                    sectionShift = i;
                    sections.set(i, Tuple.of(currentSection._1 + sectionShift, currentSection._2 + sectionShift + 1));
                } else {
                    sections.set(i, Tuple.of(currentSection._1 + sectionShift + 1, currentSection._2 + sectionShift + 1));
                }
            }
        }
        return sections;
    }

    private static Tuple2<Integer, Integer> calculateNthSection(int sectionNumber, int numberOfSections, int intervalWidth) {
        int sectionWidth = intervalWidth / numberOfSections;
        int start = sectionNumber * sectionWidth;
        int end = start + sectionWidth;
        return Tuple.of(start, end);
    }
}
