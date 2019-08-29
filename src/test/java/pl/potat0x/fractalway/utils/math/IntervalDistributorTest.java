package pl.potat0x.fractalway.utils.math;

import io.vavr.Tuple2;
import org.junit.Test;

import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class IntervalDistributorTest {
    @Test
    public void cutIntervalToSimilarSectionsTest() {
        final boolean verbose = false;

        for (int sectionsInInterval = 1; sectionsInInterval < 100; sectionsInInterval++) {
            for (int intervalWidth = 1; intervalWidth < 100; intervalWidth++) {

                List<Tuple2<Integer, Integer>> sections = IntervalDistributor.cutIntervalToSimilarSections(sectionsInInterval, intervalWidth);

                if (verbose) {
                    System.out.println("intervalWidth = " + intervalWidth + " / " + "sectionsInInterval = " + sectionsInInterval);
                    for (Tuple2<Integer, Integer> section : sections) {
                        System.out.println(section + " -> " + (section._2 - section._1));
                    }
                }

                assertEquals(sectionsInInterval, sections.size());
                assertEquals(0, sections.get(0)._1.intValue());
                assertEquals(intervalWidth, sections.get(sections.size() - 1)._2.intValue());
                assertTrue(checkIfSectionsNotOverlap(sections));
                assertEquals(intervalWidth, sumSections(sections));
            }
        }
    }

    private boolean checkIfSectionsNotOverlap(List<Tuple2<Integer, Integer>> section) {
        if (section.size() == 0) {
            return true;
        }

        for (int i = 0; i + 1 < section.size(); i++) {
            if (section.get(i)._2 > section.get(i + 1)._1) {
                return false;
            }
        }
        return true;
    }

    private int sumSections(List<Tuple2<Integer, Integer>> sections) {
        return sections.stream()
                .map(x -> x._2 - x._1)
                .reduce(0, Integer::sum);
    }
}