package pl.potat0x.fractalway.validation

import spock.lang.Specification

class IntegerValidatorTest extends Specification {
    def "Should accept valid values"() {
        expect:
        new IntegerValidator(min, max).check(value)

        where:
        min  | value | max
        null | "2"   | null
        null | "1"   | 2
        null | "2"   | 2
        2    | "2"   | null
        2    | "3"   | null
        2    | "2"   | 2
    }

    def "Should detect invalid values"() {
        expect:
        !new IntegerValidator(min, max).check(value)

        where:
        min  | value  | max
        null | null   | null
        null | ""     | null
        null | "test" | null
        null | "3"    | 2
        2    | "1"    | null
        2    | "1"    | 2
        2    | "3"    | 2
    }
}
