# braille_map.py
# Maps tuple of active dot positions (1-6) to English characters
# Braille cell layout:
#   Dot 1  Dot 4
#   Dot 2  Dot 5
#   Dot 3  Dot 6

BRAILLE_MAP = {
    (1,):           'a',
    (1, 2):         'b',
    (1, 4):         'c',
    (1, 4, 5):      'd',
    (1, 5):         'e',
    (1, 2, 4):      'f',
    (1, 2, 4, 5):   'g',
    (1, 2, 5):      'h',
    (2, 4):         'i',
    (2, 4, 5):      'j',
    (1, 3):         'k',
    (1, 2, 3):      'l',
    (1, 3, 4):      'm',
    (1, 3, 4, 5):   'n',
    (1, 3, 5):      'o',
    (1, 2, 3, 4):   'p',
    (1, 2, 3, 4, 5):'q',
    (1, 2, 3, 5):   'r',
    (2, 3, 4):      's',
    (2, 3, 4, 5):   't',
    (1, 3, 6):      'u',
    (1, 2, 3, 6):   'v',
    (2, 4, 5, 6):   'w',
    (1, 3, 4, 6):   'x',
    (1, 3, 4, 5, 6):'y',
    (1, 3, 5, 6):   'z',
    ():             ' ',
}

def dots_to_char(active_dots: list) -> str:
    """
    Convert a list of active dot positions to an English character.
    active_dots: list of integers, e.g. [1, 2, 4] → 'f'
    Returns '?' if pattern not found.
    """
    key = tuple(sorted(active_dots))
    return BRAILLE_MAP.get(key, '')


def test_mapping():
    """Quick test to verify mapping works correctly."""
    tests = {
        (1,): 'a',
        (1, 2): 'b',
        (1, 4): 'c',
        (2, 4, 5, 6): 'w',
        (): ' ',
    }
    all_passed = True
    for dots, expected in tests.items():
        result = dots_to_char(list(dots))
        status = "✅" if result == expected else "❌"
        print(f"{status} dots {dots} → '{result}' (expected '{expected}')")
        if result != expected:
            all_passed = False
    print("\n✅ All tests passed!" if all_passed else "\n❌ Some tests failed!")


if __name__ == "__main__":
    test_mapping()