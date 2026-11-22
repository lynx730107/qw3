import unittest
from tags import parse_tags

class TestTags(unittest.TestCase):
    def test_quoted_delimiter(self):
        self.assertEqual(parse_tags('news,"Rome, Italy", work'), ['news', 'Rome, Italy', 'work'])

    def test_spaces_and_empty(self):
        self.assertEqual(parse_tags(' , news ,, "  work  ", '), ['news', 'work'])

    def test_quotes(self):
        self.assertEqual(parse_tags('"say ""hi""", travel'), ['say "hi"', 'travel'])

    def test_empty_line(self):
        self.assertEqual(parse_tags(''), [])
