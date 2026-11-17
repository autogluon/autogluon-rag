import unittest

from agrag.evaluation.utils import (
    answers_equivalent,
    inclusive_exact_match_metric,
)


class TestAnswersEquivalent(unittest.TestCase):
    def test_year_matches_containing_decade(self):
        self.assertTrue(answers_equivalent("1973", "1970s"))
        self.assertTrue(answers_equivalent("the 1970s", "released in 1973"))

    def test_year_matches_decade_with_qualifier(self):
        # "Early 1980s" vs "1981" -- same decade bucket.
        self.assertTrue(answers_equivalent("early 1980s", "1981"))

    def test_different_decades_do_not_match(self):
        self.assertFalse(answers_equivalent("1973", "1980s"))
        self.assertFalse(answers_equivalent("1969", "1970s"))

    def test_exact_year_match(self):
        self.assertTrue(answers_equivalent("built in 1990", "1990"))

    def test_no_date_information_is_not_equivalent(self):
        # Non-date answers must be untouched by the date path.
        self.assertFalse(answers_equivalent("paris", "london"))

    def test_region_alias_group(self):
        self.assertTrue(answers_equivalent("usa", "united states"))
        self.assertTrue(answers_equivalent("uk", "great britain"))

    def test_region_alias_outside_group(self):
        self.assertFalse(answers_equivalent("canada", "united states"))


class TestInclusiveExactMatchWithNormalization(unittest.TestCase):
    def test_date_granularity_counts_as_match(self):
        # Prediction at year granularity, gold at decade granularity.
        matches = inclusive_exact_match_metric(
            predictions=["1973"],
            references=[["1970s"]],
            ignore_case=True,
        )
        self.assertEqual(matches, [True])

    def test_plain_substring_still_matches(self):
        matches = inclusive_exact_match_metric(
            predictions=["The capital is Paris"],
            references=[["Paris"]],
            ignore_case=True,
        )
        self.assertEqual(matches, [True])

    def test_genuine_mismatch_still_fails(self):
        matches = inclusive_exact_match_metric(
            predictions=["Berlin"],
            references=[["Paris"]],
            ignore_case=True,
        )
        self.assertEqual(matches, [False])

    def test_wrong_decade_still_fails(self):
        matches = inclusive_exact_match_metric(
            predictions=["1965"],
            references=[["1970s"]],
            ignore_case=True,
        )
        self.assertEqual(matches, [False])


if __name__ == "__main__":
    unittest.main()
