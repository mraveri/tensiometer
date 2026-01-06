"""Tests for color utilities."""

#########################################################################################################
# Imports

import unittest

from tensiometer.utilities.color_utilities import color_linear_interpolation, nice_colors
import tensiometer.utilities.color_utilities as cu

#########################################################################################################
# Test cases

class TestColorUtilities(unittest.TestCase):

    """Color utilities test suite."""
    def test_color_linear_interpolation(self):
        """Test color linear interpolation."""
        rgb_1 = (1.0, 0.0, 0.0)
        rgb_2 = (0.0, 1.0, 0.0)
        alpha = 0.5
        expected_color = (0.5, 0.5, 0.0)
        result = color_linear_interpolation(rgb_1, rgb_2, alpha)
        self.assertEqual(result, expected_color)

    def test_nice_colors_integer(self):
        """Test nice_colors with integer input."""
        color = nice_colors(2, colormap="the_gold_standard", interpolation_method="linear", output_format="RGB_255")
        self.assertEqual(color, (42, 46, 139))

    def test_nice_colors_float(self):
        """Test nice_colors with float interpolation."""
        color = nice_colors(2.5, colormap="the_gold_standard", interpolation_method="linear", output_format="RGB_255")
        self.assertIsInstance(color, tuple)

    def test_nice_colors_hex_output(self):
        """Test nice_colors hex output."""
        color = nice_colors(2.5, colormap="the_gold_standard", output_format="HEX")
        self.assertTrue(color.startswith("#"))

    def test_nice_colors_rgb_output(self):
        """Test nice_colors RGB output."""
        color = nice_colors(2.5, colormap="the_gold_standard", output_format="RGB")
        self.assertIsInstance(color, tuple)

    def test_invalid_colormap(self):
        """Test invalid colormap handling."""
        with self.assertRaises(ValueError):
            nice_colors(2, colormap="invalid_colormap")

    def test_invalid_interpolation_method(self):
        """Test invalid interpolation method handling."""
        with self.assertRaises(ValueError):
            nice_colors(2, interpolation_method="invalid_method")

    def test_invalid_output_format(self):
        """Test invalid output format handling."""
        with self.assertRaises(ValueError):
            nice_colors(2, output_format="invalid_format")


#########################################################################################################
# Bash color tests


class TestBashColors(unittest.TestCase):

    """Bash color utility test suite."""
    def test_purple(self):
        """Test bash_purple output."""
        result = cu.bash_purple("Test")
        self.assertEqual(result, "\033[95mTest\033[0m")

    def test_blue(self):
        """Test bash_blue output."""
        result = cu.bash_blue("Test")
        self.assertEqual(result, "\033[94mTest\033[0m")

    def test_green(self):
        """Test bash_green output."""
        result = cu.bash_green("Test")
        self.assertEqual(result, "\033[92mTest\033[0m")

    def test_yellow(self):
        """Test bash_yellow output."""
        result = cu.bash_yellow("Test")
        self.assertEqual(result, "\033[93mTest\033[0m")

    def test_red(self):
        """Test bash_red output."""
        result = cu.bash_red("Test")
        self.assertEqual(result, "\033[91mTest\033[0m")

    def test_bold(self):
        """Test bash_bold output."""
        result = cu.bash_bold("Test")
        self.assertEqual(result, "\033[1mTest\033[0m")

    def test_underline(self):
        """Test bash_underline output."""
        result = cu.bash_underline("Test")
        self.assertEqual(result, "\033[4mTest\033[0m")

#########################################################################################################
# Script entry point


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
