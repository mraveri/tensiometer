###############################################################################
# initial imports:

import unittest

from tensiometer.utilities.color_utilities import color_linear_interpolation, nice_colors
import tensiometer.utilities.color_utilities as cu

###############################################################################

class TestColorUtilities(unittest.TestCase):

    def test_color_linear_interpolation(self):
        rgb_1 = (1.0, 0.0, 0.0)
        rgb_2 = (0.0, 1.0, 0.0)
        alpha = 0.5
        expected_color = (0.5, 0.5, 0.0)
        result = color_linear_interpolation(rgb_1, rgb_2, alpha)
        self.assertEqual(result, expected_color)

    def test_nice_colors_integer(self):
        # Test when input is an integer
        color = nice_colors(2, colormap='the_gold_standard', interpolation_method='linear', output_format='RGB_255')
        self.assertEqual(color, (42, 46, 139))  # Expected RGB in 255 format

    def test_nice_colors_float(self):
        # Test when input is a float for interpolation
        color = nice_colors(2.5, colormap='the_gold_standard', interpolation_method='linear', output_format='RGB_255')
        self.assertIsInstance(color, tuple)

    def test_nice_colors_hex_output(self):
        # Test HEX output format
        color = nice_colors(2.5, colormap='the_gold_standard', output_format='HEX')
        self.assertTrue(color.startswith('#'))
        
    def test_nice_colors_rgb_output(self):
        # Test RGB output format
        color = nice_colors(2.5, colormap='the_gold_standard', output_format='RGB')
        self.assertIsInstance(color, tuple)

    def test_invalid_colormap(self):
        # Test invalid colormap
        with self.assertRaises(ValueError):
            nice_colors(2, colormap='invalid_colormap')

    def test_invalid_interpolation_method(self):
        # Test invalid interpolation method
        with self.assertRaises(ValueError):
            nice_colors(2, interpolation_method='invalid_method')

    def test_invalid_output_format(self):
        # Test invalid output format
        with self.assertRaises(ValueError):
            nice_colors(2, output_format='invalid_format')


class TestBashColors(unittest.TestCase):

    def test_purple(self):
        result = cu.bash_purple('Test')
        self.assertEqual(result, '\033[95mTest\033[0m')

    def test_blue(self):
        result = cu.bash_blue('Test')
        self.assertEqual(result, '\033[94mTest\033[0m')

    def test_green(self):
        result = cu.bash_green('Test')
        self.assertEqual(result, '\033[92mTest\033[0m')

    def test_yellow(self):
        result = cu.bash_yellow('Test')
        self.assertEqual(result, '\033[93mTest\033[0m')

    def test_red(self):
        result = cu.bash_red('Test')
        self.assertEqual(result, '\033[91mTest\033[0m')

    def test_bold(self):
        result = cu.bash_bold('Test')
        self.assertEqual(result, '\033[1mTest\033[0m')

    def test_underline(self):
        result = cu.bash_underline('Test')
        self.assertEqual(result, '\033[4mTest\033[0m')

###############################################################################

if __name__ == '__main__':
    unittest.main(verbosity=2)
