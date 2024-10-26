# -*- coding: utf-8 -*-

"""

Module containing beautiful color schemes and some color utilities.

Authors:
    Giulia Longhi, gllonghi@yahoo.it (color schemes)
    Marco Raveri, marco.raveri@unige.it (code)

"""

# ******************************************************************************

import math

# ******************************************************************************
# Definition of the color maps:

# ------------------------------------------------------------------------------

the_gold_standard = { 0: (203.0/255.0, 15.0/255.0, 40.0/255.0),
                      1: (255.0/255.0, 165.0/255.0, 0.0),
                      2: (42.0/255.0, 46.0/255.0, 139.0/255.0),
                      3: (0.0/255.0, 153.0/255.0, 204.0/255.0),
                      4: (0.0/255.0, 221.0/255.0, 52.0/255.0),
                      5: (0.0, 0.75, 0.75),
                      6: (0.0, 0.0, 0.0),
                    }

# ------------------------------------------------------------------------------

spring_and_winter = {
                    0: (93./255., 50./255., 137./255.),
                    1: (197./255., 43./255., 135./255.),
                    2: (237./255., 120./255., 159./255.),
                    3: (241./255., 147./255., 130./255.),
                    4: (113./255., 187./255., 220./255.),
                    5: (24./255., 120./255., 187./255.),
                    }

# ------------------------------------------------------------------------------

winter_and_spring = { 0: (24./255., 120./255., 187./255.),
                      1: (113./255., 187./255., 220./255.),
                      2: (241./255., 147./255., 130./255.),
                      3: (237./255., 120./255., 159./255.),
                      4: (197./255., 43./255., 135./255.),
                      5: (93./255., 50./255., 137./255.),
                    }

# ------------------------------------------------------------------------------

summer_sun = { 0: (234./255., 185./255., 185./255.),
               1: (234./255., 90./255., 103./255.),
               2: (255./255., 231./255., 76./255.),
               3: (249./255., 179./255., 52./255.),
               4: (55./255., 97./255., 140./255.),
               5: (82./255., 158./255., 214./255.),
             }

# ------------------------------------------------------------------------------

summer_sky = { 0: (82./255., 158./255., 214./255.),
               1: (55./255., 97./255., 140./255.),
               2: (249./255., 179./255., 52./255.),
               3: (255./255., 231./255., 76./255.),
               4: (234./255., 90./255., 103./255.),
               5: (234./255., 185./255., 185./255.),
             }

# ------------------------------------------------------------------------------

autumn_fields = { 0: (50./255., 138./255., 165./255.),
                  1: (16./255., 135./255., 98./255.),
                  2: (198./255., 212./255., 60./255.),
                  3: (255./255., 251./255., 73./255.),
                  4: (237./255., 118./255., 40./255.),
                  5: (142./255., 26./255., 26./255.),
                }

# ------------------------------------------------------------------------------

autumn_leaves = { 0: (142./255., 26./255., 26./255.),
                  1: (237./255., 118./255., 40./255.),
                  2: (255./255., 251./255., 73./255.),
                  3: (198./255., 212./255., 60./255.),
                  4: (16./255., 135./255., 98./255.),
                  5: (50./255., 138./255., 165./255.),
                }

# ------------------------------------------------------------------------------

shades_of_gray = { 0: (90./255., 90./255., 90./255.)
                  }

# ******************************************************************************
# definition of color interpolation utilities:

def color_linear_interpolation( rgb_1, rgb_2, alpha ):
    """
    This function performs a linear color interpolation in RGB space.
    alpha has to go from zero to one and is the coordinate.
    """
    _out_color = []
    for _a,_b in zip(rgb_1,rgb_2):
        _out_color.append( _a +(_b-_a)*alpha )
    return tuple(_out_color)

# ******************************************************************************
# definition of the color helper:

def nice_colors(num, colormap='the_gold_standard', interpolation_method='linear', output_format='RGB_255'):
    """
    This function returns a color from a colormap defined above, according to the
    number entered.

    :param num: input number. Can be an integer or float.
        If the number is integer the function returns one of the colors in the
        colormap. If the number is a float returns the shade combining the two
        neighbouring colors.
    :type num: :class:`int` or :class:`float`

    :param colormap: a string containing the name of the colormap.
    :type colormap: :class:`string`

    :param interpolation_method: the method to interpolate between colors.
        Legal choices are:
            interpolation_method='linear', linear interpolation;
        Further interpolation methods will be added in the future.
    :type interpolation_method: :class:`string`

    :param output_format: output format of the color.
        Legal choices are:
            output_format='HEX'
            output_format='RGB'
            output_format='RGB_255' (default)
    :type output_format: :class:`string`

    :return: string with HEX color or tuple with RGB coordinates

    """
    # get the colormap:
    try:
        _cmap = globals()[str(colormap)]
    except:
        raise ValueError('Requested color map ('+str(colormap)+') does not exist.')
    # get the indexes of the color map:
    _idx_low = int(math.floor(num % len(_cmap)))
    _idx_up = int(math.floor((_idx_low+1) % len(_cmap)))
    # perform color interpolation:
    if interpolation_method == 'linear':
        _t = num % len(_cmap)-_idx_low
        _out_color = color_linear_interpolation(_cmap[_idx_low], _cmap[_idx_up], _t)
    else:
        raise ValueError('Requested color interpolation method ('+str(interpolation_method)+') does not exist.')
    # choose the output format:
    if output_format == 'HEX':
        _out_color = '#%02x%02x%02x' % tuple([int(_c*255.) for _c in _out_color])
        _out_color = str(_out_color)
    elif output_format == 'RGB':
        pass
    elif output_format == 'RGB_255':
        _out_color = tuple(int(255.*col) for col in _out_color)
    else:
        raise ValueError('Requested output format ('+str(output_format)+') does not exist.')
    #
    return _out_color

# ******************************************************************************
# Definition of bash colors:

class bash_colors:
    """
    This class contains the necessary definitions to print to bash screen with colors.
    Sometimes it can be useful and nice!

    :ivar purple: ANSI color for light purple.
    :ivar blue: ANSI color for blue.
    :ivar green: ANSI color for green.
    :ivar yellow: ANSI color for yellow.
    :ivar fail: ANSI color for red.
    :ivar bold: ANSI code for bold text.
    :ivar underline: ANSI code for underlined text.
    :ivar ENDC: ANSI code to restore the bash default.

    """

    # --------------------------------------------------------------------------

    col_purple = '\033[95m' #: ANSI color for light purple.
    col_blue = '\033[94m' #: ANSI color for blue.
    col_green = '\033[92m' #: ANSI color for green.
    col_yellow = '\033[93m' #: ANSI color for yellow.
    col_red = '\033[91m' #: ANSI color for red.
    col_bold = '\033[1m'  #: ANSI code for bold text.
    col_underline = '\033[4m'  #: ANSI code for underlined text.
    ENDC = '\033[0m'  #: ANSI code to restore the bash default.

    # --------------------------------------------------------------------------

    def __init__(self):
        pass

    # --------------------------------------------------------------------------

    def purple(self, string):
        """
        Function that returns a string that can be printed to bash in :class:`cosmicfish_pylib.colors.bash_colors.purple` color.

        :param string: input string.
        :type string: string
        :return: the input string with the relevant ANSI code at the beginning and at the end.
        :rtype: string

        """
        return self.col_purple+str(string)+self.ENDC

    # --------------------------------------------------------------------------

    def blue(self,string):
        """
        Function that returns a string that can be printed to bash in :class:`cosmicfish_pylib.colors.bash_colors.blue` color.

        :param string: input string.
        :type string: string
        :return: the input string with the relevant ANSI code at the beginning and at the end.
        :rtype: string

        """
        return self.col_blue+str(string)+self.ENDC

    # --------------------------------------------------------------------------

    def green(self,string):
        """
        Function that returns a string that can be printed to bash in :class:`cosmicfish_pylib.colors.bash_colors.green` color.

        :param string: input string.
        :type string: string
        :return: the input string with the relevant ANSI code at the beginning and at the end.
        :rtype: string

        """
        return self.col_green+str(string)+self.ENDC

    # --------------------------------------------------------------------------

    def yellow(self,string):
        """
        Function that returns a string that can be printed to bash in :class:`cosmicfish_pylib.colors.bash_colors.yellow` color.

        :param string: input string.
        :type string: string
        :return: the input string with the relevant ANSI code at the beginning and at the end.
        :rtype: string

        """
        return self.col_yellow+str(string)+self.ENDC

    # --------------------------------------------------------------------------

    def red(self,string):
        """
        Function that returns a string that can be printed to bash in :class:`cosmicfish_pylib.colors.bash_colors.fail` color.

        :param string: input string.
        :type string: string
        :return: the input string with the relevant ANSI code at the beginning and at the end.
        :rtype: string

        """
        return self.col_red+str(string)+self.ENDC

    # --------------------------------------------------------------------------

    def bold(self,string):
        """
        Function that returns a string that can be printed to bash in :class:`cosmicfish_pylib.colors.bash_colors.bold` color.

        :param string: input string.
        :type string: string
        :return: the input string with the relevant ANSI code at the beginning and at the end.
        :rtype: string

        """
        return self.col_bold+str(string)+self.ENDC

    # --------------------------------------------------------------------------

    def underline(self,string):
        """
        Function that returns a string that can be printed to bash in :class:`cosmicfish_pylib.colors.bash_colors.underline` color.

        :param string: input string.
        :type string: string
        :return: the input string with the relevant ANSI code at the beginning and at the end.
        :rtype: string

        """
        return self.col_underline+str(string)+self.ENDC

    # --------------------------------------------------------------------------

# ******************************************************************************
