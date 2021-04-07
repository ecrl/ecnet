r"""Helper equations for functions in .predict.py"""


def _celsius_to_rankine(temp: float) -> float:
    r'''
    Converts temperature in celsius to temperature in rankine

    Args:
        temp (float): supplied temperature, in celsius

    Returns:
        float: temperature in rankine
    '''

    return (9 / 5) * temp + 491.67


def _linear_blend_ave(values: list, proportions: tuple) -> float:
    ''' Calculates the linear combination of multiple values given discrete
    proportions for each value

    Args:
        values (list): list of values to form linear average
        proportions (tuple): proportions of each value in `values`; should sum
            to 1; len(proportions) == len(values)

    Returns:
        float: weighted linear average
    '''

    weighted_ave = 0
    for idx, proportion in enumerate(proportions):
        weighted_ave += (proportion * values[idx])
    return weighted_ave


def _rankine_to_celsius(temp: float) -> float:
    ''' Converts temperature in rankine to temperature in celsius

    Args:
        temp (float): temperature in rankine

    Returns:
        float: temperature in celsius
    '''

    return (temp - 491.67) * (1 / (9 / 5))
