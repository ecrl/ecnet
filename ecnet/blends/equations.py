r"""Helper equations for functions in .predict.py"""
from typing import List
from math import sqrt


def celsius_to_rankine(temp: float) -> float:
    """
    Converts temperature in celsius to temperature in rankine

    Args:
        temp (float): supplied temperature, in celsius

    Returns:
        float: temperature in rankine
    """

    return (9 / 5) * temp + 491.67


def linear_blend_ave(values: List[float], proportions: List[float]) -> float:
    """
    Calculates the linear combination of multiple values given discrete
    proportions for each value

    Args:
        values (list[float]): list of values to form linear average
        proportions (list[float]): proportions of each value in `values`; should sum
            to 1; len(proportions) == len(values)

    Returns:
        float: weighted linear average
    """

    weighted_ave = 0
    for idx, proportion in enumerate(proportions):
        weighted_ave += (proportion * values[idx])
    return weighted_ave


def linear_blend_err(errors: List[float], proportions: List[float]) -> float:
    """
    Calculates the linear combination of multiple errors given discrete
    proportions for each value

    $$ f = aA \\rightarrow error^2 = a^2 * error_A^2 \\rightarrow error^2 \\rightarrow \\sum $$

    Args:
        errors (list[float]): list of error values
        proportions (list[float]): proportions of each value in `values`; should sum
            to 1; len(proportions) == len(errors)

    Returns:
        float: weighted linear error
    """

    total_error = 0.0
    for idx, err in enumerate(errors):
        total_error += (err * proportions[idx])**2
    return sqrt(total_error)


def exponential_blend_err(values: List[float], result: float, errors: List[float],
                          proportions: List[float], a: float, b: float) -> float:
    """
    Calculates the error of a blend whose equation is defined as $$ f = aA^b $$

    $$ f = aA^b \\rightarrow err_f^2 = (abA^{b-1}err_A)^2 = (fberr_A / A)^2
    \\rightarrow \\sum $$

    Args:
        values (list[float]): predicted values
        result (float): resulting blend property value
        errors (list[float]): errors for predicted values in `values`
        proportions (list[float]): contribution of each value to blend; sum = 1
        a (float): scalar coefficient preceeding variable A
        b (float): exponential coefficient which A is raised to

    Returns:
        float: weighed exponential error
    """

    total_error = 0.0
    for idx, err in enumerate(errors):
        total_error += (((result * b * err) / values[idx]) * proportions[idx])**2
    return sqrt(total_error)


def kv_error(values: List[float], errors: List[float], proportions: List[float]) -> float:
    """
    Calculates the error of a KV blend whose equation is in the form $$ f=aln(bA) $$

    $$ f = aln(bA) \\rightarrow err_f^2 = (a * err_A / A)^2 \\rightarrow \\sum $$
    for KV, $$a = 1.0, b = 2000$$

    Args:
        values (list[float]): predicted values
        errors (list[float]): errors for predicted values in `values`
        proportions (list[float]): contribution of each value to blend; sum = 1

    Returns:
        float: weighted inverse logarithmic error for KV blend
    """

    total_error = 0.0
    for idx, err in enumerate(errors):
        total_error += (proportions[idx] * err / values[idx])**2
    return sqrt(total_error)


def rankine_to_celsius(temp: float) -> float:
    """
    Converts temperature in rankine to temperature in celsius

    Args:
        temp (float): temperature in rankine

    Returns:
        float: temperature in celsius
    """

    return (temp - 491.67) * (1 / (9 / 5))
