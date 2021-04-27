r"""Functions for predicting blend properties"""
from typing import List
from math import log, exp

from .equations import linear_blend_ave, celsius_to_rankine, rankine_to_celsius


def cetane_number(values: List[float], vol_fractions: List[float]) -> float:
    """
    Calculates blended CN from individual CNs, volume fractions of each individual CN in blend;
    blend assumed proportionally linear: NREL/SR-540-36805

    Args:
        values (list[float]): CN values
        vol_fractions (list[float]): list of volume fractions, sum(vol_fractions) == 1.0

    Returns:
        float: blended CN
    """

    return linear_blend_ave(values, vol_fractions)


def cloud_point(values: List[float], vol_fractions: List[float]) -> float:
    """
    Calculates blended CP from individual CPs, volume fractions of each individual CP in blend;
    from paper "Predictions of pour, cloud and cold filter plugging point for future diesel
    fuels with application to diesel blending models" by Semwal et al.

    $$
    CP_{b}^{13.45} = \\sum_{i}^{N} V_{i} CP_{i}^{13.45}
    $$

    Where $$V_i$$ is the ith components weight percent, $$CP_i$$ is the ith component's CP, in
    Rankine, and $$CP_b$$ is the blend's CP, in Rankine

    Args:
        values (list[float]): CP values, in Celsius
        vol_fractions (list[float]): list of volume fractions, sum(vol_fractions) == 1.0

    Returns:
        float: blended CP, in Celsius
    """

    cp_sum = 0.0
    for idx, val in enumerate(values):
        cp_sum += (vol_fractions[idx] * celsius_to_rankine(val)**13.45)
    return rankine_to_celsius(cp_sum**(1 / 13.45))


def kinematic_viscosity(values: List[float], vol_fractions: List[float]) -> float:
    """
    Calculates blended KV from individual KVs, volume fractions of each individual KV in blend;
    equation 8 from paper "Estimation of the kinematic viscosities of bio-oil/alcohol blends:
    Kinematic viscosity-temperature formula and mixing rules" by Ding et al.

    $$
    1 / ln(2000 * kv_{blend}) = \\sum_{i}^{N} \\frac{V_i}{ln(2000 * kv_i)}
    $$

    Where $$V_i$$ is the volume fraction of the ith component, $$kv_i$$ is the kinematic viscosity
    of the ith component, and $$kv_{blend}$$ is the kinematic viscosity of the blend

    Args:
        values (list[float]): KV values, in cSt
        vol_fractions (list[float]): list of volume fractions, sum(vol_fractions) == 1.0
    """

    kv_sum = 0.0
    for idx, val in enumerate(values):
        kv_sum += (vol_fractions[idx] / log(2000 * val))
    return exp(1 / kv_sum) / 2000


def lower_heating_value(values: List[float], vol_fractions: List[float]) -> float:
    """
    Calculates blended LHV from individual LHVs, volume fractions of each individual LHV in blend;
    blend assumed proportionally linear: https://doi.org/10.1016/j.ejpe.2015.11.002

    Args:
        values (list[float]): LHV values
        vol_fractions (list[float]): list of volume fractions, sum(vol_fractions) == 1.0

    Returns:
        float: blended LHV
    """

    return linear_blend_ave(values, vol_fractions)


def yield_sooting_index(values: List[float], vol_fractions: List[float]) -> float:
    """
    Calculates blended YSI from individual YSIs, volume fractions of each individual YSI in blend;
    blend assumed proportionally linear: https://doi.org/10.1016/j.fuel.2020.119522

    Args:
        values (list[float]): YSI values
        vol_fractions (list[float]): list of volume fractions, sum(vol_fractions) == 1.0

    Returns:
        float: blended YSI
    """

    return linear_blend_ave(values, vol_fractions)
