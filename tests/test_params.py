"""Тестирование модуля params."""

import os
import sys

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../"))
from src.params import read_params

PARAMS_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "config", "params.yaml"
)


def test_read_params_all_attribute_exist():

    params = read_params(PARAMS_PATH)

    assert hasattr(params, "earth_target")
    assert hasattr(params.earth_target, "latitude")
    assert hasattr(params.earth_target, "longitude")
    assert hasattr(params.earth_target, "altitude")

    assert hasattr(params, "satellite_initial_position_eci")
    assert hasattr(params.satellite_initial_position_eci, "x")
    assert hasattr(params.satellite_initial_position_eci, "y")
    assert hasattr(params.satellite_initial_position_eci, "z")
    assert hasattr(params.satellite_initial_position_eci, "vx")
    assert hasattr(params.satellite_initial_position_eci, "vy")
    assert hasattr(params.satellite_initial_position_eci, "vz")
    assert hasattr(params.satellite_initial_position_eci, "julian_day")
