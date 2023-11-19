"""Тестирование модуля transformer."""

import os
import sys
from datetime import datetime

import numpy as np
import pytest

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../"))
from src.frames import (lla_to_eci, target_frame)

# Примерный радиус Земли на экваторе, км.
EARTH_RADIUS_EQUATOR = 6378
# Примерный радиус Земли на полюсе, км.
EARTH_RADIUS_POLE = 6357
# Начальный момент времени.
START_JULIAN_DAY = 8084.05644194318
START_DATE = datetime(
    year=2022, month=2, day=18,
    hour=13, minute=21, second=16, microsecond=584,
)


@pytest.fixture
def radius_vector_satellite():
    r = np.random.random(3)
    r /= np.linalg.norm(r)
    r *= 1e6
    return tuple(r)


@pytest.fixture
def radius_vector_target():
    r = np.random.random(3)
    r /= np.linalg.norm(r)
    r *= 1e5
    return tuple(r)


@pytest.fixture
def radius_vector_normal():
    r = np.random.random(3)
    r /= np.linalg.norm(r)
    return tuple(r)


def test_lla_to_eci_equator():
    x, y, z = lla_to_eci(
        latitude=0,
        longitude=0,
        altitude=0,
        dt=datetime.now()
    )
    assert np.isclose(
        EARTH_RADIUS_EQUATOR,
        np.round(1e-3 * np.sqrt(x ** 2 + y ** 2 + z ** 2))
    )
    assert np.abs(z) < np.abs(x)
    assert np.abs(z) < np.abs(y)


def test_lla_to_eci_north_pole():
    x, y, z = lla_to_eci(
        latitude=90,
        longitude=0,
        altitude=0,
        dt=datetime.now()
    )
    assert np.isclose(
        EARTH_RADIUS_POLE,
        np.round(1e-3 * np.sqrt(x ** 2 + y ** 2 + z ** 2))
    )
    assert np.abs(z) > np.abs(x)
    assert np.abs(z) > np.abs(y)
    assert z > 0


def test_lla_to_eci_south_pole():
    x, y, z = lla_to_eci(
        latitude=-90,
        longitude=0,
        altitude=0,
        dt=datetime.now()
    )
    assert np.isclose(
        EARTH_RADIUS_POLE,
        np.round(1e-3 * np.sqrt(x ** 2 + y ** 2 + z ** 2))
    )
    assert np.abs(z) > np.abs(x)
    assert np.abs(z) > np.abs(y)
    assert z < 0


def test_target_frame_simple_case():

    r_sat = (2., 0, 0)
    r_target = (1., 0, 0)
    orbit_normal = (0, 0, 1.)

    tf = target_frame(
        r_sat=r_sat,
        r_target=r_target,
        orbit_normal=orbit_normal
    )
    x, y, z = tf[0], tf[1], tf[2]

    assert np.allclose((0, 1., 0), x)
    assert np.allclose((1., 0, 0), y)
    assert np.allclose((0, 0, -1.), z)


def test_target_frame_random_case(
        radius_vector_satellite,
        radius_vector_target,
        radius_vector_normal
):
    tf = target_frame(
        r_sat=radius_vector_satellite,
        r_target=radius_vector_target,
        orbit_normal=radius_vector_normal,
    )
    x, y, z = tf[0], tf[1], tf[2]

    # Проверка, что все вектора имеют единичную длину.
    assert np.isclose(1, np.linalg.norm(x))
    assert np.isclose(1, np.linalg.norm(y))
    assert np.isclose(1, np.linalg.norm(z))
    # Проверка, что все оси попарно ортогональны.
    assert np.isclose(0, np.dot(x, y))
    assert np.isclose(0, np.dot(x, z))
    assert np.isclose(0, np.dot(y, z))
    # Проверка, что полученная СК правая.
    assert np.allclose(x, np.cross(y, z))
    assert np.allclose(y, np.cross(z, x))
    assert np.allclose(z, np.cross(x, y))
    assert np.isclose(1, np.linalg.det(tf))
    # Проверка, что вектора Y, Z и нормали лежат в одной плоскости
    assert np.isclose(0, np.linalg.det(np.vstack((y, z, radius_vector_normal))))
    # Проверка, что ось Z и вектор нормали смотрят в противоположные стороны.
    assert np.dot(z, radius_vector_normal) < 0
