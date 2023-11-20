"""Тестирование модуля orbit.w"""
import os
import sys
from datetime import datetime

import numpy as np
import pytest

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../"))
from src.orbit import Orbit

# Параметры орбиты в начальный момент времени.
# Координаты, м.
X = 4362521.19692133
Y = -2174459.71448059
Z = 4720847.40402189
# Скорости, м/с.
VX = 5356.39915538069
VY = 4741.41348686709
VZ = -2761.5472632395
# Начальная дата.
JULIAN_DAY = 8084.05644194318
DATETIME = datetime(
    year=2022, month=2, day=18,
    hour=13, minute=21, second=16,
)
# Долгота восходящего узла, рад.
RAAN = 3.555013635168731
# Наклонение орбиты, рад.
INC = 0.900305491058385
# Эксцентриситет, рад.
ECC = 0.0010517710777224682
# Шаг моделирования, с.
TIME_STEP = 1


@pytest.fixture
def random_raan():
    return np.pi * np.random.random()


def test_orbit_init():
    orbit = Orbit(time_step=TIME_STEP)
    orbit.init(
        x=X, y=Y, z=Z,
        vx=VX, vy=VY, vz=VZ,
        julian_day=JULIAN_DAY,
    )
    assert orbit.time_step == TIME_STEP
    assert np.isclose(X, orbit.x)
    assert np.isclose(Y, orbit.y)
    assert np.isclose(Z, orbit.z)
    assert np.isclose(VX, orbit.vx)
    assert np.isclose(VY, orbit.vy)
    assert np.isclose(VZ, orbit.vz)
    assert np.isclose(JULIAN_DAY, orbit.jd)
    assert DATETIME.year == orbit.dt.year
    assert DATETIME.month == orbit.dt.month
    assert DATETIME.day == orbit.dt.day
    assert DATETIME.hour == orbit.dt.hour
    assert DATETIME.minute == orbit.dt.minute
    assert DATETIME.second == orbit.dt.second


def test_orbit_normal_equatorial_orbit(random_raan):
    orbit = Orbit()
    # наклонение 0 градусов
    normal = orbit._orbit_normal(
        raan=random_raan, inc=0
    )
    assert np.allclose((0, 0, 1), normal)
    # наклонение 180 градусов
    normal = orbit._orbit_normal(
        raan=random_raan, inc=np.pi
    )
    assert np.allclose((0, 0, -1), normal)


def test_orbit_normal_polar_orbit():
    orbit = Orbit()
    # наклонение 90 градусов
    normal = orbit._orbit_normal(
        raan=0, inc=np.pi / 2
    )
    assert np.allclose((0, -1, 0), normal)
    # наклонение -90 градусов
    normal = orbit._orbit_normal(
        raan=0, inc=-np.pi / 2
    )
    assert np.allclose((0, 1, 0), normal)


def test_orbit_normal_initial_position():
    orbit = Orbit()
    normal = orbit._orbit_normal(
        raan=RAAN, inc=INC
    )
    assert np.isclose(1, np.linalg.norm(normal))
    assert np.isclose(0, np.dot((X, Y, Z), normal))
    assert np.isclose(0, np.dot((VX, VY, VZ), normal))


@pytest.mark.slow
def test_date_propagate_one_minute():
    orbit = Orbit()
    orbit.init(
        x=X, y=Y, z=Z,
        vx=VX, vy=VY, vz=VZ,
        julian_day=JULIAN_DAY,
    )
    orbit.step(time_step=60)
    assert np.isclose(JULIAN_DAY + 60 / 86400, orbit.jd)
    assert DATETIME.year == orbit.dt.year
    assert DATETIME.month == orbit.dt.month
    assert DATETIME.day == orbit.dt.day
    assert DATETIME.hour == orbit.dt.hour
    assert DATETIME.minute + 1 == orbit.dt.minute
    assert DATETIME.second == orbit.dt.second


@pytest.mark.slow
def test_date_propagate_one_hour():
    orbit = Orbit()
    orbit.init(
        x=X, y=Y, z=Z,
        vx=VX, vy=VY, vz=VZ,
        julian_day=JULIAN_DAY,
    )
    orbit.step(time_step=3600)
    assert np.isclose(JULIAN_DAY + 1 / 24, orbit.jd)
    assert DATETIME.year == orbit.dt.year
    assert DATETIME.month == orbit.dt.month
    assert DATETIME.day == orbit.dt.day
    assert DATETIME.hour + 1 == orbit.dt.hour
    assert DATETIME.minute == orbit.dt.minute
    assert DATETIME.second == orbit.dt.second


@pytest.mark.slow
def test_date_propagate_one_day():
    orbit = Orbit()
    orbit.init(
        x=X, y=Y, z=Z,
        vx=VX, vy=VY, vz=VZ,
        julian_day=JULIAN_DAY,
    )
    orbit.step(time_step=86400)
    assert np.isclose(JULIAN_DAY + 1, orbit.jd)
    assert DATETIME.year == orbit.dt.year
    assert DATETIME.month == orbit.dt.month
    assert DATETIME.day + 1 == orbit.dt.day
    assert DATETIME.hour == orbit.dt.hour
    assert DATETIME.minute == orbit.dt.minute
    assert DATETIME.second == orbit.dt.second


@pytest.mark.slow
def test_propagate_without_perturbation():
    orbit = Orbit()
    orbit.init(
        x=X, y=Y, z=Z,
        vx=VX, vy=VY, vz=VZ,
        julian_day=JULIAN_DAY,
    )
    orbit.step(time_step=3600, use_perturbation=False)
    assert RAAN == orbit.orb.raan.value
    assert INC == orbit.orb.inc.value
    assert ECC == orbit.orb.ecc.value


@pytest.mark.slow
def test_propagate_with_perturbation():
    orbit = Orbit()
    orbit.init(
        x=X, y=Y, z=Z,
        vx=VX, vy=VY, vz=VZ,
        julian_day=JULIAN_DAY,
    )
    orbit.step(time_step=3600, use_perturbation=True)
    assert not np.isclose(RAAN, orbit.orb.raan.value)
    assert not np.isclose(INC, orbit.orb.inc.value)
    assert not np.isclose(ECC, orbit.orb.ecc.value)
