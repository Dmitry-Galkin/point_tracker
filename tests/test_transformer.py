"""Тестирование модуля transformer."""

import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../"))
from src.transformer import (
    lla_to_eci,
    simulation_time_to_datetime_utc,
    simulation_time_to_julian_day
)

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


def test_simulation_time_to_datetime_utc_start_point():
    dt = simulation_time_to_datetime_utc(
        time=0,
        start_julian_day=START_JULIAN_DAY,
    )
    assert START_DATE.year == dt.year
    assert START_DATE.month == dt.month
    assert START_DATE.day == dt.day
    assert START_DATE.hour == dt.hour
    assert START_DATE.minute == dt.minute
    assert START_DATE.second == dt.second


def test_simulation_time_to_datetime_utc_after_one_hour():
    dt = simulation_time_to_datetime_utc(
        time=3600,
        start_julian_day=START_JULIAN_DAY,
    )
    assert START_DATE.year == dt.year
    assert START_DATE.month == dt.month
    assert START_DATE.day == dt.day
    assert START_DATE.hour + 1 == dt.hour
    assert START_DATE.minute == dt.minute
    assert START_DATE.second == dt.second


def test_simulation_time_to_datetime_utc_after_one_day():
    dt = simulation_time_to_datetime_utc(
        time=86400,
        start_julian_day=START_JULIAN_DAY,
    )
    assert START_DATE.year == dt.year
    assert START_DATE.month == dt.month
    assert START_DATE.day + 1 == dt.day
    assert START_DATE.hour == dt.hour
    assert START_DATE.minute == dt.minute
    assert START_DATE.second == dt.second


def test_simulation_time_to_julian_day_start():
    jd = simulation_time_to_julian_day(
        time=0,
        start_julian_day=START_JULIAN_DAY,
    )
    assert START_JULIAN_DAY == jd


def test_simulation_time_to_julian_after_one_hour():
    jd = simulation_time_to_julian_day(
        time=3600,
        start_julian_day=START_JULIAN_DAY,
    )
    assert START_JULIAN_DAY + 1 / 24 == jd


def test_simulation_time_to_julian_after_one_day():
    jd = simulation_time_to_julian_day(
        time=86400,
        start_julian_day=START_JULIAN_DAY,
    )
    assert START_JULIAN_DAY + 1 == jd
