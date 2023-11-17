"""Функции преобразования координат и времени."""

from datetime import datetime, timezone
from typing import Tuple

import juliandate as jd
from astropy import units as u
from astropy.time import Time, TimeDelta
from pymap3d import Ellipsoid, geodetic2eci


def lla_to_eci(
        latitude: float,
        longitude: float,
        altitude: float,
        dt: datetime,
        ellipsoid: Ellipsoid = None
) -> Tuple[float, float, float]:
    """Перевод географических координат в координаты ИСК (ECI).

    Args:
        latitude: широта, град
        longitude: долгота, град
        altitude: высота над уровнем моря, м
        dt: время UTC
        ellipsoid: параметры Земли (по умолчанию pymap3d.Ellipsoid.from_name("wgs84"))

    Returns:
        Координаты в ИСК (ECI), м.
    """
    x, y, z = geodetic2eci(
        lat=latitude,
        lon=longitude,
        alt=altitude,
        t=dt,
        ell=ellipsoid,
        deg=True,
    )
    return x, y, z


def simulation_time_to_datetime_utc(
        time: float, start_julian_day: float
) -> datetime:
    """Перевод времени моделирования в datetime.

    Args:
        time: время от начала моделирования, с
        start_julian_day: начальная юлианская дата, дн

    Returns:
        Соответствующий текущему моменту datetime.
    """
    base = Time(val=2000, format="jyear").jd + start_julian_day
    delta = TimeDelta(time * u.s).jd
    dt = jd.to_gregorian(base + delta)
    dt = datetime(*dt, tzinfo=timezone.utc)
    return dt


def simulation_time_to_julian_day(
        time: float, start_julian_day: float
) -> float:
    """Перевод времени моделирования в юлианскую дату.

    Args:
        time: время от начала моделирования, с
        start_julian_day: начальная юлианская дата, дн

    Returns:
        Текущая юлианская дата.
    """
    # расчеты ниже равносильны start_julian_day + time / 86400
    current_julian_day = start_julian_day + TimeDelta(time * u.s).jd
    return current_julian_day
