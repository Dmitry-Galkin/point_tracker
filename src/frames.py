"""Функции работы с системами координат."""

from datetime import datetime
from typing import Tuple

import numpy as np
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


def target_frame(
        r_sat: Tuple[float, float, float],
        r_target: Tuple[float, float, float],
        orbit_normal: Tuple[float, float, float],
) -> np.ndarray:
    """Целевая система координат.

    Args:
        r_sat: радиус-вектор спутника в ECI, м.
        r_target: радиус-вектор целевой точки на Земле, м.
        orbit_normal: вектор нормали к орбите.

    Returns:
        Координаты целелевой системы координат XYZ.
        1-ая строка матрицы - координаты вектора X и т.д.
    """

    eps = 1e-9  # в случае деления на ноль

    # Ось Y направлена от целевой точки к аппарату.
    y = np.array(r_sat) - np.array(r_target)
    y /= np.linalg.norm(y)

    # Ось Z лежит в плоскости, образуемой нормалью к орбите и осью Y.
    # Направлена в сторону от нормали к орбите.
    z = np.zeros(3)
    # Коэффициенты уравнения плоскости a * Zx + b * Zy + c * Zz = 0,
    # где лежат все вектора (из условия компланарности).
    a = y[2] * orbit_normal[1] - y[1] * orbit_normal[2]
    b = y[0] * orbit_normal[2] - y[2] * orbit_normal[0]
    c = y[1] * orbit_normal[0] - y[0] * orbit_normal[1]
    # Коэффициенты уравнений Zx = d * Zz и Zy = e * Zz.
    # Выводятся из условия перпендикулряности векторов y и z.
    d = (b * y[2] - c * y[1]) / (a * y[1] - b * y[0])
    e = (c * y[0] - a * y[2]) / (a * y[1] - b * y[0])
    # Знак координаты Zz, определяющий,
    # что вектор Z направлен в противоположную сторону от нормали.
    # Выводится из условия, что скалярное произведение векторов Z и нормали отрицательно.
    sgn = -np.sign(d * orbit_normal[0] + e * orbit_normal[1] + orbit_normal[2])
    z[2] = sgn * np.sqrt(1 / (1 + d ** 2 + e ** 2))
    z[1] = (c * y[0] - a * y[2]) / (a * y[1] - b * y[0]) * z[2]
    z[0] = -(b * z[1] + c * z[2]) / (a + eps)
    z /= np.linalg.norm(z)

    # Ось X дополняет тройку до правой и направлена в сторону скорости аппарата.
    x = np.cross(y, z)
    x /= np.linalg.norm(x)

    frame = np.vstack((x, y, z))

    # Перед нормировкой поменяем местами оси X и Y,
    # т.к., в моем понимании, правильнее относительно нее ортоганализировать.
    frame = gram_schmidt(frame[[1, 0, 2]])
    # Вернем все восвояси.
    frame = frame[[1, 0, 2]]

    return frame


def gram_schmidt(matrix: np.ndarray) -> np.ndarray:
    """Ортогонализвция Грамма-Шмидта."""
    for i in range(matrix.shape[0]):
        q = matrix[i, :]  # i-th column of A
        for j in range(i):
            q = q - np.dot(matrix[j, :], matrix[i, :]) * matrix[j, :]
        # normalize q
        q = q / np.sqrt(np.dot(q, q))
        # write the vector back in the matrix
        matrix[i, :] = q
    return matrix
