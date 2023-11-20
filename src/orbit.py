"""Орбита спутника."""

from datetime import datetime, timezone
from typing import Tuple

import juliandate as jd
import numpy as np
from astropy import units as u
from astropy.time import Time, TimeDelta
from poliastro import twobody
from poliastro.bodies import Earth
from poliastro.core.perturbations import J2_perturbation
from poliastro.core.propagation import func_twobody

# Время, относительно которого считаем Юлианский день.
BASE_TIME = Time(val=2000, format="jyear")


class Orbit:
    """Орбита спутника."""

    def __init__(self, time_step: float = 1.0):
        """Орбита спутника.

        Args:
            time_step: шаг моделирования, с.
        """
        # Шаг моделирования.
        self.time_step = time_step
        # Орбита спутника.
        self.orb = None
        # Координаты спутника в ИСК, м.
        self.x, self.y, self.z = None, None, None
        # Скорости спутника в ИСК, м / c.
        self.vx, self.vy, self.vz = None, None, None
        # Текущий юлианский день.
        self.jd = None
        # Текущие дата-время.
        self.dt = None
        # Нормаль к орбите.
        self.normal = None

    def init(
            self,
            x: float, y: float, z: float,
            vx: float, vy: float, vz: float,
            julian_day: float,
    ):
        """Задание орбиты.

        Args:
            x: координата X в ИСК, м.
            y: координата Y в ИСК, м.
            z: координата Z в ИСК, м.
            vx: скорость спутника по оси X в ИСК, м / с.
            vy: скорость спутника по оси Y в ИСК, м / с.
            vz: скорость спутника по оси Z в ИСК, м / с.
            julian_day: юлианский день от 2000-го года.
        """
        r = [x, y, z] << u.m
        v = [vx, vy, vz] << u.m / u.s
        epoch = BASE_TIME + TimeDelta(julian_day << u.d)
        self.orb = twobody.Orbit.from_vectors(
            attractor=Earth,
            r=r,
            v=v,
            epoch=epoch,
        )
        self._set_state_vector()
        self.normal = self._orbit_normal(
            raan=self.orb.raan.value, inc=self.orb.inc.value,
        )

    def step(self, time_step: float = None, use_perturbation: bool = True):
        """Расчет параметров орбиты на следующий временной шаг.

        Args:
            time_step: шаг моделирования, с.
            use_perturbation: учет возмущений (только J2).
        """

        if time_step is None:
            time_step = self.time_step

        if use_perturbation:
            self.orb = self.orb.propagate(
                time_step * u.s,
                method=twobody.propagation.CowellPropagator(f=self._f_perturbation),
            )
        else:
            self.orb = self.orb.propagate(time_step * u.s)

        # Вектор-состояния.
        self._set_state_vector()
        # Нормаль к орбите.
        self.normal = self._orbit_normal(
            raan=self.orb.raan.value, inc=self.orb.inc.value,
        )

    def _orbit_normal(
            self, raan: float, inc: float
    ) -> Tuple[float, float, float]:
        """Нормаль к орбите.

        Args:
            raan: долгота восходящего узла, рад.
            inc: наклонение орбиты, рад.

        Returns:
            Вектор нормали к орбите.
        """
        # Матрица поворота вокруг оси Z на угол,
        # равный долготе восходящего узла.
        rot_z = np.array([
            [np.cos(raan), -np.sin(raan), 0.],
            [np.sin(raan), np.cos(raan), 0.],
            [0., 0., 1.],
        ])
        # Матрица поворота вокруг оси X на угол,
        # равный наклонению орбиты.
        rot_x = np.array([
            [1., 0., 0.],
            [0., np.cos(inc), -np.sin(inc)],
            [0., np.sin(inc), np.cos(inc)],
        ])
        normal = rot_z @ rot_x @ np.array([0, 0, 1])
        normal = tuple(normal)
        return normal

    def _f_perturbation(self, t0, u_, k):
        """Возмущения, только J2."""
        du_kep = func_twobody(t0, u_, k)
        ax, ay, az = J2_perturbation(
            t0, u_, k, J2=Earth.J2.value, R=Earth.R.to(u.km).value
        )
        du_ad = np.array([0, 0, 0, ax, ay, az])
        return du_kep + du_ad

    def _set_state_vector(self):
        """Вектор-состояния спутника."""
        # Координаты в ИСК, м.
        self.x, self.y, self.z = self.orb.r.to(u.m).value
        # Скорости в ИСК, м / c.
        self.vx, self.vy, self.vz = self.orb.v.to(u.m / u.s).value
        # Юлианский день.
        self.jd = self.orb.epoch.jd - BASE_TIME.jd
        # Дата-время.
        self.dt = datetime(
            *jd.to_gregorian(self.orb.epoch.jd),
            tzinfo=timezone.utc
        )
