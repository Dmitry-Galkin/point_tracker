"""Моделирование движения КА по орбите и построение целевой ориентации."""

import os
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from src.frames import lla_to_eci, target_frame
from src.orbit import Orbit
from src.params import read_params

PARAMS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "params.yaml",
)


class Simulation:
    """Моделирование движения КА по орбите и построение целевой ориентации."""

    def __init__(self, filename_with_result: Optional[str] = None):
        """Моделирование движения КА по орбите и построение целевой ориентации.

        Args:
            filename_with_result: имя файла для записи результатов моделирования.
        """
        # Считывание параметров моделирования из yaml-файла.
        self.params = read_params(PARAMS_PATH)
        # Файл, куда будут записываться результаты.
        if filename_with_result is None:
            filename_with_result = "default.txt"
        self.filename_with_result = os.path.join(
            os.path.dirname(__file__), "..", "results", filename_with_result
        )
        # Орбита.
        self.orbit = None
        # Целевая система координат.
        self.tf = None
        # Кватернион целевой ориентации.
        self.quat = None

    def run(
            self,
            simulation_time: float,
            time_step: float,
            use_perturbation: bool = True,
    ):
        """Моделирование.

        Args:
            simulation_time: время моделирования, с.
            time_step: шаг моделирования, с.
            use_perturbation: учет возмущений при прогнозе орбиты (только J2).
        """

        # Задание орбиты.
        self.init_orbit()
        # Создание файла, куда будут записываться результаты моделирования.
        self.create_file_with_result()

        t_arr = np.arange(0, simulation_time + time_step, time_step)
        with tqdm(total=simulation_time) as pbar:
            for i, _ in enumerate(t_arr):
                # Формирование целевой системы координат.
                self.tf = self.build_target_frame()
                # Формирование целевого кватерниона.
                self.quat = self.target_quaternion(tf=self.tf)
                # Запись в файл.
                self.write_result_to_file()
                # Интегрирование, следующий шаг.
                self.orbit.step(
                    time_step=time_step, use_perturbation=use_perturbation,
                )
                if i < t_arr.size - 1:
                    pbar.update(time_step)

    def target_quaternion(self, tf: np.ndarray) -> np.ndarray:
        """Расчет кватерниона целевой ориентации КА.

        Args:
            tf: матрица, описывающая целевую СК.

        Returns:
            Кватернион целевой ориентации КА.

        Вид матрицы, описывающей целевую СК: tf = [[x.T], [y.T], [z.T]].
        Первая строка - координаты оси X целевой СК в ИСК и т.д.
        """
        # tf.T - матрица перехода из целевой СК в ИСК.
        quat = R.from_matrix(tf.T).as_quat()
        # В более привычный вид: скалярная часть + векторная.
        quat = quat[[3, 0, 1, 2]]
        return quat

    def build_target_frame(self) -> np.ndarray:
        """Построение целевой системы координат."""
        # Координаты точки на Земле в ИСК.
        r_target = lla_to_eci(
            latitude=self.params.earth_target.latitude,
            longitude=self.params.earth_target.longitude,
            altitude=self.params.earth_target.altitude,
            dt=self.orbit.dt,
        )
        # Целевая система координат.
        tf = target_frame(
            r_sat=(self.orbit.x, self.orbit.y, self.orbit.z),
            r_target=r_target,
            orbit_normal=self.orbit.normal,
        )
        return tf

    def init_orbit(self):
        """Задание параметров орбиты."""
        self.orbit = Orbit()
        self.orbit.init(
            x=self.params.satellite_initial_position_eci.x,
            y=self.params.satellite_initial_position_eci.y,
            z=self.params.satellite_initial_position_eci.z,
            vx=self.params.satellite_initial_position_eci.vx,
            vy=self.params.satellite_initial_position_eci.vy,
            vz=self.params.satellite_initial_position_eci.vz,
            julian_day=self.params.satellite_initial_position_eci.julian_day,
        )

    def create_file_with_result(self):
        """Создание текстового файла для записи результатов моделирования."""
        with open(f"{self.filename_with_result}", "w", encoding="utf-8") as f:
            f.write("jd x y z vx vy vx q0 q1 q2 q3\n")

    def write_result_to_file(self):
        """Запись результатов моделирования в текстовый файл."""
        line = (f"{self.orbit.jd} "
                f"{self.orbit.x} {self.orbit.y} {self.orbit.z} "
                f"{self.orbit.vx} {self.orbit.vy} {self.orbit.vz} "
                f"{self.quat[0]} {self.quat[1]} {self.quat[2]} {self.quat[3]}"
                f"\n")
        with open(f"{self.filename_with_result}", "a", encoding="utf-8") as f:
            f. write(line)
