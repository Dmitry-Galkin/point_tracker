"""Тестирование расчета кватерниона ориентации."""

# Как проверить, что кватернион ориентации построен правильно?
# Т.к. он задает переход из ССК в ИСК, то у меня идея простая:
#
# * У нас считается на каждом шаге ССК в виде координат осей в ИСК.
#   Правильность построения ССК проверяется тестами в файле test_frames.py.
#
# * Если мы будем переводить в ИСК единичные вектора, лежащие вдоль осей ССК,
#   c помощью полученного кватерниона, то результат должен совпасть
#   с координатами векторов, задающих оси ССК в ИСК.
#
# * Дополнительно будем переводить из ССК в ИСК собственный вектор матрицы поворота.
#   Его координаты не должны поменяться.
#   Что тоже будет свидетельствовать о корректности построения кватерниона.
#
# И такую проверку сделаем в каждой точки орбиты.

import os
import sys

import numpy as np
import pytest
from pyquaternion import Quaternion

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/../"))
from src.simulation import Simulation

TIME_STEP = 30
SIMULATION_TIME = 86400


@pytest.mark.slow
def test_target_quaternion_calculation():
    """Проверка правильности расчета кватерниона ориентации."""

    sim = Simulation()
    sim.init_orbit()

    for _ in np.arange(0, SIMULATION_TIME + TIME_STEP, TIME_STEP):

        # Построение ССК.
        tf = sim.build_target_frame()

        # Поиск собственного вектора, соответствующего собственному числу = 1.
        eigenvalues, eigenvectors = np.linalg.eig(tf.T)
        idx = np.where(np.isclose(eigenvalues.real, 1))[0]
        eigenvector = eigenvectors[:, idx].flatten().real

        quat = Quaternion(sim.target_quaternion(tf))

        # Ось X ССК.
        assert np.allclose(tf[0], quat.rotate((1, 0, 0)))
        # Ось Y ССК.
        assert np.allclose(tf[1], quat.rotate((0, 1, 0)))
        # Ось Z ССК.
        assert np.allclose(tf[2], quat.rotate((0, 0, 1)))
        # Собственный вектор.
        assert np.allclose(eigenvector, quat.rotate(eigenvector))
        # Норма кватерниона = 1.
        assert np.isclose(1, np.linalg.norm(sim.target_quaternion(tf)))

        sim.orbit.step(time_step=TIME_STEP)
        