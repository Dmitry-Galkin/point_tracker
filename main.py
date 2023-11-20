"""Запуск моделирования."""

from src.simulation import Simulation

# Время моделирования, с.
SIMULATION_TIME = 60
# Шаг моделирования, с.
TIME_STEP = 60
# Учет возмущений.
USE_PERTURBATION = True
# Имя файла, куда записываются результаты (лежит в папке results).
FILENAME_WITH_RESULTS = "default.txt"

if __name__ == "__main__":

    print("\nSimulation started...", flush=True)

    sim = Simulation(filename_with_result=FILENAME_WITH_RESULTS)
    sim.run(
        simulation_time=SIMULATION_TIME,
        time_step=TIME_STEP,
        use_perturbation=USE_PERTURBATION,
    )

    print("Finished", flush=True)
    print()
