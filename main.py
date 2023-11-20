"""Запуск моделирования."""

from src.simulation import Simulation

# Время моделирования, с.
SIMULATION_TIME = 86400
# Шаг моделирования, с.
TIME_STEP = 60
# Учет возмущений.
USE_PERTURBATION = True

if __name__ == "__main__":

    print("\nSimulation started...", flush=True)

    sim = Simulation(filename_with_result="default.txt")
    sim.run(
        simulation_time=SIMULATION_TIME,
        time_step=TIME_STEP,
        use_perturbation=USE_PERTURBATION,
    )

    print("Finished", flush=True)
