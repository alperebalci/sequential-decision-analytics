from itertools import product

from online_or import (
    deterministic_ski_rental_cost,
    list_schedule,
    offline_identical_machine_optimum,
    ski_rental_offline_cost,
)


def test_deterministic_ski_rental_is_strictly_below_ratio_two():
    buy_cost = 7
    worst = 0.0
    for days in range(1, 5 * buy_cost + 1):
        online = deterministic_ski_rental_cost(days, buy_cost)
        offline = ski_rental_offline_cost(days, buy_cost)
        worst = max(worst, online / offline)
    assert worst == (2 * buy_cost - 1) / buy_cost
    assert worst < 2.0


def test_list_scheduling_respects_classical_bound_exhaustively_for_small_sequences():
    for machines in (2, 3):
        bound = 2.0 - 1.0 / machines
        for n_jobs in range(1, 6):
            for jobs in product((1, 2, 3), repeat=n_jobs):
                online = list_schedule(jobs, machines)
                offline = offline_identical_machine_optimum(jobs, machines)
                ratio = online.makespan / offline.makespan
                assert ratio <= bound + 1e-12


def test_offline_oracle_never_worse_than_online_schedule():
    jobs = [3, 1, 4, 2, 5]
    online = list_schedule(jobs, 2)
    offline = offline_identical_machine_optimum(jobs, 2)
    assert offline.makespan <= online.makespan
