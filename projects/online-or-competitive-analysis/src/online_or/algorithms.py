"""Two canonical online algorithms with exact small-instance comparators."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


def ski_rental_offline_cost(days: int, buy_cost: int) -> int:
    if days < 0 or buy_cost <= 0:
        raise ValueError("days must be nonnegative and buy_cost positive")
    return min(days, buy_cost)


def deterministic_ski_rental_cost(days: int, buy_cost: int) -> int:
    """Rent until day B-1, then buy on day B if the season continues."""
    if days < 0 or buy_cost <= 0:
        raise ValueError("days must be nonnegative and buy_cost positive")
    if days < buy_cost:
        return days
    return (buy_cost - 1) + buy_cost


@dataclass(frozen=True)
class MachineSchedule:
    assignments: tuple[int, ...]
    loads: tuple[int, ...]

    @property
    def makespan(self) -> int:
        return max(self.loads, default=0)


def list_schedule(processing_times: list[int] | tuple[int, ...], machines: int) -> MachineSchedule:
    """Online list scheduling: dispatch each arriving job to a least-loaded machine."""
    if machines < 1:
        raise ValueError("machines must be positive")
    if any(p <= 0 for p in processing_times):
        raise ValueError("processing times must be positive")

    loads = [0] * machines
    assignments: list[int] = []
    for p in processing_times:
        machine = min(range(machines), key=lambda i: (loads[i], i))
        assignments.append(machine)
        loads[machine] += int(p)
    return MachineSchedule(tuple(assignments), tuple(loads))


def offline_identical_machine_optimum(
    processing_times: list[int] | tuple[int, ...],
    machines: int,
) -> MachineSchedule:
    """Exact enumeration oracle for small P||Cmax instances."""
    if machines < 1:
        raise ValueError("machines must be positive")
    if any(p <= 0 for p in processing_times):
        raise ValueError("processing times must be positive")
    jobs = tuple(int(p) for p in processing_times)
    best: MachineSchedule | None = None
    for assignment in product(range(machines), repeat=len(jobs)):
        loads = [0] * machines
        for job, machine in enumerate(assignment):
            loads[machine] += jobs[job]
        candidate = MachineSchedule(tuple(assignment), tuple(loads))
        if best is None or candidate.makespan < best.makespan:
            best = candidate
    assert best is not None
    return best
