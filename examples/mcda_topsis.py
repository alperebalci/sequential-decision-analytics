from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TopsisResult:
    alternative: str
    score: float
    distance_to_ideal: float
    distance_to_anti_ideal: float
    rank: int


class Topsis:
    """Transparent TOPSIS implementation for mixed benefit and cost criteria."""

    def __init__(
        self,
        alternatives,
        criteria,
        decision_matrix,
        weights,
        benefit_criteria,
    ):
        self.alternatives = tuple(alternatives)
        self.criteria = tuple(criteria)
        self.matrix = np.asarray(decision_matrix, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.benefit_criteria = np.asarray(benefit_criteria, dtype=bool)

        if not self.alternatives or not self.criteria:
            raise ValueError("alternatives and criteria must be non-empty")
        if self.matrix.shape != (len(self.alternatives), len(self.criteria)):
            raise ValueError("decision_matrix shape must be alternatives x criteria")
        if self.weights.shape != (len(self.criteria),):
            raise ValueError("weights must match criteria")
        if self.benefit_criteria.shape != (len(self.criteria),):
            raise ValueError("benefit_criteria must match criteria")
        if not np.isfinite(self.matrix).all():
            raise ValueError("decision_matrix must contain finite values")
        if not np.isfinite(self.weights).all() or np.any(self.weights < 0):
            raise ValueError("weights must be finite and non-negative")
        if np.isclose(self.weights.sum(), 0.0):
            raise ValueError("at least one weight must be positive")

        self.weights = self.weights / self.weights.sum()

    def normalized_matrix(self) -> np.ndarray:
        norms = np.linalg.norm(self.matrix, axis=0)
        if np.any(np.isclose(norms, 0.0)):
            raise ValueError("criteria columns must not have zero Euclidean norm")
        return self.matrix / norms

    def weighted_matrix(self) -> np.ndarray:
        return self.normalized_matrix() * self.weights

    def ideal_points(self) -> tuple[np.ndarray, np.ndarray]:
        weighted = self.weighted_matrix()
        ideal = np.where(
            self.benefit_criteria,
            weighted.max(axis=0),
            weighted.min(axis=0),
        )
        anti_ideal = np.where(
            self.benefit_criteria,
            weighted.min(axis=0),
            weighted.max(axis=0),
        )
        return ideal, anti_ideal

    def scores(self) -> np.ndarray:
        weighted = self.weighted_matrix()
        ideal, anti_ideal = self.ideal_points()
        distance_to_ideal = np.linalg.norm(weighted - ideal, axis=1)
        distance_to_anti = np.linalg.norm(weighted - anti_ideal, axis=1)
        denominator = distance_to_ideal + distance_to_anti
        return np.divide(
            distance_to_anti,
            denominator,
            out=np.full_like(distance_to_anti, 0.5),
            where=~np.isclose(denominator, 0.0),
        )

    def ranking(self) -> tuple[TopsisResult, ...]:
        weighted = self.weighted_matrix()
        ideal, anti_ideal = self.ideal_points()
        d_plus = np.linalg.norm(weighted - ideal, axis=1)
        d_minus = np.linalg.norm(weighted - anti_ideal, axis=1)
        scores = np.divide(
            d_minus,
            d_plus + d_minus,
            out=np.full_like(d_minus, 0.5),
            where=~np.isclose(d_plus + d_minus, 0.0),
        )
        order = np.argsort(-scores, kind="stable")

        rows = []
        for rank, index in enumerate(order, start=1):
            rows.append(
                TopsisResult(
                    alternative=self.alternatives[int(index)],
                    score=float(scores[index]),
                    distance_to_ideal=float(d_plus[index]),
                    distance_to_anti_ideal=float(d_minus[index]),
                    rank=rank,
                )
            )
        return tuple(rows)


def supplier_selection_example() -> Topsis:
    return Topsis(
        alternatives=("Supplier A", "Supplier B", "Supplier C", "Supplier D"),
        criteria=("Unit cost", "Quality", "Lead time", "Reliability"),
        decision_matrix=(
            (8.8, 92.0, 7.0, 0.96),
            (8.1, 86.0, 9.0, 0.93),
            (9.4, 97.0, 6.0, 0.98),
            (8.5, 90.0, 8.0, 0.95),
        ),
        weights=(0.30, 0.30, 0.20, 0.20),
        benefit_criteria=(False, True, False, True),
    )


def main():
    model = supplier_selection_example()
    print("TOPSIS multi-criteria supplier selection")
    print()
    print(f"{'Rank':<6}{'Alternative':<14}{'Score':>10}")
    for row in model.ranking():
        print(f"{row.rank:<6}{row.alternative:<14}{row.score:>10.4f}")


if __name__ == "__main__":
    main()
