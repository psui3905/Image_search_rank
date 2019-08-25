#!/usr/bin/env python3

import argparse
import csv
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Result:
    query: str
    real_score: float
    predicted_score: float


def read_data(path: str) -> Dict[str, List[Result]]:
    # Collate by query.
    by_query = {}
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            query, real_score, predicted_score = row
            by_query.setdefault(query, []).append(
                Result(
                    query=query,
                    real_score=float(real_score),
                    predicted_score=float(predicted_score),
                )
            )

    # Sort by predicted_score.
    for query in by_query.keys():
        by_query[query].sort(key=lambda r: r.predicted_score, reverse=True)
    return by_query


def calculate_dcg(results: List[Result]) -> float:
    dcg = 0
    for rank, result in enumerate(results, 1):
        dcg += result.real_score / math.log(rank + 1, 2)
    return dcg


def calculate_ncdg(data_by_query: Dict[str, List[Result]]) -> float:
    ndcgs = []
    for query, results in data_by_query.items():
        dcg = calculate_dcg(results)
        optimal = calculate_dcg(
            [r for r in sorted(results, key=lambda r: r.real_score, reverse=True)]
        )
        ndcgs.append(dcg / optimal)
    return statistics.mean(ndcgs)


# TODO Add more tests here with assertions.
# Note that in practice, we have much more tooling around this - this is just an example!!!
def test_dcg():
    assert calculate_dcg([Result('apple', 1, 1)]) == 1

def test_dcg_sample2():
    assert abs(round(calculate_dcg([Result('banana', 0, 0.6), Result('banana', 1, 0.4)]), 2) - 0.63) <= 1e-08

def test_ncdg_sample1():
    sample_dic = {'orange': [Result('orange', 0, 0.6), Result('orange', 0, 0.3), Result('orange', 1, 0.1)]}
    assert calculate_ncdg(sample_dic) == 0.5
    

def test_ncdg_dcg_diff():
    sample_dic = {'peach': [Result('orange', 0.7, 0.6), Result('orange', 0.2, 0.3), Result('orange', 0.1, 0.1)]}
    assert calculate_ncdg(sample_dic) == 1


def test_all():
    print("Running tests")
    # TODO Add your functions to this list!
    for func in [test_dcg, test_dcg_sample2, test_ncdg_sample1, test_ncdg_dcg_diff]:
        try:
            func()
        except AssertionError:
            status = "😫"
        else:
            status = "🎉"
        print(f"{status}\t{func.__name__}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs=1, help="Predictions in .TSV format")
    args = parser.parse_args()

    path = args.path[0]
    if not os.path.exists(path):
        parser.error(f"Invalid path {path}")

    data_by_query = read_data(path)
    mean_ndcg = calculate_ncdg(data_by_query)
    print(mean_ndcg)

    test_all()


if __name__ == "__main__":
    main()
