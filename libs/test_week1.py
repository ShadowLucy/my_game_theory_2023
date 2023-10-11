#import libs.week1 as week1
import numpy as np
#import pytest


def test_week1():
    matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    row_strategy = np.array([[0.1, 0.2, 0.7]])
    column_strategy = np.array([[0.3, 0.2, 0.5]]).transpose()

    row_value = week1.evaluate(matrix=matrix, row_strategy=row_strategy, column_strategy=column_strategy)
    assert row_value == pytest.approx(0.08)

    br_value_row = week1.best_response_value_row(matrix=matrix, row_strategy=row_strategy)
    br_value_column = week1.best_response_value_column(matrix=matrix, column_strategy=column_strategy)
    assert br_value_row == pytest.approx(-0.6)
    assert br_value_column == pytest.approx(-0.2)

def mixed_strategy_values(matrix1, matrix2, strategy1, strategy2):
    strategy1 = strategy1.transpose()

    # value for player 1
    value1 = strategy2 @ matrix1 @ strategy1

    # value player 2
    value2 = strategy2 @ matrix2 @ strategy1

    return (value1[0][0], value2[0][0])

def zero_sum_mixed_strategy_values(matrix1, strategy1, strategy2):
    neg_matrix1 = -1 * matrix1
    matrix2 = neg_matrix1.transpose()
    return mixed_strategy_values(matrix1, matrix2, strategy1, strategy2)

def best_response_calculation(matrix1, strategy2):
    strategy2 = strategy2.transpose()
    utilities = matrix1 @ strategy2
    br = np.argmax(utilities)
    br_strategy = np.zeros(3)
    br_strategy[br] = 1
    return np.array([br_strategy])

def best_response_value_calculation(matrix1, strategy2):
    br_strategy1 = best_response_calculation(matrix1, strategy2)
    ut_values = zero_sum_mixed_strategy_values(matrix1, br_strategy1, strategy2)
    return ut_values[0]

# I supposed "matrix" was utility matrix of column player
# as it was not more specified, it might be the other way around and some results are inverted

col_matrix = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
row_matrix = -1 * (col_matrix.transpose())

row_strategy = np.array([[0.1, 0.2, 0.7]])
col_strategy = np.array([[0.3, 0.2, 0.5]])

# 2)
#print(mixed_strategy_values(row_matrix, col_matrix, row_strategy, col_strategy))
#print(zero_sum_mixed_strategy_values(row_matrix, row_strategy, col_strategy))

# 3)
#print(best_response_calculation(col_matrix, row_strategy))
#print(best_response_calculation(row_matrix, col_strategy))

# 4)
#print(best_response_value_calculation(row_matrix, col_strategy))
#print(best_response_value_calculation(col_matrix, row_strategy))

# 5)
