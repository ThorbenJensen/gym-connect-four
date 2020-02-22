""" Pytest-unittests for testing class ConnectFourEnv """

import numpy as np

from connect_four_bot.envs import ConnectFourEnv

BOARD_VALIDATION = np.array([[0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 0],
                    [0, 0, -1, 1, 0, -1, 0],
                    [0, 0, 1, 1, 0, 1, 1],
                    [-1, -1, -1, -1, 0, -1, -1],
                    [1, 1, -1, 1, -1, 1, 1]])

BOARD_WIN_ROW = np.array([[0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, -1, 0, 0, 0],
                 [0, 0, -1, 1, 0, -1, 0],
                 [0, 0, 1, 1, 0, 1, 1],
                 [-1, -1, -1, -1, 0, -1, -1],
                 [1, 1, -1, 1, -1, 1, 1]])

BOARD_WIN_COLUMN = np.array([[0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, -1, 1, 0, -1, 0],
                    [0, 0, 1, 1, 0, 1, 1],
                    [-1, 0, -1, -1, 0, -1, -1],
                    [1, 1, -1, 1, -1, 1, 1]])

BOARD_WIN_DIAGONAL = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, -1, 1, 0, -1, 0],
                      [0, 0, 1, 1, 1, 1, 1],
                      [-1, 1, -1, -1, 0, 1, -1],
                      [1, 1, -1, 1, -1, 1, 1]])

BOARD_WIN_BDIAGONAL = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0],
                       [0, 0, -1, 1, 0, -1, 0],
                       [0, 0, 1, 1, 0, 1, 1],
                       [-1, 1, -1, -1, 0, -1, -1],
                       [1, 1, -1, 1, -1, 1, 1]])

BOARD_AVAILABLE_0123 = np.array([[0, 0, 0, 0, -1, 1, -1],
                        [0, 0, 0, 1, 1, -1, 1],
                        [0, 0, -1, 1, 1, -1, -1],
                        [0, 0, 1, 1, 1, 1, 1],
                        [-1, 1, -1, -1, -1, -1, -1],
                        [1, 1, -1, 1, -1, 1, 1]])

BOARD_AVAILABLE_2 = np.array([[1, 1, 0, -1, -1, 1, -1],
                     [1, 1, -1, 1, 1, -1, 1],
                     [1, 1, -1, 1, 1, -1, -1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [-1, 1, -1, -1, -1, -1, -1],
                     [1, 1, -1, 1, -1, 1, 1]])

BOARD_AVAILABLE_6 = np.array([[1, 1, 1, 1, -1, 1, 0],
                     [1, 1, -1, 1, 1, -1, 1],
                     [1, 1, -1, 1, 1, -1, -1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [-1, 1, -1, -1, -1, -1, -1],
                     [1, 1, -1, 1, -1, 1, 1]])

BOARD_AVAILABLE_NONE = np.array([[1, 1, 1, 1, -1, 1, 1],
                        [1, 1, -1, 1, 1, -1, 1],
                        [1, 1, -1, 1, 1, -1, -1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [-1, 1, -1, -1, -1, -1, -1],
                        [1, 1, -1, 1, -1, 1, 1]])


def test_is_valid_action():
    env = ConnectFourEnv()
    env.reset(BOARD_VALIDATION)
    assert env.is_valid_action(0)
    assert not env.is_valid_action(3)


def test_is_win_state():
    env = ConnectFourEnv()
    env.reset(BOARD_WIN_ROW)
    assert env.is_win_state()

    env.reset(BOARD_WIN_COLUMN)
    assert env.is_win_state()

    env.reset(BOARD_WIN_DIAGONAL)
    assert env.is_win_state()

    env.reset(BOARD_WIN_BDIAGONAL)
    assert env.is_win_state()


def test_available_moves():
    env = ConnectFourEnv()
    env.reset(BOARD_AVAILABLE_0123)
    assert set(env.available_moves()) == {0, 1, 2, 3}

    env.reset(BOARD_AVAILABLE_2)
    assert set(env.available_moves()) == {2}

    env.reset(BOARD_AVAILABLE_6)
    assert set(env.available_moves()) == {6}

    env.reset(BOARD_AVAILABLE_NONE)
    assert set(env.available_moves()) == set([])
