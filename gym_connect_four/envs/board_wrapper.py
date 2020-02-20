
from gym_connect_four.envs.connect_four_env import ConnectFourEnv


class ConnectFourWrapper(ConnectFourEnv):

    def __init__(self, board_shape):
        super(ConnectFourWrapper, self).__init__(board_shape)
        self._winner = 0

    def get_current_player(self) -> int:
        return self._ConnectFourEnv__current_player

    def _change_player(self) -> int:
        self._ConnectFourEnv__current_player *= -1
        return self._ConnectFourEnv__current_player

    def step(self, action: int):
        # make action
        out = super().step(action)
        # add info on who won to output
        if (out[2] is True) and (self._winner == 0):
            self._winner = self.get_current_player()
        out[3]["winner"] = self._winner
        # change player and return output
        self._change_player()
        return out


c = ConnectFourWrapper(board_shape=(4, 4))

c.step(2)
c.step(1)

c.board

c.get_current_player()

c.available_moves()

c.render()


c.is_win_state()
c.LOSS_REWARD

c.board
