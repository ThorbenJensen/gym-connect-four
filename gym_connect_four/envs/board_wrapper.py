
from gym_connect_four.envs.connect_four_minimal import ConnectFourEnv


c = ConnectFourEnv(board_shape=(4, 4))

for _ in range(3):
    c.step(0)
    c.step(1)
    c.step(2)
    c.step(3)

c.step(1)
c.step(2)


print(c.step(3))
print(c.is_win_state())
print(c.step(0))
print(c.is_win_state())
