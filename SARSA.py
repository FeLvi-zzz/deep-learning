import random as rd
import pandas as pd

Actions = ["↑", "↓", "←", "→"]
States  = ["s1", "s2", "s3", "s4"]

# Q-Table -- Q[state][action]
Q = [[0] * len(Actions) for i in range(len(States))]

# Revenue -- R[state]
R = [-1, -1, -1, 10]

# Transition -- T[state][action]
T = (
  (1, 0, 0, 0), # s1 - (↑, ↓, ←, →)
  (1, 0, 1, 2), # s2 - (↑, ↓, ←, →)
  (2, 3, 1, 2), # s3 - (↑, ↓, ←, →)
  (2, 3, 3, 3)  # s4 - (↑, ↓, ←, →)
)

#####################################
#                                   #
#             Parameter             #
#                                   #
#####################################
alpha   = 0.3   # learning rate
gamma   = 0.9   # discount factor
epsilon = 0.3   # ε-greedy
step    = 10**6 # steps
state   = 0     # initial state


def argmax(l):
  return l.index(max(l))

for _ in range(step):
  # ε-greedy
  action = argmax(Q[state]) if rd.random() > epsilon else rd.randrange(len(Q[state]))

  # update Q-Table
  Q[state][action] = alpha * Q[state][action] + (1 - alpha) * (R[state] + gamma * max(Q[T[state][action]]))
  
  # change state
  state = T[state][action]

# display result
print("Optimal Direction")
print(*[(s + " - " + Actions[argmax(q)]) for s, q in zip(States, Q)], sep=", ")

print("\nQ-Table")
print(pd.DataFrame(
  Q,
  columns = Actions,
  index   = States
))

# result example
#
# Optimal Direction
# s1 - ↑, s2 - →, s3 - ↓, s4 - ↓
# 
# Q-Table
#         ↑        ↓        ←        →
# s1  70.19   62.171   62.171   62.171
# s2  70.19   62.171   70.190   79.100
# s3  79.10   89.000   70.190   79.100
# s4  90.10  100.000  100.000  100.000
