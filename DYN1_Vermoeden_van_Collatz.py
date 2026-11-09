"""
@author: Sebastian Bigge
@date:   2026-11-09
"""

# Imports
import time
import matplotlib.pyplot as plt


####################################################################################
################################  IMPLEMENTATION  ##################################
####################################################################################

"""
"""
def brute_force(n: int) -> int:
    if n <= 1:
        return 0
    n = n // 2 if n % 2 == 0 else 3 * n + 1
    return 1 + brute_force(n)


"""
"""
def brute_force_list(max_n: int) -> list[int]:
    steps = [0] * (max_n + 1)
    for n in range(1, max_n + 1):
        steps[n] = brute_force(n)
    return (steps, steps)


"""
"""
def top_down(n: int, steps: dict, counter: list[int]) -> int:
    if n in steps:
        return steps[n]
    counter[0] += 1
    n = n // 2 if n % 2 == 0 else 3 * n + 1
    steps[n] = 1 + top_down(n, steps, counter)
    return steps[n]


"""
"""
def top_down_list(max_n: int) -> list[int]:
    counter = [0]
    steps = {1: 0}
    result = [0] * (max_n + 1)
    for i in range(2, max_n + 1):
        result[i] = top_down(i, steps, counter)
    return (result, counter[0])


"""
"""
def bottom_up_list(max_n: int) -> list[int]:
    counter = 0
    steps = [0] * (max_n + 1)
    for i in range(2, max_n + 1):
        if i % 2 == 0:
            steps[i] = steps[i // 2] + 1
            counter += 1
        else:
            j, c = i, 0
            while j >= i:
                j = j // 2 if j % 2 == 0 else 3 * j + 1
                c += 1
            steps[i] = c + steps[j]
            counter += c
    return steps, counter


####################################################################################
####################################  PLOTS  #######################################
####################################################################################



####################################################################################
####################################  TESTS  #######################################
####################################################################################
