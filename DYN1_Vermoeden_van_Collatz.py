"""
@author: Sebastian Bigge
@date:   2026-11-07
"""

# Imports


####################################################################################
################################  IMPLEMENTATION  ##################################
####################################################################################

"""
"""
def collatz(n: int) -> dict[int,int]:
    # dynamic programming
    pass


"""
"""
def brute_force(n: int) -> int:
    steps = 0
    while n != 1:
        if (n % 2 == 0):
            n = n/2
        else:
            n = 3*n + 1
        steps += 1
    return steps

