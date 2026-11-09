def top_down(max_n: int) -> tuple[list[int], int]:
    counter = [0]; steps = {1: 0}
    result = [0] * (max_n + 1)
    for i in range(2, max_n + 1):
        result[i] = top_down_aux(i, steps, counter)
    return result, counter[0]
def top_down_aux(n: int, steps: dict, counter: list[int]) -> int:
    if n in steps:
        return steps[n]
    counter[0] += 1
    next_n = n // 2 if n % 2 == 0 else 3 * n + 1
    steps[n] = 1 + top_down_aux(next_n, steps, counter)
    return steps[n]