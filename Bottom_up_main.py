def bottom_up_list(max_n: int) -> tuple[list[int], int]:
    counter = 0; steps = [0] * (max_n + 1)
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