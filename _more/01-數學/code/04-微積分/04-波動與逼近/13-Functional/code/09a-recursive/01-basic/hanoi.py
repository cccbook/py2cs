def hanoi(n, curr, dest, rest):
    if n == 0: return
    hanoi(n - 1, curr, rest, dest)
    print(f"move {n} from {curr} to {dest}")
    hanoi(n - 1, rest, dest, curr)
    return

# hanoi(3, "A", "B", "C")
hanoi(64, "A", "B", "C")
