from sympy import symbols, Eq, solve


def trilateration_3d(x1, y1, z1, r1, x2, y2, z2, r2, x3, y3, z3, r3):
    x, y, z = symbols("x y z")
    ans = solve(
        [
            Eq((x1 - x) ** 2 + (y1 - y) ** 2 + (z1 - z) ** 2, r1 ** 2),
            Eq((x2 - x) ** 2 + (y2 - y) ** 2 + (z2 - z) ** 2, r2 ** 2),
            Eq((x3 - x) ** 2 + (y3 - y) ** 2 + (z3 - z) ** 2, r3 ** 2),
        ],
        [x, y, z],
    )
    return ans[0]


def trilateration_2d(x1, y1, r1, x2, y2, r2, x3, y3, r3):
    A = 2 * x2 - 2 * x1
    B = 2 * y2 - 2 * y1
    C = r1 ** 2 - r2 ** 2 - x1 ** 2 + x2 ** 2 - y1 ** 2 + y2 ** 2
    D = 2 * x3 - 2 * x2
    E = 2 * y3 - 2 * y2
    F = r2 ** 2 - r3 ** 2 - x2 ** 2 + x3 ** 2 - y2 ** 2 + y3 ** 2
    x = (C * E - F * B) / (E * A - B * D)
    y = (C * D - A * F) / (B * D - A * E)
    return x, y


def distance_formula(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def rssi_to_distance(rssi, A, n):
    distance = 10 ** ((A - rssi) / (10 * n))
    return distance
