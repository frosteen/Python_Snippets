def Distance_Formula(pos_center, pos, radius):
    if ((pos[0] - pos_center[0]) ** 2 + (pos[1] - pos_center[1]) ** 2) <= radius ** 2:
        return True
    return False
