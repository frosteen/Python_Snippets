def check_location(image, center):
    h, w, _ = image.shape
    positions = []
    if center[1] > int(h * 0.75):
        positions.append("bottom")
    if center[1] < int(h * 0.25):
        positions.append("top")
    if center[0] > int(w * 0.75):
        positions.append("right")
    if center[0] < int(w * 0.25):
        positions.append("left")
    if (
        center[1] < int(h * 0.75)
        and center[1] > int(h * 0.25)
        and center[0] < int(w * 0.75)
        and center[0] > int(w * 0.25)
    ):
        positions.append("center")
    return positions
