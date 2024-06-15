import cv2


def Draw_Region(frame, pos, radius, transparency=0.2):
    overlay = frame.copy()
    cv2.circle(overlay, pos, radius, (0, 255, 0), -1)
    alpha = transparency  # transparency factor
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
