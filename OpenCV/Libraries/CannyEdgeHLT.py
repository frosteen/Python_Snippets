import cv2


def apply_cannyedge_hlt(
    img, thickness_line, thickness_rect, color_line, color_rect, is_line, is_rect
):
    # GET IMAGE SIZE
    H, W = img.shape[:2]

    # CANNY EDGE
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)  # ADD GAUSSIAN BLUR
    img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)  # ADD GAUSSIAN BLUR
    img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)  # ADD GAUSSIAN BLUR
    img_blur = cv2.GaussianBlur(img_blur, (5, 5), 0)  # ADD GAUSSIAN BLUR
    img_canny = cv2.Canny(img_blur, 50, 0)  # APPLY CANNY
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)  # CONVERT IMAGE TO BGR

    # HOU LINE TRANSFORM

    # FEATURE 1: VALUE LEFT
    # LINE
    POINTS = [
        (int(W * 0.15), int(0)),
        (int(W * 0.15), int(H)),
        (int(W * 0.01), int(0)),
        (int(W * 0.01), int(H)),
        (int(0), int(H * 0.05)),
        (int(W), int(H * 0.05)),
        (int(0), int(H * 0.8)),
        (int(W), int(H * 0.8)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 2: THREAD
    # LINE
    POINTS = [
        (int(W * 0.58), int(0)),
        (int(W * 0.58), int(H)),
        (int(W * 0.66), int(0)),
        (int(W * 0.66), int(H)),
        (int(0), int(H * 0.025)),
        (int(W), int(H * 0.025)),
        (int(0), int(H * 0.95)),
        (int(W), int(H * 0.95)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 3: BSP
    # LINE
    POINTS = [
        (int(W * 0.67), int(0)),
        (int(W * 0.67), int(H)),
        (int(W * 0.8), int(0)),
        (int(W * 0.8), int(H)),
        (int(0), int(H * 0.2)),
        (int(W), int(H * 0.2)),
        (int(0), int(H * 0.45)),
        (int(W), int(H * 0.45)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 4: VALUE RIGHT
    # LINE
    POINTS = [
        (int(W * 0.81), int(0)),
        (int(W * 0.81), int(H)),
        (int(W * 0.96), int(0)),
        (int(W * 0.96), int(H)),
        (int(0), int(H * 0.82)),
        (int(W), int(H * 0.82)),
        (int(0), int(H * 0.98)),
        (int(W), int(H * 0.98)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 5: GOLD FIBER
    # LINE
    POINTS = [
        (int(W * 0.14), int(0)),
        (int(W * 0.14), int(H)),
        (int(W * 0.26), int(0)),
        (int(W * 0.26), int(H)),
        (int(0), int(H * 0.44)),
        (int(W), int(H * 0.44)),
        (int(0), int(H * 0.75)),
        (int(W), int(H * 0.75)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 6: SERIAL NUMBER TOP
    # LINE
    POINTS = [
        (int(W * 0.78), int(0)),
        (int(W * 0.78), int(H)),
        (int(W * 0.95), int(0)),
        (int(W * 0.95), int(H)),
        (int(0), int(H * 0.042)),
        (int(W), int(H * 0.042)),
        (int(0), int(H * 0.22)),
        (int(W), int(H * 0.22)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 7: REPUBLIKA NG PILIPINAS
    # LINE
    POINTS = [
        (int(W * 0.4), int(0)),
        (int(W * 0.4), int(H)),
        (int(W * 0.77), int(0)),
        (int(W * 0.77), int(H)),
        (int(0), int(H * 0.04)),
        (int(W), int(H * 0.04)),
        (int(0), int(H * 0.15)),
        (int(W), int(H * 0.15)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 8: FLAG BADGE
    # LINE
    POINTS = [
        (int(W * 0.5), int(0)),
        (int(W * 0.5), int(H)),
        (int(W * 0.6), int(0)),
        (int(W * 0.6), int(H)),
        (int(0), int(H * 0.18)),
        (int(W), int(H * 0.18)),
        (int(0), int(H * 0.47)),
        (int(W), int(H * 0.47)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 9: VALUE IN WORDS
    # LINE
    POINTS = [
        (int(W * 0.45), int(0)),
        (int(W * 0.45), int(H)),
        (int(W * 0.79), int(0)),
        (int(W * 0.79), int(H)),
        (int(0), int(H * 0.88)),
        (int(W), int(H * 0.88)),
        (int(0), int(H * 0.99)),
        (int(W), int(H * 0.99)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 10: TOP LEFT LOGO
    # LINE
    POINTS = [
        (int(W * 0.13), int(0)),
        (int(W * 0.13), int(H)),
        (int(W * 0.25), int(0)),
        (int(W * 0.25), int(H)),
        (int(0), int(H * 0.005)),
        (int(W), int(H * 0.005)),
        (int(0), int(H * 0.35)),
        (int(W), int(H * 0.35)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 11: SERIAL NUMBER BOTTOM
    # LINE
    POINTS = [
        (int(W * 0.025), int(0)),
        (int(W * 0.025), int(H)),
        (int(W * 0.22), int(0)),
        (int(W * 0.22), int(H)),
        (int(0), int(H * 0.74)),
        (int(W), int(H * 0.74)),
        (int(0), int(H * 0.92)),
        (int(W), int(H * 0.92)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    # FEATURE 12: PRESIDENT'S SIGNATURE
    # LINE
    POINTS = [
        (int(W * 0.49), int(0)),
        (int(W * 0.49), int(H)),
        (int(W * 0.61), int(0)),
        (int(W * 0.61), int(H)),
        (int(0), int(H * 0.48)),
        (int(W), int(H * 0.48)),
        (int(0), int(H * 0.6)),
        (int(W), int(H * 0.6)),
    ]
    draw_line(img_gray, color_line, thickness_line, POINTS, is_line)
    # RECTANGLE
    PT1_R = (POINTS[0][0], POINTS[4][1])
    PT2_R = (POINTS[2][0], POINTS[6][1])
    draw_rect(img_gray, color_rect, thickness_rect, PT1_R, PT2_R, is_rect)

    return img_canny, img_gray


# HELPFUL FUNCTIONS


def draw_line(img, color, thickness, POINTS, is_draw):
    if is_draw:
        cv2.line(img, POINTS[0], POINTS[1], color, thickness)
        cv2.line(img, POINTS[2], POINTS[3], color, thickness)
        cv2.line(img, POINTS[4], POINTS[5], color, thickness)
        cv2.line(img, POINTS[6], POINTS[7], color, thickness)


def draw_rect(img, color, thickness, PT1_R, PT2_R, is_draw):
    if is_draw:
        cv2.rectangle(img, PT1_R, PT2_R, color, thickness)


if __name__ == "__main__":
    # CONFIGURATIONS
    thickness_line, thickness_rect = 2, 4
    color_line, color_rect = (0, 255, 0), (0, 0, 255)
    is_line, is_rect = True, True

    # READ IMAGE
    img = cv2.imread("Dataset/UV/AUTHENTIC/Captured-09.08.2023 02.28.39.jpg")
    H, W = img.shape[:2]

    # PROCESSING
    img_canny, img = apply_cannyedge_hlt(
        img, thickness_line, thickness_rect, color_line, color_rect, True, True
    )

    # SHOW IMAGE
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Result", int(W / 2), int(H / 2))
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
