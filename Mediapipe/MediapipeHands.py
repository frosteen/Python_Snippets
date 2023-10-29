import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic  # Hands model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_poly(image, results_landmarks, landmarks, depth):
    h, w = image.shape[:2]

    poly_coordinates = [
        (
            int(results_landmarks.landmark[landmark].x * w),
            int(results_landmarks.landmark[landmark].y * h),
        )
        for landmark in landmarks
    ]

    z_values = [results_landmarks.landmark[landmark].z for landmark in landmarks]

    mean_z_values = sum(z_values) / len(z_values)

    threshold = (
        int(abs(depth * mean_z_values)) if int(abs(depth * mean_z_values)) > 0 else 0
    )

    gray_threshold = 128  # - threshold * 2 if (128 - threshold * 2) >= 0 else 0

    cv2.fillPoly(
        image,
        [np.array(poly_coordinates)],
        (gray_threshold, gray_threshold, gray_threshold),
        lineType=cv2.LINE_AA,
    )


def draw_lines(image, results_landmarks, landmarks, depth):
    h, w = image.shape[:2]

    for index, landmark in enumerate(landmarks):
        if index == len(landmarks) - 1:
            break

        threshold = (
            int(abs(depth * results_landmarks.landmark[landmark + 1].z))
            if int(abs(depth * results_landmarks.landmark[landmark + 1].z)) > 0
            else 0
        )

        gray_threshold = 128  # - threshold if (128 - threshold) >= 0 else 0

        cv2.line(
            image,
            (
                int(results_landmarks.landmark[landmark].x * w),
                int(results_landmarks.landmark[landmark].y * h),
            ),
            (
                int(results_landmarks.landmark[landmark + 1].x * w),
                int(results_landmarks.landmark[landmark + 1].y * h),
            ),
            (gray_threshold, gray_threshold, gray_threshold),
            thickness=5 + int(threshold / 2),
            lineType=cv2.LINE_AA,
        )


def draw_styled_landmarks(image, results):
    if results is not None and results.right_hand_landmarks is not None:

        # landmark points (reference: https://google.github.io/mediapipe/images/mobile/hand_landmarks.png)
        thumb_finger_landmarks = [1, 2, 3, 4]
        index_finger_landmarks = [5, 6, 7, 8]
        middle_finger_landmarks = [9, 10, 11, 12]
        ring_finger_landmarks = [13, 14, 15, 16]
        pinky_finger_landmarks = [17, 18, 19, 20]
        palm_landmarks = [1, 2, 5, 9, 13, 17, 0]

        # draw polygon for the palm
        draw_poly(image, results.right_hand_landmarks, palm_landmarks, 400)

        # draw lines for each fingers
        draw_lines(image, results.right_hand_landmarks, thumb_finger_landmarks, 800)
        draw_lines(image, results.right_hand_landmarks, index_finger_landmarks, 400)
        draw_lines(image, results.right_hand_landmarks, middle_finger_landmarks, 400)
        draw_lines(image, results.right_hand_landmarks, ring_finger_landmarks, 400)
        draw_lines(image, results.right_hand_landmarks, pinky_finger_landmarks, 400)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        while cap.isOpened():

            # Capture frame-by-frame
            ret, frame = cap.read()

            image, results = mediapipe_detection(frame, holistic)

            black_image = np.zeros(image.shape, dtype=np.uint8)

            draw_styled_landmarks(black_image, results)

            black_image_with_hand = cv2.flip(black_image, 1)

            black_image_with_hand_frame = black_image_with_hand.copy()

            black_image_with_hand = cv2.resize(
                black_image_with_hand,
                (100, 100),  # image size
                interpolation=cv2.INTER_AREA,  # best interpolation for shrinking an image
            )

            # Display the resulting frame
            cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("frame", 1280, 720)
            cv2.imshow("frame", black_image_with_hand_frame)

            # ESC to quit
            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
