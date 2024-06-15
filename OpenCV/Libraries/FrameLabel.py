import cv2


def frame_label(
    frame, labels, font_scale=0.5, thickness=2, spacing=25, color=(0, 255, 0)
):
    for index, text in enumerate(labels):
        # black outline
        cv2.putText(
            frame,
            text,
            (10, spacing + spacing * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
        )

        # normal text color
        cv2.putText(
            frame,
            text,
            (10, spacing + spacing * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )
