import cv2


def probability_visuals(result, outputs, input_frame, spacing=25):
    output_frame = input_frame.copy()

    for num, prob in enumerate(result):
        # visualize probability distribution
        cv2.rectangle(
            output_frame,
            (0, 70 + num * spacing),
            (int(prob * 100), 90 + num * spacing),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            output_frame,
            outputs[num] + " " + str(round(float(prob * 100), 2)) + "%",
            (0, 85 + num * spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output_frame
