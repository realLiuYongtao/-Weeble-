import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

"""
ObjectDetectorResult:
 Detection #0:
  Box: (x: 355, y: 133, w: 190, h: 206)
  Categories:
   index       : 17
   score       : 0.73828
   class name  : dog
 Detection #1:
  Box: (x: 103, y: 15, w: 138, h: 369)
  Categories:
   index       : 17
   score       : 0.73047
   class name  : dog

"""


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                         MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return image


def visualizePerson(image, detection_result) -> np.ndarray:
    """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """

    for detection in detection_result.detections:
        if detection.categories[0].category_name == 'person':
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 2)

    return image


def get_detection_target(detection_result):
    target = []
    for detection in detection_result.detections:
        if detection.categories[0].category_name == 'person':
            # 目标位置
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            # 置信度
            probability = round(detection.categories[0].score, 2)

            target.append([probability, start_point, end_point])

    return target


"""
DetectionResult(detections=
                [
                  Detection(bounding_box=BoundingBox(origin_x=668, origin_y=179, width=126, height=242), categories=[Category(index=None, score=0.5234375, display_name=None, category_name='person')], keypoints=[]), 
                  Detection(bounding_box=BoundingBox(origin_x=1085, origin_y=528, width=216, height=513), categories=[Category(index=None, score=0.4765625, display_name=None, category_name='person')], keypoints=[]), 
                  Detection(bounding_box=BoundingBox(origin_x=963, origin_y=13, width=474, height=424), categories=[Category(index=None, score=0.4296875, display_name=None, category_name='boat')], keypoints=[]), 
                  Detection(bounding_box=BoundingBox(origin_x=652, origin_y=299, width=90, height=138), categories=[Category(index=None, score=0.26171875, display_name=None, category_name='bicycle')], keypoints=[]), 
                  Detection(bounding_box=BoundingBox(origin_x=680, origin_y=355, width=112, height=104), categories=[Category(index=None, score=0.19921875, display_name=None, category_name='bicycle')], keypoints=[])
                ]
                )
"""


def get_image(image, start_point, end_point):
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask, start_point, end_point, (255, 255, 255), -1)
    img_masked = cv2.bitwise_and(image, image, mask=mask)

    return img_masked


# 谷歌的 绘制关键点
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
