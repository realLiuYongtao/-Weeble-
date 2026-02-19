"""过滤关键点 计算辅助参数"""
"""X,Y,visibility"""


def filterKP(landmarks,w,h,minVisibility=0):

    pic__landmark = []

    """0"""
    # 头部关键点索引
    HEAD_KeyPoint = (0,1,2,3,4,5,6,7,8,9,10)
    visible_head_landmarks = []
    for P in HEAD_KeyPoint:
        if landmarks[P].visibility>minVisibility:
            visible_head_landmarks.append(landmarks[P])

    # 如果无可见关键点,则跳过计算
    if len(visible_head_landmarks) != 0:
        # 计算可见关键点的平均坐标
        head_x = sum(landmark.x for landmark in visible_head_landmarks) / len(visible_head_landmarks)
        head_y = sum(landmark.y for landmark in visible_head_landmarks) / len(visible_head_landmarks)
        # 计算可见程度平均值
        head_visibility = sum(landmark.visibility for landmark in visible_head_landmarks) / len(visible_head_landmarks)

        pic__landmark.append([head_x*w, head_y*h, head_visibility])
    else:
        pic__landmark.append([0, 0, 0])

    """1 2 3 4"""
    BODY_KeyPoint = [11,12,13,14]
    for P in BODY_KeyPoint:
        pic__landmark.append([landmarks[P].x*w,landmarks[P].y*h,landmarks[P].visibility])


    """5 6"""
    # 左手关键点索引
    RIGHT_HAND_KeyPoint = (15, 17, 19, 21)

    visible_right_hand_landmarks = []
    for P in RIGHT_HAND_KeyPoint:
        if landmarks[P].visibility > minVisibility:
            visible_right_hand_landmarks.append(landmarks[P])

    if len(visible_right_hand_landmarks) != 0:
        right_hand_x = sum(landmark.x for landmark in visible_right_hand_landmarks) / len(visible_right_hand_landmarks)
        right_hand_y = sum(landmark.y for landmark in visible_right_hand_landmarks) / len(visible_right_hand_landmarks)
        right_hand_visibility = sum(landmark.visibility for landmark in visible_right_hand_landmarks) / len(
            visible_right_hand_landmarks)

        pic__landmark.append([right_hand_x*w, right_hand_y*h, right_hand_visibility])
    else:
        pic__landmark.append([0, 0, 0])

    # 右手关键点索引
    LEFT_HAND_KeyPoint = (16, 18, 20, 22)

    visible_left_hand_landmarks = []
    for P in LEFT_HAND_KeyPoint:
        if landmarks[P].visibility > minVisibility:
            visible_left_hand_landmarks.append(landmarks[P])

    if len(visible_left_hand_landmarks) != 0:
        left_hand_x = sum(landmark.x for landmark in visible_left_hand_landmarks) / len(visible_left_hand_landmarks)
        left_hand_y = sum(landmark.y for landmark in visible_left_hand_landmarks) / len(visible_left_hand_landmarks)
        left_hand_visibility = sum(landmark.visibility for landmark in visible_left_hand_landmarks) / len(
            visible_left_hand_landmarks)

        pic__landmark.append([left_hand_x*w, left_hand_y*h, left_hand_visibility])
    else:
        pic__landmark.append([0, 0, 0])


    """7 8 9 10"""
    BODY_KeyPoint = [23,24,25,26]
    for P in BODY_KeyPoint:
        pic__landmark.append([landmarks[P].x*w,landmarks[P].y*h,landmarks[P].visibility])


    """11 12"""
    # 左脚关键点索引
    LEFT_FOOT_KeyPoint = (27, 29, 31)

    visible_left_foot_landmarks = []
    for P in LEFT_FOOT_KeyPoint:
        if landmarks[P].visibility>minVisibility:
            visible_left_foot_landmarks.append(landmarks[P])

    if len(visible_left_foot_landmarks) != 0:
        left_foot_x = sum(landmark.x for landmark in visible_left_foot_landmarks) / len(visible_left_foot_landmarks)
        left_foot_y = sum(landmark.y for landmark in visible_left_foot_landmarks) / len(visible_left_foot_landmarks)
        left_foot_visibility = sum(landmark.visibility for landmark in visible_left_foot_landmarks) / len(
            visible_left_foot_landmarks)

        pic__landmark.append([left_foot_x*w, left_foot_y*h, left_foot_visibility])
    else:
        pic__landmark.append([0, 0, 0])


    # 右脚关键点索引
    RIGHT_FOOT_KeyPoint = (28, 30, 32)

    visible_right_foot_landmarks = []
    for P in RIGHT_FOOT_KeyPoint:
        if landmarks[P].visibility>minVisibility:
            visible_right_foot_landmarks.append(landmarks[P])

    if len(visible_right_foot_landmarks) != 0:
        right_foot_x = sum(landmark.x for landmark in visible_right_foot_landmarks) / len(visible_right_foot_landmarks)
        right_foot_y = sum(landmark.y for landmark in visible_right_foot_landmarks) / len(visible_right_foot_landmarks)
        right_foot_visibility = sum(landmark.visibility for landmark in visible_right_foot_landmarks) / len(
            visible_right_foot_landmarks)

        pic__landmark.append([right_foot_x*w, right_foot_y*h, right_foot_visibility])
    else:
        pic__landmark.append([0, 0, 0])



    """distance"""

    """angle"""



    return pic__landmark


# ###表征关键点之间的关系（如角度）作为额外输入
#  ###过滤关键点类型，只选择部分与判断摔倒相关的关键关键点
#  ###关键点数据处理
#  ###平滑关键点坐标

