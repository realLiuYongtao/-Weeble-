"""绘制关键点"""

import numpy as np
import cv2

# 点的连线
POSE_CONNECTIONS = [(6, 4), (4, 2), (2, 13), (13, 1), (5, 3), (3, 1), (12, 10),
                    (10, 8), (8, 2), (11, 9), (9, 7), (7, 1), (13, 0),(7, 8)]
# 点的颜色
POINT_COLORS = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                (0, 255, 255)]  # LHip, RHip, LKnee, Renee, LAnkle, RAnkle, Neck
# 线的颜色
LINE_COLORS = [(64,0,75), (139,85,55), (0, 134, 255), (86,170,94), (9,86,36),
               (139,85,55), (77, 135, 255), (244,241,244), (86,170,95), (9,86,36),
               (255, 156, 127), (0, 127, 255), (64,0,75), (0, 77, 255), (255, 77, 36),(244,241,244)]


def drawPoint(frame, pts,thinckness=2):

    part_line = {}
    pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)
    for n in range(pts.shape[0]):
        if pts[n, 2] <= 0.05:
            continue
        cor_x, cor_y = int(pts[n, 0]), int(pts[n, 1])
        part_line[n] = (cor_x, cor_y)
        cv2.circle(frame, (cor_x, cor_y), 3, POINT_COLORS[n], -1)
        # cv2.putText(frame, str(n), (cor_x+10, cor_y+10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    for i, (start_p, end_p) in enumerate(POSE_CONNECTIONS):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(frame, start_xy, end_xy, LINE_COLORS[i], thinckness) # 自适应线宽：int(1 * (pts[start_p, 2] + pts[end_p, 2]) + 3)
    return frame










