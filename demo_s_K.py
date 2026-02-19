import cv2
from collections import deque
import numpy as np
import datetime
import os
import torch

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from SGTCN.stgcn.stgcn import STGCN

from myFunction.FilteringKeyPoints2 import filterKP as KP2
from myFunction.draw_chinese_text import add_chinese_text
from myFunction.drawP import drawPoint
from myFunction.visualize import get_detection_target, get_image

from gpt_new import Chat_function as Chat_function_baidu


def single_k(if_falldown=None, MODEL='live', dc=False):

    if if_falldown is None:
        if_falldown = [False, 'video-output/FallDown/fall-down.mp4']

    global fall_down_video, fall_down_video_path
    out_video_path = './video-output'
    in_video_path = 'video-test/0.mp4'
    fall_down_path = './video-output/FallDown'
    video_fps = 24
    ACTION_MODEL_MAX_FRAMES = 30  # 单位：帧
    Min_Fall_Confidence = 0  # 0.2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PointThinness = 2
    BoxThinness = 2
    TestThinness = 30
    Max_Person_Num = 3

    box_l = 0  # 检测框
    box_t = 0
    box_r = 0
    box_b = 0
    d = [-100, -100, 100, 100]
    num_fall_down = 1

    if if_falldown is None:
        if_falldown = [False, 'video-output/FallDown/fall-down.mp4']

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    if MODEL == 'live':
        video_cap = cv2.VideoCapture(0)
    else:
        video_cap = cv2.VideoCapture(in_video_path)

    # 图像大小
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    '''检测器'''
    # 框
    # BaseOptions = mp.tasks.BaseOptions
    # ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='ModelPath/efficientdet_lite0.tflite')
    options = ObjectDetectorOptions(
        base_options=base_options,
        max_results=1,
        running_mode=VisionRunningMode.IMAGE)
    # 框检测
    detector = vision.ObjectDetector.create_from_options(options)

    # 点
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='ModelPath/pose_landmarker_full.task'),
        min_pose_presence_confidence=0,
        num_poses=1,
        running_mode=VisionRunningMode.IMAGE)
    # 点检测
    pose_tracker = PoseLandmarker.create_from_options(options)

    # 动作检测
    action_model = STGCN(weight_file='SGTCN/weights/ts-stg-model-1.pth', device=device)

    # 获取当前时间
    makefile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(out_video_path):
        os.mkdir(out_video_path)
    out_video_path = out_video_path + '/' + makefile_time + '.mp4'

    # 创建一个VideoWriter对象,用于后续将图像框架写入视频文件。
    out_video = cv2.VideoWriter(out_video_path,  # 输出视频的路径及文件名
                                cv2.VideoWriter_fourcc(*'mp4v'),  # 指定视频编码格式为mp4
                                video_fps, (video_width, video_height))  # 设置视频帧率 分辨率

    # 记录一组 pose_landmarks 传入时序网络
    video_pose_landmark = deque(maxlen=ACTION_MODEL_MAX_FRAMES)

    # 记录一组图像用于保存跌倒动作视频
    FD = deque(maxlen=16 * 4)
    key = 0  # 用来表示out_video是否已经被创建，默认为0

    print('按键盘的q或者esc退出')
    while video_cap.isOpened():
        # 获取视频的下一帧
        success, input_frame = video_cap.read()
        if not success:
            break

        middle_frame = input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        # 框
        numpy_image = np.array(input_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        detection_result = detector.detect(mp_image)
        detection_target = get_detection_target(detection_result)
        if len(detection_target) > 0:  # 只获取最大置信度的结果
            box_l, box_t = detection_target[0][1]
            box_r, box_b = detection_target[0][2]

            box_l, box_r = (max(int(box_l) + d[0], 0),
                            min(int(box_r) + d[2], video_width))
            box_t, box_b = (max(int(box_t) + d[1], 0),
                            min(int(box_b) + d[3], video_height))

            middle_frame = get_image(input_frame, (box_l, box_t), (box_r, box_b))

        # 点
        numpy_image = np.array(middle_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        PoseLandmarkerResult = pose_tracker.detect(image=mp_image)
        pose_landmarks = PoseLandmarkerResult.pose_landmarks

        action_name = ''

        output_frame = input_frame.copy()
        if len(pose_landmarks) != 0:
            # 关键点：
            pic_pose_landmark = np.array(KP2(pose_landmarks[0], video_width, video_height))  # 关键点转化
            video_pose_landmark.append(pic_pose_landmark)

            # 绘制关键点
            output_frame = drawPoint(frame=output_frame, pts=pic_pose_landmark, thinckness=PointThinness)  # 自定义绘制工具

            '''mian 时序网络'''
            if len(video_pose_landmark) == ACTION_MODEL_MAX_FRAMES:
                pts = np.array(video_pose_landmark, dtype=np.float32)
                out = action_model.predict(pts, (video_height, video_width))
                action_name = action_model.class_names[out[0].argmax()]
                confid = out[0].max() * 100

                # 绘制状态
                if action_name == 'Fall Down':
                    clr = (0, 0, 255)
                else:
                    action_name = 'fine'
                    clr = (0, 0, 0)

                output_frame = cv2.rectangle(output_frame, (box_l, box_t),
                                             (box_r, box_b), clr, BoxThinness)
                output_frame = add_chinese_text(output_frame,
                                                f'idx: {0}\ndevice: {device}\nstate: {action_name}\nconfidence: {confid}',
                                                (box_l, box_t), clr, TestThinness)

        '''显示结果并保存视频'''
        output_frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)
        cv2.imshow('video', output_frame)
        # print('                                                        ', action_name)

        # ---------------------------------------------------------------------------------
        if action_name == 'Fall Down':
            if key == 0:  # 跌倒过程开始
                # 创建一个VideoWriter对象,用于后续将图像框架写入视频文件。

                if not os.path.exists(fall_down_path):
                    os.mkdir(fall_down_path)
                makefile_time = datetime.datetime.strftime(datetime.datetime.now(),
                                                           '%Y-%m-%d-%H-%M-%S')
                fall_down_video_path = (fall_down_path + '/' + 'fall-down' +
                                        makefile_time + '.mp4')
                num_fall_down += 1
                fall_down_video = cv2.VideoWriter(fall_down_video_path,
                                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                                  16, (video_width, video_height))  # 视频帧率 分辨率

            # 跌倒过程
            key += 1
            # print(key)
            FD.append(output_frame)

        elif action_name == 'fine' and key == 0:  # 未跌倒
            FD.append(output_frame)

        elif action_name == 'fine' and key != 0:  # 跌倒过程结束

            for fd in FD:
                fall_down_video.write(fd)
            fall_down_video.release()
            print("迭倒视频保存在：", fall_down_video_path)

            if_falldown[0] = True
            if_falldown[1] = fall_down_video_path

            key = 0
        # ---------------------------------------------------------------------------------

        out_video.write(output_frame)
        # 按键盘的q或者esc退出
        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    '''关闭所有功能，结束程序'''
    # 关闭读取摄像头与保存视频功能
    out_video.release()
    video_cap.release()
    cv2.destroyAllWindows()
    # 关闭检测功能
    pose_tracker.close()
    detector.close()

    print(f"输出视频保存在{out_video_path}")

    if dc:
        Chat_function_baidu()


if __name__ == '__main__':
    single_k(MODEL='video')
