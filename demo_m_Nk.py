"""读取本地视频检测关键点"""
import cv2
from collections import deque
import numpy as np
import datetime
import os

import mediapipe as mp
from mediapipe.tasks import python

from SGTCN.stgcn.stgcn import STGCN

from myFunction.FilteringKeyPoints2 import filterKP as KP2
from myFunction.draw_chinese_text import add_chinese_text
from myFunction.drawP import drawPoint


def multiple_Nk(if_falldown=None,MODEL='live'):

    if if_falldown is None:
        if_falldown = [False, 'video-output/FallDown/fall-down.mp4']

    global fall_down_video, fall_down_video_path

    out_video_path = './video-output'
    in_video_path = 'video-test/2.mp4'  # "D:/AIAM_project/test_demo/data/dataset/train/normal_18.mp4"
    fall_down_path = './video-output/FallDown'
    video_fps = 24
    ACTION_MODEL_MAX_FRAMES = 30  # 单位：帧
    device = 'cpu'
    PointThinness = 2
    BoxThinness = 2
    TestThinness = 30
    Max_Person_Num = 3

    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    if MODEL == 'live':
        video_cap = cv2.VideoCapture(0)
    else:
        video_cap = cv2.VideoCapture(in_video_path)

    # 图像大小
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    d = [-100, -100, 100, 100]
    num_fall_down = 0

    '''检测器'''

    # 点
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='ModelPath/pose_landmarker_heavy.task'),
        min_pose_presence_confidence=0,
        num_poses=Max_Person_Num,
        running_mode=VisionRunningMode.IMAGE)
    # 点检测
    pose_tracker = PoseLandmarker.create_from_options(options)

    # 动作检测
    action_model = STGCN(weight_file='SGTCN/weights/ts-stg-model-1.pth', device=device)

    # 获取当前时间
    makefile_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('./video-output'):
        os.mkdir('./video-output')
    out_video_path = out_video_path + '/' + makefile_time + '.mp4'

    # 创建一个VideoWriter对象,用于后续将图像框架写入视频文件。
    out_video = cv2.VideoWriter(out_video_path,  # 输出视频的路径及文件名
                                cv2.VideoWriter_fourcc(*'mp4v'),  # 指定视频编码格式为mp4
                                video_fps, (video_width, video_height))  # 设置视频帧率 分辨率

    video_pose_landmark = []
    # 记录一组 pose_landmarks 传入时序网络
    for i in range(Max_Person_Num):
        m = deque(maxlen=ACTION_MODEL_MAX_FRAMES)
        video_pose_landmark.append(m)

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

        # 点
        # numpy_image = np.array(middle_frame)
        numpy_image = np.array(input_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)

        PoseLandmarkerResult = pose_tracker.detect(image=mp_image)
        pose_landmarks = PoseLandmarkerResult.pose_landmarks

        output_frame = middle_frame.copy()
        if len(pose_landmarks) != 0:
            for idx in range(len(pose_landmarks)):  # 实际上这里的idx只能是0
                # 关键点：
                pic_pose_landmark = np.array(KP2(pose_landmarks[idx], video_width, video_height))  # 关键点转化
                video_pose_landmark[idx].append(pic_pose_landmark)

                # 绘制关键点
                output_frame = drawPoint(frame=output_frame, pts=pic_pose_landmark, thinckness=PointThinness)  # 自定义绘制工具

                # 人体框
                box_l, box_r = (max(int(pic_pose_landmark[:, 0].min()) + d[0], 0),
                                min(int(pic_pose_landmark[:, 0].max()) + d[2], video_width))
                box_t, box_b = (max(int(pic_pose_landmark[:, 1].min()) + d[1], 0),
                                min(int(pic_pose_landmark[:, 1].max()) + d[3], video_height))

                '''mian 时序网络'''
                if len(video_pose_landmark[idx]) == ACTION_MODEL_MAX_FRAMES:
                    pts = np.array(video_pose_landmark[idx], dtype=np.float32)
                    out = action_model.predict(pts, (video_height, video_width))
                    action_name = action_model.class_names[out[0].argmax()]
                    confid = out[0].max() * 100

                    # 绘制状态
                    if action_name == 'Fall Down':
                        num_fall_down += 1
                        clr = (0, 0, 255)
                    else:
                        action_name = 'fine'
                        clr = (0, 0, 0)

                    output_frame = cv2.rectangle(output_frame, (box_l, box_t),
                                                 (box_r, box_b), clr, BoxThinness)
                    output_frame = add_chinese_text(output_frame,
                                                    f'idx: {idx}\ndevice: {device}\nstate: {action_name}\nconfidence: {confid}',
                                                    (box_l, box_t), clr, TestThinness)

        '''显示结果并保存视频'''
        output_frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR)
        # 实时输出检测画面
        cv2.imshow('video', output_frame)
        # print('                                                        ', action_name)

        # ---------------------------------------------------------------------------------
        if num_fall_down > 0:
            if key == 0:  # 跌倒过程开始
                # 创建一个VideoWriter对象,用于后续将图像框架写入视频文件。

                if not os.path.exists(fall_down_path):
                    os.mkdir(fall_down_path)
                makefile_time = datetime.datetime.strftime(datetime.datetime.now(),
                                                           '%Y-%m-%d-%H-%M-%S')
                fall_down_video_path = (fall_down_path + '/' + 'fall-down' +
                                        makefile_time + '.mp4')

                fall_down_video = cv2.VideoWriter(fall_down_video_path,
                                                  cv2.VideoWriter_fourcc(*'mp4v'),
                                                  16, (video_width, video_height))  # 视频帧率 分辨率

            # 跌倒过程
            key += 1
            print(key)
            FD.append(output_frame)

        elif num_fall_down == 0 and key == 0:  # 未跌倒
            FD.append(output_frame)

        elif num_fall_down == 0 and key != 0:  # 跌倒过程结束

            for fd in FD:
                fall_down_video.write(fd)
            fall_down_video.release()
            print("迭倒视频保存在：", fall_down_video_path)

            if_falldown[0] = True
            if_falldown[1] = fall_down_video_path

            key = 0
        # ---------------------------------------------------------------------------------

        # 保存out_video
        out_video.write(output_frame)
        # 按键盘的q或者esc退出
        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    '''关闭所有功能，结束程序'''
    # 关闭读取摄像头与保存视频功能
    out_video.release()
    video_cap.release()
    cv2.destroyAllWindows()
    # 关闭关键点检测功能
    pose_tracker.close()

    print(f"输出视频保存在{out_video_path}")


if __name__ == '__main__':
    multiple_Nk([])
