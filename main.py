import cv2
import tkinter as tk
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk
from demo_s_K import single_k as S_K
from demo_s_Nk import single_Nk as S_NK
from gpt import Chat_function as Chat_function_openai
from gpt_new import Chat_function as Chat_function_baidu

model = "live"  #  live video
DC = False  #  True False
if_falldown = [False, 'video-output/FallDown/fall-down.mp4']


class VideoPlayer:
    def __init__(self, root):
        self.closed = False
        self.root = root
        self.root.title("不倒翁")
        self.root.geometry('900x600')
        self.root.resizable(0, 0)
        self.frame0 = tk.Frame(root)

        self.buttons = tk.Frame(self.frame0)
        self.start_button = tk.Button(self.buttons, text="开始判断", font="Arial,20", command=self.start_fall_detection, width=20,
                                      height=2)
        # self.stop_button = tk.Button(self.buttons, text="停止判断", command=self.stop,width=12)
        self.open_button = tk.Button(self.buttons, text="打开视频", font="Arial,20", command=self.open_video, width=20,
                                     height=2)
        self.pause_button = tk.Button(self.buttons, text="暂停播放", font="Arial,20", command=self.pause_video,
                                      width=20, height=2)
        self.continue_button = tk.Button(self.buttons, text="继续播放", font="Arial,20", command=self.continue_video,
                                         width=20, height=2)
        self.close_button = tk.Button(self.buttons, text="关闭界面", font="Arial,20", command=self.close_video,
                                      width=20, height=2)
        self.gpt_button = tk.Button(self.buttons, text="语音功能", font="Arial,20", command=self.start_gpt,
                                    width=20, height=2)
        self.start_button.grid(row=0, column=0, pady=15)

        # self.stop_button.grid(row=1,column=0,pady=5)
        self.open_button.grid(row=1, column=0, pady=15)
        self.pause_button.grid(row=2, column=0, pady=15)
        self.continue_button.grid(row=3, column=0, pady=15)
        self.gpt_button.grid(row=4,column=0,pady=15)
        self.close_button.grid(row=5, column=0, pady=15)

        # video size
        self.v_width = 600
        self.v_height = 500
        image = Image.open(r"photo.png")
        self.image0 = image.resize((self.v_width, self.v_height))
        self.photo0 = ImageTk.PhotoImage(self.image0)
        self.canvas = tk.Canvas(self.frame0, width=self.v_width + 1, height=self.v_height + 1)
        self.canvas.create_image(0, 0, image=self.photo0, anchor=tk.NW)
        self.canvas.grid(row=0, column=0, pady=10, padx=(0, 20))
        self.buttons.grid(row=0, column=1, padx=(10, 0))
        self.frame0.pack()

        self.scale = tk.Scale(root, orient=tk.HORIZONTAL, resolution=1)  # 进度条
        self.scale.pack()

        self.video = None
        self.is_paused = False
        self.image_frame_all = 0

    def start_fall_detection(self):
        print("start testing")
        # 推荐使用SK或者MNK
        # MODEL=live表示调用摄像头实时检测，否则调用测试视频
        S_K(MODEL=model, if_falldown=if_falldown, dc=DC)
        #S_NK(MODEL=model, if_falldown=if_falldown, dc=DC)
        return

    def open_video(self):
        print("open the video   ", if_falldown)
        if not if_falldown[0]:
            # file_path = 'video-output/FallDown/fall-down.mp4'
            # 获取视频的文件地址
            file_path = filedialog.askopenfilename(initialdir="./video-output")
        else:
            file_path = if_falldown[1]
        # 导入本地视频
        self.video = cv2.VideoCapture(file_path)
        # 获取视频帧数
        self.image_frame_all = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        # 设置进度条的长度为视频的帧数
        self.scale.config(to=self.image_frame_all)
        self.play_video()

    def play_video(self):
        if self.video is None:
            return

        while not self.is_paused:
            image_frame_count = int(self.scale.get())

            if image_frame_count == self.image_frame_all:
                image_frame_count = 0

            self.video.set(cv2.CAP_PROP_POS_FRAMES, image_frame_count)

            is_over, image_frame = self.video.read()

            if not is_over:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

            if image_frame is not None:  # 检查视频帧是否为空
                image_frame = cv2.resize(image_frame, (self.v_width, self.v_height))
                cv2image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGBA)
                current_image = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=current_image)
                self.canvas.imgtk = imgtk
                # self.label.config(image=imgtk)
                self.canvas.create_image(0, 0, image=imgtk, anchor=tk.NW)
                self.root.update()
                image_frame_count += 1
                self.scale.set(image_frame_count)

    def pause_video(self):
        print("close video")
        self.is_paused = True

    def continue_video(self):
        print("Continue to play the video")
        self.is_paused = False
        self.play_video()

    def close_video(self):
        print("CLOSE")
        if self.video:
            self.video.release()
            cv2.destroyAllWindows()
        self.root.destroy()

    def start_gpt(self):
        print("start GPT function")
        #Chat_function_openai()
        Chat_function_baidu()
        return


if __name__ == '__main__':
    root = tk.Tk()
    player = VideoPlayer(root)
    root.mainloop()
