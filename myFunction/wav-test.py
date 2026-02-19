"""
import sounddevice as sd
from scipy.io.wavfile import write
import pygame
import time
# 初始化混音器
pygame.mixer.init()
# 加载提示音
beep_sound = pygame.mixer.Sound("../audio/beep.wav")

fs = 44100
seconds = 10

recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)

# 开始录音
for i in range(seconds):
    # 每秒播放一次提示音
    if i % 1 == 0:
        beep_sound.play()
    time.sleep(1)
# 停止录音
sd.wait()

write("audio/output_test2.wav", fs, recording)
"""

from openai import OpenAI
import pygame

client = OpenAI()
pygame.mixer.init()

speech_file_path = "../audio/speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input="您好，我是您的智能安全客服，我检测到您摔倒了，请问您是否需要帮助"
)

response.stream_to_file(speech_file_path)

# 加载音频文件
pygame.mixer.music.load(speech_file_path)

# 播放音频文件
pygame.mixer.music.play()

# 等待音频播放完成
while pygame.mixer.music.get_busy():
    continue