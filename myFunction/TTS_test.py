from openai import OpenAI
import pygame

client = OpenAI()
pygame.mixer.init()

speech_file_path = "../audio/speech.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="您好，我是您的智能安全客服，我检测到您摔倒了，请问您是否需要帮助"
)

response.stream_to_file(speech_file_path)

# 加载音频文件
pygame.mixer.music.load("speech_file_path")

# 播放音频文件
pygame.mixer.music.play()

# 等待音频播放完成
while pygame.mixer.music.get_busy():
    continue


