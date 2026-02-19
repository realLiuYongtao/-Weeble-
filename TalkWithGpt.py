from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import pygame
import time

client = OpenAI()

GET = [
    {"role": "system", "content": "你是一位智能聊天机器人，可以与用户沟通互动聊天，如果用户表示结束对话"
                                  "你只需要回答：“0” "},
    {"role": "user", "content": "让我们开始聊天吧"}
]

while True:
    pygame.mixer.init()  # 初始化混音器
    beep_sound = pygame.mixer.Sound("audio/beep.wav")  # 加载提示音
    beep_sound.set_volume(0.1)  # 调节提示音音量 0.0 - 1.0
    fs = 44100
    seconds = 5

    # 开始录音
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    for i in range(seconds):
        if i % 1 == 0:  # 每秒播放一次提示音
            print("录音ing...", i)
            beep_sound.play()
        time.sleep(1)
    sd.wait()
    print("录音ing...done")
    write("audio/TalkWithGptOutput.wav", fs, recording)
    print("save to", "audio/TalkWithGptOutput.wav\n")

    # 获取语音I
    audio_file = "audio/TalkWithGptOutput.wav"
    audio = open(audio_file, "rb")

    # 语音I转文字
    print("语音->text ing...")
    transcript = client.audio.transcriptions.create(
        timeout=100000,
        model="whisper-1",
        file=audio,
        response_format="text"
    )
    print("接收到语音:", str(transcript))

    # GET.append(str(transcript))
    GET.append({"role": "user", "content": str(transcript)})

    print("ChatGPT-4 working...")
    completion = client.chat.completions.create(
        timeout=100000,
        model="gpt-3.5-turbo",  # "gpt-4"
        messages=GET
    )
    print("ChatGPT-4 working...done")
    gpt_get = str(completion.choices[0].message.content)
    if len(gpt_get) == 1 and int(gpt_get) == 0:
        break
    TalkWithGptGet_path = "./audio/TalkWithGptGet.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=gpt_get
    )
    response.stream_to_file(TalkWithGptGet_path)
    pygame.mixer.music.load(TalkWithGptGet_path)
    print("语音播报ing...", "“" + gpt_get + "”\n")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
