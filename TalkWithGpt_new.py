import sounddevice as sd
from scipy.io.wavfile import write
import pygame
import time
import requests
import json
import base64

# openai
url = "https://api.openai-proxy.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer openai密钥"
}


# baidu
def audio_to_base64(filepath):
    with open(filepath, 'rb') as f:
        audio_data = f.read()
    audio_len = len(audio_data)
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    return audio_len, audio_base64


# 百度语音识别API的URL
bd_url = "https://vop.baidu.com/pro_api"
synthesis_url = 'https://tsn.baidu.com/text2audio'
# 请求的headers
bd_headers = {
    "Content-Type": "application/json"
}

GET = [
    {"role": "system", "content": "你是一位智能聊天机器人，可以与用户沟通互动聊天，如果用户表示结束对话"
                                  "你只需要回答：“0” "},
    {"role": "user", "content": "让我们开始聊天吧"}
]


pygame.mixer.init()  # 初始化混音器
speech_file_path = "./audio/speed_2.mp3"
# 加载音频文件
pygame.mixer.music.load(speech_file_path)
# 播放音频文件
print("语音播报ing...", "“有什么可以帮助你的吗”")
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    continue

# 播放完成后释放加载
pygame.mixer.music.unload()
while True:
    '''录音'''
    beep_sound = pygame.mixer.Sound("audio/beep.wav")  # 加载提示音
    speech_file_path = "./audio/speech.mp3"
    beep_sound.set_volume(0.1)  # 调节提示音音量 0.0 - 1.0
    fs = 16000  # 必须是
    seconds = 5

    # 开始录音
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
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
    # 将音频文件转换为base64编码
    audio_len, speech = audio_to_base64(audio_file)
    # 请求的数据
    da_data = {
        "format": "wav",
        "rate": 16000,
        "dev_pid": 80001,
        "channel": 1,
        "token": "百度 access token",  # 请替换为你的access token
        "cuid": "百度 用户唯一标识",  # 请替换为你的用户唯一标识
        "len": audio_len,
        "speech": speech
    }
    # 发送POST请求
    response = requests.post(bd_url, headers=bd_headers, data=json.dumps(da_data))
    transcript = str(response.json()['result'][0])
    print("接收到语音:", str(transcript))

    # GET.append(str(transcript))
    GET.append({"role": "user", "content": str(transcript)})

    print("ChatGPT-4 working...")
    data = {
        "model": "gpt-4",
        "messages": GET,
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_content = response.json()
    message = response_content['choices'][0]['message']['content']
    print("ChatGPT-4 working...done")
    gpt_get = str(message)
    if len(gpt_get) == 1 and int(gpt_get) == 0:
        break
    TalkWithGptGet_path = "./audio/TalkWithGptGet.mp3"
    params = {
        'tex': gpt_get,
        'tok': "百度 access token",  # 请替换为你的access token
        'cuid': '百度 用户唯一标识',
        'ctp': 1,
        'lan': 'zh',
        'spd': 5,
        'pit': 5,
        'vol': 5,
        'per': 5118,
        'aue': 3,
    }
    # Send the synthesis request
    response = requests.post(synthesis_url, data=params)

    # Write the synthesized audio to a file
    with open(TalkWithGptGet_path, 'wb') as f:
        f.write(response.content)

    pygame.mixer.music.load(TalkWithGptGet_path)
    print("语音播报ing...", "“" + TalkWithGptGet_path + "”\n")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

    # 播放完成后释放加载
    pygame.mixer.music.unload()



