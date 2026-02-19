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


def Chat_function():
    print("****************Chat function begin****************")

    '''录音'''
    pygame.mixer.init()  # 初始化混音器
    beep_sound = pygame.mixer.Sound("audio/beep.wav")  # 加载提示音
    speech_file_path = "./audio/speech_1.mp3"
    beep_sound.set_volume(0.1)  # 调节提示音音量 0.0 - 1.0
    fs = 16000  # 必须是
    seconds = 5
    '''openai & 音->字 and 字->音'''

    # 加载音频文件
    pygame.mixer.music.load(speech_file_path)
    # 播放音频文件
    print("语音播报ing...", "“我检测到您摔倒了，请问您是否需要帮助”")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

    # 播放完成后释放加载
    pygame.mixer.music.unload()

    gpt_get_num = 0
    while True:
        if gpt_get_num == 0:
            # 开始录音
            recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
            for i in range(seconds):
                if i % 1 == 0:  # 每秒播放一次提示音
                    print("录音ing...", i)
                    beep_sound.play()
                time.sleep(1)
            sd.wait()
            print("录音ing...done")
            write("audio/output.wav", fs, recording)
            print("save to", "audio/output.wav")

            # 获取语音I
            audio_file = "audio/output.wav"  # 必须是wav

            # 语音I转文字
            print("语音->text ing...")

            # 将音频文件转换为base64编码
            audio_len, speech = audio_to_base64(audio_file)
            # 请求的数据
            da_data = {
                "format": "wav",
                "rate": 16000,
                "dev_pid": 80001,
                "channel": 1,
                "token": "百度 access token", # 请替换为你的access token
                "cuid": "百度 用户唯一标识",  # 请替换为你的用户唯一标识
                "len": audio_len,
                "speech": speech
            }
            # 发送POST请求
            response = requests.post(bd_url, headers=bd_headers, data=json.dumps(da_data))
            get_wav = str(response.json()['result'][0])

            print("接收到语音:",get_wav)

            print("ChatGPT-4 working...")
            data = {
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "你是一位负责老人安全的家居智能客服，负责关注老人的身体健康安全，你能根据老人的要求进行以下回答："
                                                  "1.当老人表示需要向xxx求助时，你要回答：“1好的，已帮您联系xxx”"
                                                  "2.当老人表示需要120医疗救护服务时，你要回答：“2好的，已帮您拨打120”"
                                                  "3.当老人表示需要119意外救护服务时，你要回答：“3好的，已帮您拨打119”"
                                                  "4.当老人表示自己没有问题，不需要帮助时，你要回答：“4好的，随时为您服务”"
                                                  "5.如果你对他说的话表示不清楚，你要回答：“0我没有理解您的意思，请您再说一遍”"},
                    {"role": "user", "content": get_wav}
                ]
            }
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response_content = response.json()
            message = response_content['choices'][0]['message']['content']
            print("ChatGPT-4 working...done")

            gpt_get = str(message)
            print("ChatGPT-4 return:", gpt_get)
            gpt_get_num = int(gpt_get[0])
            gpt_get_str = gpt_get[1:]

            # 语音O播报
            GptResult_path = "./audio/GptResult.mp3"
            # Set the parameters for the synthesis request
            params = {
                'tex': gpt_get_str,
                'tok': "百度 access token",
                'cuid': '百度 用户唯一标识',
                'ctp': 1,
                'lan': 'zh',
                'spd': 5, # 语速
                'pit': 5, # 音调
                'vol': 4, # 音量
                'per': 5118, # 音库
                'aue': 3, # 格式3 mp3
            }
            # Send the synthesis request
            response = requests.post(synthesis_url, data=params)

            # Write the synthesized audio to a file
            with open(GptResult_path, 'wb') as f:
                f.write(response.content)

            pygame.mixer.music.load(GptResult_path)
            print("语音播报ing...", "" + gpt_get_str + "")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue

            # 播放完成后释放加载
            pygame.mixer.music.unload()

        elif 1 <= gpt_get_num <= 4:
            break
        else:
            print("error(1)")
            break

    print("****************Chat function over*****************\n")
    return


if __name__ == "__main__":
    Chat_function()
