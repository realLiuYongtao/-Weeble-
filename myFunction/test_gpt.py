import requests
import json
import base64
import os

url = "https://api.openai-proxy.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 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"
}
data = {
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "system", "content": "你是一位负责老人安全的家居智能客服，负责关注老人的身体健康安全，你能根据老人的要求进行以下回答："
                                      "1.当老人表示需要向xxx求助时，你要回答：“1好的，已帮您联系xxx”"
                                      "2.当老人表示需要120医疗救护服务时，你要回答：“2好的，已帮您拨打120”"
                                      "3.当老人表示需要119意外救护服务时，你要回答：“3好的，已帮您拨打119”"
                                      "4.当老人表示自己没有问题，不需要帮助时，你要回答：“4好的，随时为您服务”"
                                      "5.如果你对他说的话表示不清楚，你要回答：“0我没有理解您的意思，请您再说一遍”"},
        {"role": "user", "content": "我摔倒了，请帮我叫救护车"}
    ]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
response_content = response.json()
message = response_content['choices'][0]['message']['content']
print(message)


# ******************************************************************************************


# 将音频文件转换为base64编码
def audio_to_base64(filepath):
    with open(filepath, 'rb') as f:
        audio_data = f.read()
    audio_len = len(audio_data)
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    return audio_len, audio_base64


# 百度语音识别API的URL
url = "https://vop.baidu.com/pro_api"

# 音频文件的路径
filepath = "../audio/16k.wav"

# 将音频文件转换为base64编码
audio_len, speech = audio_to_base64(filepath)

# 请求的headers
headers = {
    "Content-Type": "application/json"
}

# 请求的数据
data = {
    "format": "wav",
    "rate": 16000,
    "dev_pid": 80001,
    "channel": 1,
    "token": "25.761dd46c56c8192b5505676c129ef6ec.315360000.2018254522.282335-45239324",  # 请替换为你的access token
    "cuid": "45239324",  # 请替换为你的用户唯一标识
    "len": audio_len,
    "speech": speech
}

# 发送POST请求
response = requests.post(url, headers=headers, data=json.dumps(data))

# 打印响应
print(response.json())
