from openai import OpenAI
import sounddevice as sd
from scipy.io.wavfile import write
import pygame
import time


def Chat_function():
    print("****************Chat function begin****************")

    '''录音'''
    pygame.mixer.init()  # 初始化混音器
    beep_sound = pygame.mixer.Sound("audio/beep.wav")  # 加载提示音
    speech_file_path = "./audio/speech.mp3"
    beep_sound.set_volume(0.1)  # 调节提示音音量 0.0 - 1.0
    fs = 16000
    seconds = 5
    '''openai & 音->字 and 字->音'''
    client = OpenAI(

    )

    # 加载音频文件
    pygame.mixer.music.load(speech_file_path)
    # 播放音频文件
    print("语音播报ing...", "“您好，我是您的智能安全客服，我检测到您摔倒了，请问您是否需要帮助”")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

    gpt_get_num = 0

    while True:
        if gpt_get_num == 0:
            # 开始录音
            recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
            for i in range(seconds):
                if i % 1 == 0:  # 每秒播放一次提示音
                    print("录音ing...", i)
                    beep_sound.play()
                time.sleep(1)
            sd.wait()
            print("录音ing...done")
            write("audio/output.wav", fs, recording)
            print("save to", "audio/output.wav\n")

            # 获取语音I
            audio_file = "audio/test2.mp3"  # "audio/output.wav"
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

            print("ChatGPT-4 working...")
            completion = client.chat.completions.create(
                timeout=100000,
                model="gpt-4",  # "gpt-3.5-turbo"
                messages=
                [
                    {"role": "system", "content": "你是一位负责老人安全的家居智能客服，负责关注老人的身体健康安全，你能根据老人的要求进行以下回答："
                                                  "1.当老人表示需要向xxx求助时，你要回答：“1好的，已帮您联系xxx”"
                                                  "2.当老人表示需要120医疗救护服务时，你要回答：“2好的，已帮您拨打120”"
                                                  "3.当老人表示需要119意外救护服务时，你要回答：“3好的，已帮您拨打119”"
                                                  "4.当老人表示自己没有问题，不需要帮助时，你要回答：“4好的，随时为您服务”"
                                                  "5.如果你对他说的话表示不清楚，你要回答：“0我没有理解您的意思，请您再说一遍”"},
                    {"role": "user", "content": transcript}
                ]
            )
            print("ChatGPT-4 working...done")

            gpt_get = str(completion.choices[0].message.content)
            print("ChatGPT-4 return:", gpt_get)
            gpt_get_num = int(gpt_get[0])
            gpt_get_str = gpt_get[1:]

            # 语音O播报
            GptResult_path = "./audio/GptResult.mp3"
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=gpt_get_str
            )
            response.stream_to_file(GptResult_path)
            pygame.mixer.music.load(GptResult_path)
            print("语音播报ing...", "“" + gpt_get_str + "”\n")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue

        elif 1 <= gpt_get_num <= 4:
            break
        else:
            print("error(1)")
            break

    print("****************Chat function over*****************\n")
    return
