# -Weeble-
利用ST-GCN实现摔倒动作自动检测，并能够接通经过prompt的chatGPT-4客服提供帮助

<img width="689" height="477" alt="image" src="https://github.com/user-attachments/assets/dfd76eb6-ef28-4dd4-917b-e352814da816" />


系统通过摄像头实时收集的视频帧，经mediapipe Object Detector & Pose Landmark 处理得到骨骼序列,通过ST-GCN实现实时的动作判断

在ST-GCN模型中，利用骨骼关键点的运动特征建立时空特征图，对得到的特征图进行分类操作，同时对多层时空图进行卷积操作，形成能更高效表示样本分类的特征图，通过分类网络得到动作类别

项目使用OpenAI接口调用GPT-4，主要有语音交流和跌倒响应两部分功能

· 跌倒响应
当系统判断有人跌倒时，向用户以语音形式发出帮助问询，根据用户回答拨打紧急联系人或者急救电话

· 语音交流
接入GPT-4增强系统-用户的交互性

<img width="624" height="322" alt="image" src="https://github.com/user-attachments/assets/35a57895-ed17-4649-b2d7-20e0c8de1538" />
