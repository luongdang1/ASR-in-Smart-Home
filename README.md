# ASR-in-Smart-Home
# Introduction
SmartHome-ASR is a speech recognition system built base on the DeepSpeech2 architecture (originally from Mozilla), trained from scratch with the [Fluent Speech Commands Voice Dataset](https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus) to support smart home applications.

# Architecture Model 
[1]
![image](https://github.com/user-attachments/assets/92eba511-d473-4978-8369-553a4b9785f3) 

Each audio command (.wav file, 16kHz, mono) is transformed into a spectrogram via Short-Time Fourier Transform (STFT), effectively representing intensity and frequency characteristics over time. The spectrograms are then fed into a series of convolutional neural networks (CNNs) to extract local features and reduce noise, followed by five layers of Bidirectional Gated Recurrent Units (Bi-GRU) to capture temporal context in both directions. The output is processed through Dense + ReLU layers and trained using Connectionist Temporal Classification (CTC) Loss with Greedy Search , enabling the model to flexibly map audio feature sequences to character sequences without requiring frame-level alignment. The entire model is well-suited for speech recognition applications in controlling smart home devices

[2]

![image](https://github.com/user-attachments/assets/c172e7f2-f414-4de3-88b9-26d1bb97e461)

For more details on the SmartHome-ASR model's architecture and operation refer to the [documentation](https://deepspeech.readthedocs.io/en/v0.6.1/DeepSpeech.html0) [3]. Additionally, a related paper can be found here [in here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10966154) [4]

To utilize the pre-trained model, you can use [best_model.weights.h5] and the TFLite format [model.tflite] for efficient deployment.
# Dataset
[Author: Tommy NgX](https://www.kaggle.com/tommyngx)

[Fluent Speech Commands Dataset](https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus) [5]

This dataset contains 97 speakers recording 248 unique phrases, mapping to 31 intents across three slots: action, object, and location. Designed as a benchmark for end-to-end spoken language understanding, it was collected via crowdsourcing in the US and Canada. Participants recorded each phrase twice in random order, with anonymized demographic data included. Audio was validated to remove noisy, inaudible, or incorrect recordings. Licensed under the Fluent Speech Commands Public License.

# Quick start 
You can refer to [this notebook](https://github.com/luongdang1/ASR-in-Smart-Home/blob/main/asr_speech_recognition.ipynb) to easily learn how to use it
# Raspberry Pi Deployment 
The trained SmartHome-ASR model is deployed on a Raspberry Pi to enable a fully functional voice-controlled smart home assistant. Voice commands are recognized in real time using the optimized TFLite version of the model. Based on recognized commands, the system can control lights, fans, or other appliances through GPIO-connected relays. It also integrates environmental sensors like the DHT11 to monitor temperature and humidity, and a PIR motion sensor for detecting human presence. Users can interact with the system via a local Tkinter-based GUI, while music playback is supported through VLC. Additionally, the system can control servo motors for mechanical actions such as adjusting blinds or doors, making it a versatile and responsive smart home solution.
# Clone projecet
git clone https://github.com/your-username/SmartHome-ASR.git

cd SmartHome-ASR

# Requirement setting
pip install -r requirements.txt

# References 
"Some of the papers I referred to include:"

Optimizing Speech Recognition for the Edge — [arXiv:1909.12408](https://arxiv.org/abs/1909.12408)

Tiny Transducer: A Highly Efficient Speech Recognition Model for Edge Devices — [arXiv:2101.06856](https://arxiv.org/pdf/2101.06856)

Conformer-Based Speech Recognition on Extreme Edge-Computing Devices — [arXiv:2312.10359](https://arxiv.org/pdf/2312.10359)

https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html [1]

https://www.researchgate.net/figure/DeepSpeech-2-architecture41_fig23_348706070 [2]

https://deepspeech.readthedocs.io/en/v0.6.1/DeepSpeech.html [3]

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10966154 [4]

https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus [5]

https://github.com/mozilla/DeepSpeech 
# License
The project is released under the MIT License.
