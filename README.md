
# Sonos_challenge

This Repo contain the source code for Voice Activity Detector task

>Author: Thai Ba Tuan: ba-tuan.thai@dauphine.eu
## 1. Introduction

**The main purpose of this competition is to design deep learning algorithms that detect the audio frames is speech or non specch.**

The project consists of two file: gen_data to extract features of data and train, train a Deep learning model and some jupyter notebook file for testing the model in audio file or real-time.

The models, training, testing Dataset can be download at this [Google Drive](https://drive.google.com/drive/folders/1N8JF0tk48oz3R4Z0XPKM5PNmYVZAilzj?usp=sharing)

## 2. Dataset
The vad_dataset contain 956 audio file, 16kHz, mono, each file have the label for the segments that have the speech. The total audio length is ~ 3h 

I used the external data LibriSpeech ASR corpus dataset(https://www.openslr.org/resources/12/test-clean.tar.gz) at sampling rate 16kHz consist of approximately 10 hours and English speech from audiobook.

The `label` of sepeech in this dataset was used as the training dataset in my project can download [here](https://drive.google.com/drive/folders/1ZPQ6wnMhHeE7XP5dqpAEmBAryFzESlin) provided by the [filippogiruzzi](https://github.com/filippogiruzzi/voice_activity_detection).

## 3. Run this project

### 3.1 Creating_data 

**Run script (Command Line)**

```python
python gendata.py 
```
This mode is used to create input  data including the four features(MFCC, MFCC-△, MFCC-△2, RMSE of mel-spectogram) of VAD model.
And save data set to `data_train_test` foler.

### 3.2 Training 

**Run script (Command Line)**

```python
python train.py --nepoch 50 --model-mode=0 --save-name='./models/full_con_vad_eh.h5' --batch=8196 --enhance
```

```
- --nepoch : Number of epochs for training
- --model-mode : 0 mean fully connected, 1 mean LSTM, 2 mean CNN, 3 mean Resnet
- --save-name : path to save model
- --batch : batch size of training the input for model in train will be (batch, 16, 65) or (batch, 16, 65, 1) for CNN and resnet
- --enhance: use external data or not
```
the external data can be download at [this link](https://drive.google.com/drive/folders/1N8JF0tk48oz3R4Z0XPKM5PNmYVZAilzj?usp=sharing)

### 3.3 Inference and real-time VAD
```
- Inference.ipynb : testing the VAD model in the audiofile 
- real_time_vac.ipynb : demo of real-time VAD
- Dataset_librosa.ipynb : Statistics of the vad dataset and demo of librosa function
```
##  4.TODO in the futures

- [ ] Train with noise dataset
- [ ] Extract diffrence fetures
- [ ] Use the smooth technique.
## 5. References

1.	https://github.com/filippogiruzzi/voice_activity_detection
2.	https://github.com/skgusrb12/voice_activity_detection
3.	Audio Signal Processing for Machine Learning Valerio Velardo - The Sound of AI [link](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)
4.	CS 224S / LINGUIST 285 Spoken Language Processing [link](http://web.stanford.edu/class/cs224s/syllabus/)

## 6. Milestones 

-	Monday (31/01/2022) to Wednesday  (02/02/2022): reading book, paper, watch video to understand the concept of speech processing.
-	Thursday (03/02/2022) and Friday (04/02/2022): Implements the VAD 
-	Saturday (05/02/2022) and Sunday (06/02/2022): Run all experiment and write report
