# AI Voice Detector 🎙️

Can you tell if a voice is real or AI-generated? Turns out it's harder than I thought - both for humans AND for the model I built.

Built by Aanay Chandra · [GitHub repo](https://github.com/AanayChandraUConn/voice-detector)

## What this is

I built a model that listens to a voice clip and guesses if it's a real person or AI-generated. Started as a pretty standard classification project but turned into something more interesting once I tried testing it on AI voices from a generator it had never seen during training - that's where most of the actual learning happened.

Voice cloning scams have been in the news a lot lately, so this felt like a relevant problem, not just a random dataset I picked because it was available.

## The data

- 350 real human voice clips - people naturally answering questions, from the Fake-or-Real (FoR) dataset on Kaggle. Specifically used the `for-2sec` version so every clip is truncated to 2 seconds, which made things way easier to work with consistently
- 350 AI-generated clips from that same dataset - built using various TTS (text-to-speech) systems

I pulled these using the Kaggle API instead of downloading the whole dataset (which is like 20GB across all its versions) - wrote a script that pages through the file list and grabs just the training files I actually needed.

## Turning audio into something a model can learn from

Audio isn't numbers a model can just read, so I had to convert it first. Tried two approaches:

**MFCCs** - averaged each clip down into 13 numbers describing its tone/timbre. Simple, fast, but loses a lot of detail since you're squashing an entire 2-second clip into one flat summary.

**Mel spectrograms** - a full 128x63 grid showing frequency vs. time, basically a picture of the sound. Way more detail preserved, which is what let the CNN actually pick up on more subtle patterns.

## The models

**Baseline: Logistic Regression on the MFCCs**
Nothing fancy, just wanted a number to compare against before building anything complicated. Got 77.86% accuracy.

**CNN on the spectrograms**
Two conv layers + pooling, then a dense layer, ~865k parameters. First version I trained hit 100% training accuracy by epoch 7, which sounds great until you notice validation accuracy was stuck around 75-83% the whole time and validation loss was actually getting worse - classic overfitting, the model was just memorizing training clips instead of learning anything generalizable.

Fixed this with Dropout (randomly disables neurons during training so it can't over-rely on specific patterns) and Early Stopping (automatically stops training once validation performance stops improving, and rewinds to whichever epoch was actually best). Final result: 92.14% accuracy, and no more suspicious 100% training accuracy.

## The part I actually think is interesting

I wanted to check if my model was actually learning what makes AI voices sound fake in general, or if it just memorized the one dataset's specific TTS quirks. So I generated some brand new AI voices using ElevenLabs (a completely different voice generator than anything in my training data) and ran my model on those.

First test: 6 clips, one voice, got 5 right. 83% - felt great, thought I was basically done with this part.

Then I got paranoid about the tiny sample size and generated 16 more clips across 3 different ElevenLabs voices (Brian, Roger, Sarah - wanted some variety, not just one voice's characteristics). Accuracy dropped to 43.75%. Worse than a coin flip.

So my first result was just lucky - 6 clips is not enough to trust. The model didn't actually learn general "AI voice" patterns, it learned patterns specific to the Fake-or-Real dataset's particular voice generator. I tried fixing this by adding 10 of the ElevenLabs clips into training and holding back 6 as a genuine test - it barely helped, 50% on the clips it still hadn't seen, basically no better than before.

Honestly this was kind of a bummer to find after feeling good about the 83% number, but I think it's a more useful and honest thing to report than just picking my best result and moving on. It's also just... how ML actually goes a lot of the time - your first promising result isn't always the real story.

## Known limitations

- Only trained on one AI voice generator's output, so it doesn't generalize well to other generators (see above)
- Small dataset overall (700 clips) compared to production-scale voice detection systems
- Only tested generalization against one alternative generator (ElevenLabs) - would need to test against several to draw stronger conclusions

## Try it yourself

Built a small Streamlit app where you can upload an audio clip and get a live prediction with a confidence score. Fair warning based on everything above: it'll do great on clips similar to its training data, and probably struggle on voices from generators like ElevenLabs - you can actually see that limitation happen live if you try it.

To run it locally:
git clone https://github.com/AanayChandraUConn/voice-detector.git
cd voice-detector
pip install -r requirements.txt
streamlit run app.py

## If I had more time

- Train on a bunch of different voice generators instead of just one, so the model actually has to learn generalizable patterns instead of one dataset's quirks
- Look into feature representations that are less tied to one specific generator's audio characteristics
- Test on way more than 16 clips lol, that's still a pretty small sample to draw big conclusions from
- Maybe try a pretrained audio embedding model instead of building spectrograms from scratch

## Built with

Python, librosa, TensorFlow/Keras, scikit-learn, pandas, numpy, Streamlit

## Project structure
voice-detector/
├── src/
│   ├── download_data.py          (pulls data from Kaggle API)
│   ├── extract_features.py       (audio -> MFCCs)
│   ├── extract_spectrograms.py   (audio -> mel spectrograms)
│   ├── train_baseline.py         (logistic regression model)
│   ├── train_cnn.py              (CNN model)
│   ├── compare_models.py         (confusion matrices for both models)
│   └── test_generalization.py    (the ElevenLabs generalization test)
├── app.py                        (the streamlit demo)
└── models/                       (saved trained model, not in repo, generated locally)
