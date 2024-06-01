# Automatic CPR Action Quality Assessment

This repository contains the code and pre-trained model for our recent paper on automatic CPR action quality assessment.

## Installation

To use our code, please follow the steps below:

### Step 1: Environment Setup

First, install the required dependencies. Run the following command:

```
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

To train on your own data, you could using MediaPipe to extract the pose information for your scenarios (See the official demo of Mediapipe: https://ai.google.dev/edge/mediapipe/solutions/guide). And you could save the extracted data in the npy files.

### Step 3: Train Your Own Model

To train your own model, run the following script:

```
python main_wandb.py
```
