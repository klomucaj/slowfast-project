# slowfast-project

SlowFast Video Training Project
This repository contains a CPU-based implementation of the SlowFast model for video training, focusing on processing videos with corresponding subtitle (.srt) files.
Table of Contents
1.	Introduction
2.	Features
3.	Requirements
4.	Installation
5.	Dataset Preparation
6.	Usage
7.	Project Structure
8.	Acknowledgements
________________________________________
Introduction
The SlowFast model is a video understanding framework that processes videos at both slow and fast pathways to capture motion and context. This project adapts the model for CPU usage and includes functionality for training, validation, and inference.
________________________________________
Features
●	Custom implementation of the SlowFast architecture.

●	Support for .srt file annotations for video processing.

●	Modular design for datasets, models, and utilities.

●	Scripts for training, validation, and testing.
________________________________________
Requirements
Before starting, ensure the following are installed:

●	Python >= 3.8

●	pip >= 20.0

●	ffmpeg (for video processing)

Install the required Python libraries using:
pip install -r requirements.txt

________________________________________
Installation

Step 1: Clone the Repository
git clone https://github.com/klomucaj/slowfast-project.git
cd slowfast-project

Step 2: Set Up the Environment
It is recommended to use a virtual environment for dependency management.
Using venv:
python -m venv slowfast_env
source slowfast_env/bin/activate  # For Linux/Mac
slowfast_env\Scripts\activate    # For Windows

Using conda:

conda create --name slowfast_env python=3.8
conda activate slowfast_env

Step 3: Install Dependencies

Install the required Python libraries:
pip install -r requirements.txt

________________________________________
Dataset Preparation
1.	Place your video files in the datasets/videos/ directory:

○	Training videos: datasets/videos/train/

○	Validation videos: datasets/videos/val/

○	Testing videos: datasets/videos/test/

2.	Add subtitle .srt files in datasets/annotations/:

○	Training: datasets/annotations/srt_train/

○	Validation: datasets/annotations/srt_val/

○	Testing: datasets/annotations/srt_test/

3.	Use the script utils/generate_json_annotations.py to convert .srt files into JSON annotations:

python utils/generate_json_annotations.py

4.	Split the dataset into train/val/test if needed using:
python datasets/split_dataset.py

________________________________________
Usage

Training

Run the main training script:

python train.py --config configs/slowfast_training.yaml

Validation

Evaluate the model:

python validate.py --config configs/slowfast_validation.yaml

Testing/Inference

Perform inference on test data:

python test.py --config configs/slowfast_inference.yaml

________________________________________

Acknowledgements

This project builds upon the principles outlined in the SlowFast repository by Facebook Research.
________________________________________
