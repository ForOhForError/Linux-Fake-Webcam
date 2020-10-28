# Linux-Fake-Webcam (OBS Source Fork)

Branch of Linux Fake Webcam for using as a Browser Source in OBS (Studio).

Setup:
* Set up a python 3 environment, and run ```python -m pip install -r requirements.txt``` to install all dependencies
* Edit the variables ```REAL_CAMERA```, ```REAL_CAMERA_WIDTH```, and ```REAL_CAMERA_HEIGHT``` to the index and dimensions of your webcam.
* run ```python greenscreen.py``` to launch the program. A preview window will appear showing the output to a greenscreen.
* Add ```localhost:7777``` as a browser source in OBS. Set its dimensions to the same specified above.
* To exit the program, focus the preview window and press the ```q``` key

# Credits
## Ben Elder
https://elder.dev/posts/open-source-virtual-background/
Wrote the original script

## rogierhofboer on HackerNews
https://news.ycombinator.com/item?id=22823070
Translated the get_mask function to Python

## Anil Sathyan aka anilsathyan7
https://github.com/anilsathyan7/Portrait-Segmentation
Model file used

## Fangfufu on Github
https://github.com/fangfufu/Linux-Fake-Background-Webcam
Guidance on getting v4l2loopback working

## Lukas Vacula aka ldv8434
The original get_mask function from rogierhofboer only worked under Tensorflow v1. I've ported it to v2, and made some other adjustments to the original script from Ben Elder.

# Requirements (based on arch repo names)
python-tensorflow
python-opencv

