import os, sys
import logging

# Silence tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import cv2
import numpy as np
import tensorflow as tf
import random

import threading
from imutils.video import VideoStream

import http.server
import socketserver

main_page = b""

with open("./templates/index.html", "rb") as f:
	main_page = f.read()


class WebHandler(http.server.SimpleHTTPRequestHandler):
	def __init__(self, *args, **kwargs):
		super(http.server.SimpleHTTPRequestHandler, self).__init__(*args, **kwargs)

	def handle_single_request(self, *args, **kwargs):
		self.do_GET()
		self.wfile.flush()

	def do_GET(self, *args, **kwargs):
		global main_page
		if self.path.startswith("/video_feed"):
			response = generate()
			self.send_response(200)
			self.send_header("Content-type", "image/png")
			self.send_header("Content-length", len(response))
			self.end_headers()
			self.wfile.write(response)
		else:
			response = main_page
			self.send_response(200)
			self.send_header("Content-type", "text/html")
			self.send_header("Content-length", len(response))
			self.end_headers()
			self.wfile.write(response)


httpd = socketserver.TCPServer(("", 7777), WebHandler)

REAL_CAMERA = 0
REAL_CAMERA_WIDTH = 640
REAL_CAMERA_HEIGHT = 480
REAL_CAMERA_FPS = 10
DO_HOLOGRAM = False

# find the GPU we want to work with
devices = tf.config.list_physical_devices()
for dev in devices:
	if dev.device_type == 'GPU':
		tf.config.experimental.set_memory_growth(dev, True)

keras_model = tf.keras.models.load_model('deconv_bnoptimized_munet.h5', compile=True)

def get_mask(frame):
	# Preprocess
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	simg = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
	simg = simg.reshape((1, 128, 128, 3)) / 255.0

	# Predict
	out = keras_model.predict(simg)

	#res = session.run(["op"],{"input_1",simg})
	#print("res", res)

	# Postprocess
	msk = out.reshape((128, 128, 1))
	mask = cv2.resize(msk, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

	return  mask


def post_process_mask(mask):
	_, mask = cv2.threshold(mask,0.5,1,cv2.THRESH_BINARY)
	kernel = np.ones((5,5),np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.blur(mask.astype(float), (10,10))
	return mask

def to_rgba(frame, mask):
	mask = mask.astype(np.uint8)
	b_channel, g_channel, r_channel = cv2.split(frame)
	return cv2.merge((b_channel, g_channel, r_channel, mask))

def shift_image(img, dx, dy):
	img = np.roll(img, dy, axis=0)
	img = np.roll(img, dx, axis=1)
	if dy>0:
		img[:dy, :] = 0
	elif dy<0:
		img[dy:, :] = 0
	if dx>0:
		img[:, :dx] = 0
	elif dx<0:
		img[:, dx:] = 0
	return img


def hologram_effect(img):
	# add a blue tint
	holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
	
	# add a halftone effect
	#bandLength, bandGap = 2, 3
	bandLength, bandGap = random.randint(1,4), random.randint(1,4)
	for y in range(holo.shape[0]):
		if y % (bandLength+bandGap) < bandLength:
			holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)

	# add some ghosting
	holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
	holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)

	# combine with the original color, oversaturated
	out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
	return out


def get_frame(cap, background_scaled):
	global lock, outputFrame
	_, frame = cap.read()

	# fetch the mask with retries (the app needs to warmup and we're lazy)
	# e v e n t u a l l y c o n s i s t e n t
	mask = None
	while mask is None:
		try:
			mask = get_mask(frame)
		except Exception:
			pass

	mask = post_process_mask(mask)

	if DO_HOLOGRAM:
		frame = hologram_effect(frame)

	with lock:
		outputFrame = to_rgba(frame, mask*255)

	# composite the foreground and background
	inv_mask = 1-mask
	for c in range(frame.shape[2]):
		frame[:,:,c] = frame[:,:,c]*mask + background_scaled[:,:,c]*inv_mask
	return frame


cap = cv2.VideoCapture(REAL_CAMERA)
height, width = REAL_CAMERA_HEIGHT, REAL_CAMERA_WIDTH
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, REAL_CAMERA_FPS)


background_scaled = np.ones((height,width,3))*[0,255,0]
outputFrame = None
lock = threading.Lock()

def generate():
	global outputFrame, lock
	while True:
		with lock:
			if outputFrame is None:
				continue
			(flag, encodedImage) = cv2.imencode(".png", outputFrame)
			if not flag:
				continue
			return bytearray(encodedImage)

def run_server():
	global httpd
	print("serving at port", 7777)
	httpd.serve_forever()

def run_cv():
	global cap
	global httpd
	run = True
	while run:
		frame = get_frame(cap, background_scaled)
		cv2.imshow('frame',frame)
		key = cv2.waitKey(10)
		if key == ord('q'):
			run = False
			httpd.shutdown()

cv_thread = threading.Thread(target=run_cv)
cv_thread.daemon = True
cv_thread.start()

http_thread = threading.Thread(target=run_server)
http_thread.daemon = True
http_thread.start()

cv_thread.join()
http_thread.join()



