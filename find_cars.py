import argparse
import os
import python.darknet as dn
#video reading
import numpy as np
import cv2
from time import sleep

def init_net():
    net = dn.load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    meta = dn.load_meta("cfg/coco.data")
    return net, meta

def detect_video(video_loc, frames_to_skip, out_dir, threshold):

    #Start yolonet
    net, meta = init_net()

    nr = 0
    skip = frames_to_skip
    frames_loc = 'videoframes'
    files_with_cars = []

    if not os.path.exists(frames_loc):
    	os.makedirs(frames_loc)

    if not os.path.exists(out_dir):
    	os.makedirs(out_dir)


    # Read input video using cv2
    cap = cv2.VideoCapture(video_loc)

    while(cap.isOpened()):
    
        ret, frame = cap.read()

	# Stop if the video is over 
        if ret == False:
		break

        nr += 1

	if nr % skip != 0:
		continue


        # save the frame so darknet could detect it
	# could be skipped and feed image straight to network
	name = frames_loc + '/frame' + str(nr) + '.jpg'
	cv2.imwrite(name, frame)

	r = dn.detect(net, meta, name, thresh = threshold)

	# only save image if there is a car in frame
	cars = [x for x in r if x[0] == 'car']

	if len(cars) > 0:
		outname = out_dir +'/frame' + str(nr) + '.jpg' 
		files_with_cars.append(outname)             	
		cv2.imwrite(outname, frame)	

	#cv2.imshow('frame',frame)
	print "Found {} car(s) from frame {}".format(str(len(cars)), str(nr))
	for _, conf, coords in cars:
			print "\tConfidence {}".format(conf) 

    print "Files with cars: {}".format(", ".join(files_with_cars))
    cap.release()
    #cv2.destroyAllWindows()
    import shutil
    shutil.rmtree(frames_loc, ignore_errors=False)


parser = argparse.ArgumentParser()

parser.add_argument("-v", "--video", help="path to input video", required=True)
parser.add_argument("-o", "--output", help="output path for detected cars",
		default = "detected_cars")
parser.add_argument("-s", "--skip", help="number of frames to skip",
		default = 15, type=int)
parser.add_argument("-t", "--threshold", help="threshold for darknet",
		default = 0.5, type=int)

args = parser.parse_args()
print 'args'
print args

detect_video(args.video, args.skip, args.output, args.threshold)
