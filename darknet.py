from ctypes import *
import math
import random
import cv2
import numpy as np

CUP_ID = 41
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    return (ctype * len(values))(*values)

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [c_void_p, IMAGE, c_float, c_float, c_float, POINTER(BOX), POINTER(POINTER(c_float))]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, im, thresh=.15, hier_thresh=.5, nms=.45):
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            print(i,  meta.names[i])
            if probs[j][i] > 0:
                res.append((meta.names[i], probs[j][i], (boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res

def detect_cup(net, meta, im, thresh=.15, hier_thresh=.5, nms=.45):
    boxes = make_boxes(net)
    probs = make_probs(net)
    num =   num_boxes(net)
    network_detect(net, im, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        if probs[j][CUP_ID] > 0:
            res.append((boxes[j].x, boxes[j].y, boxes[j].w, boxes[j].h))
    #res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res

class CupDetector():
    def __init__(self):
        self.net = load_net(b"cfg/yolo.cfg", b"yolo.weights", 0)
        #self.net = load_net(b"cfg/tiny-yolo-voc.cfg", b"tiny-yolo-voc.weights", 0)
        self.meta = load_meta(b"cfg/coco.data")
        #self.meta = load_meta(b"cfg/coco.data")
        pass
    
    def __call__(self, image): #, target='cup'):
        #target = target.encode('UTF-8')
        result = detect_cup(self.net, self.meta, image)
        #rtn = [row[2] for row in result if row[0] == target]
        #rtn = [row[2] for row in result]
        return result

    def file_detect(self, path):
        image = load_image(path.encode("UTF-8"), 0, 0)
        return self.__call__(image)

def visualize(path, diags):
    img=cv2.imread(path)
    trimmed = []
    for diag in diags:
        p = (diag[0]-diag[2]/2, diag[1]-diag[3]/2)
        q = (p[0]+diag[2], p[1]+diag[3])
        p = tuple(map(int, p))
        q = tuple(map(int, q))
        trimmed.append(img[p[1]:q[1], p[0]:q[0]])
        #cv2.rectangle(img, p, q, (0,0,255), 10)
    for i, t in enumerate(trimmed):
        h = cv2.cvtColor(t, cv2.COLOR_BGR2HSV) 
        lower = np.array((12,128,128))
        higher = np.array((25,255,255))
        img_mask = cv2.inRange(h, lower, higher)
        cv2.imshow(str(i)+"mask",img_mask)
        cv2.imshow(str(i)+"img",t)
        cv2.waitKey(0)
    return trimmed

def visualize2(path, diags):
    original=cv2.imread(path)
    img = original.copy()
    trimmed = []
    for diag in diags:
        p = (diag[0]-diag[2]/2, diag[1]-diag[3]/2)
        q = (p[0]+diag[2], p[1]+diag[3])
        p = tuple(map(int, p))
        q = tuple(map(int, q))
        trimmed.append(img[p[1]:q[1], p[0]:q[0]])
        cv2.rectangle(img, p, q, (0,0,255), 5)
    
    h = cv2.cvtColor(original, cv2.COLOR_BGR2HSV) 
    lower = np.array((12,128,128))
    higher = np.array((25,255,255))
    img_mask = cv2.inRange(h, lower, higher)
        #cv2.imshow(str(i)+"mask",img_mask)
        #cv2.imshow(str(i)+"img",t)
        #cv2.waitKey(0)
    #cv2.imshow("img", img)
    #cv2.imshow("mask", img_mask)
    #cv2.waitKey(0)
    cv2.imwrite("results_cupDetect/"+"img"+path[-6:], img)
    cv2.imwrite("results_cupDetect/"+"mask"+path[-6:], img_mask)
    return trimmed

cd = CupDetector()
def test(path):
    import cv2
    import matplotlib.pyplot as plt
    diags = cd.file_detect(path)
    trimmed= visualize2(path, diags)
    #hsv = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in trimmed]
    #return hsv
        

if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    
    #"""
    net = load_net(b"cfg/yolo.cfg", b"yolo.weights", 0)
    meta = load_meta(b"cfg/coco.data")
    image = load_image(b"1.png", 0, 0)
    r = detect(net, meta, image)
    print(r)
    #"""

