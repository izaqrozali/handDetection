import cv2
import imutils
import sys
import numpy as np

def autocenter(bbox, pad={'r':100, 'l':50, 't':50, 'b':100}, ratio=1):
    x,y,w,h=bbox
    ratio_real = h/w
    if w>h:
        hn = int(w * ratio)
        wn = w
    else:
        hn = h
        wn = int(h * ratio)
    
    yn=(y + (h//2)) - (hn//2)
    xn=(x + (w//2)) - (wn//2)
    xn = xn - pad['l']
    yn = yn - pad['t']
    hn = hn + pad['b']
    wn = wn + pad['r']
    
    return (xn, yn, wn, hn)

def get_mask(frame):
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 10)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 200, 50])
    upper = np.array([255, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    return mask

def crop_bbox(frame, bbox):
    x1,y1,w1,h1=autocenter(bbox)
    return frame[y1:y1+h1, x1:x1+w1]

def draw_rectangle(frame,bbox):
    x1,y1,w1,h1=autocenter(bbox)
    cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (0,255,0),2)
    return frame

def get_large_contour(mask):
    im, cntr, hierarcy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find contours of large enough area
    min_coin_area = 600
    large_contours = [cnt for cnt in cntr if cv2.contourArea(cnt) > min_coin_area]
    return large_contours

def get_cropped_image(frame, large_contours):
    for idx in range(len(large_contours)):
        bbox = cv2.boundingRect(large_contours[idx])
        rect = draw_rectangle(frame,bbox)
        frame = crop_bbox(rect, bbox)
    return frame

def get_bbox(frame, large_contours):
    for idx in range(len(large_contours)):
        bbox = cv2.boundingRect(large_contours[idx])
        rect = draw_rectangle(frame,bbox)
    return rect
