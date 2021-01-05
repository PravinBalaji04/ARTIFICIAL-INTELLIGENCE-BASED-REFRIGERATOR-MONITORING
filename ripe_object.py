import cv2
import imutils
import numpy as np
from twilio.rest import Client


# Your Account SID from twilio.com/console
account_sid = "AC57c573942a181f60d2228e7423386a96"
# Your Auth Token from twilio.com/console
auth_token  = "bd6b30fe495a532f607849c1a6d1075b"
# Load Yolo
lower = {'brown':(10, 100, 20)}
upper = {'brown':(20, 255, 200)}
def sms(String):
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        from_="+12029527065", 
        to="+917708601554",
        body=String)

    print(message.sid)

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture(0)
a=0
b=0
p=0
d=0
e=0
f=0
g=0
count=0
raduis = 0
while True:
    count+=1
    apple = 0
    orange = 0
    cake = 0
    carrot = 0
    broccoli = 0
    banana = 0
    sandwich = 0
    ret,img=cap.read()
    img = imutils.resize(img, width=400)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)
    frame = imutils.resize(img, width=600)
 
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    for key, value in upper.items():
        # construct a mask for the color from dictionary`1, then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        kernel = np.ones((11,11),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE)[-2]
        center = None
       
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if(raduis>.5):
                cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                cv2.putText(frame,"Ripe", (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                print("::::::::::Ripe stock detected::::::::")
                sms("Ripe stock detected")
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
##            print(confidence)
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if(classes[class_ids[i]] == "apple"):
                apple+=1
                a=0
            if(classes[class_ids[i]] == "orange"):
                orange+=1
                b=0
            if(classes[class_ids[i]] == "banana"):
                banana+=1
                d=0
            if(classes[class_ids[i]] == "sandwich"):
                sandwich+=1
                e=0
            if(classes[class_ids[i]] == "broccoli"):
                broccoli+=1
                g=0
            if(classes[class_ids[i]] == "carrot"):
                carrot+=1
                p=0
            if(classes[class_ids[i]] == "cake"):
                cake+=1
                f=0
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
    print("stock available:")
    print("apple: ",apple)
    print("orange: ",orange)
    print("bananna: ",banana)
    print("sandwich: ",sandwich)
    print("carrot: ",carrot)
    print("cake: ",cake)
    if(apple<1 and a == 0):
        sms("apple out of stock")
        a=1
    if(orange<1 and b == 0):
        sms("orange out of stock")
        b=1
    if(carrot<1 and p == 0):
        sms("carrot out of stock")
        p=1
    if(banana<1 and d == 0):
        sms("banana out of stock")
        d=1
    if(sandwich<1 and e == 0):
        sms("sandwich out of stock")
        e=1
    if(cake<1 and f == 0):
        sms("cake out of stock")
        f=1
    if(broccoli<1 and g == 0):
        sms("broccoli out of stock")
        g=1
    if(count == 30):#in every 30 iteration a stock message will be sent 
        data="apple: "+str(apple)+"\n\r"+"orange: "+str(orange)+"\n\r"+"banana"+str(banana)+"\n\r"+"broccoli: "+str(broccoli)+"\n\r"+"carrot: "+str(carrot)+"\n\r"+"cake: "+str(cake)+"\n\r"
        sms(data)
        count = 0
    # show the output frame
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
            break
cv2.destroyAllWindows()
