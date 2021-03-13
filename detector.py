import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import ProductSKU as psku
import argparse
from Database import Connection as DB
from datetime import datetime

log_path = psku.Main().cwd + '\\predicted output\\log'
output_image_path = psku.Main().cwd + '\\predicted output\\images'


# test yolo model
def predict_output(test_input):
    d1 = datetime.today().strftime('%Y%m%d')
    dt = datetime.today().strftime('%Y-%m-%d_%H:%M:%S:%f')

    print("Connecting to MongoDB......Please wait.......")
    collection = DB.db_obj['product_master']
    if collection:
        print('Collection is retrieved successfully')
    else:
        print('Error. No collection found from database.')
        return

    print("Initializing YoloV3......Please wait.......")
    # weights = 'base weights'
    weights = 'custom weights\\data augmented\\median filtered\\'
    net = cv2.dnn.readNet(f'{os.getcwd()}\\{weights}\\yolo-obj_best.weights',
                          'F:\\APU\\Modules\\CP\\CP2\\darknet\\build\\darknet\\x64\\cfg\\yolo-obj.cfg')

    # save all the names in file o the list classes
    classes = []
    with open("F:\\APU\\Modules\\CP\\CP2\\darknet\\build\\darknet\\x64\\data\\obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # get layers of the network
    layer_names = net.getLayerNames()

    # Determine the output layer names from the YOLO model

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    print("Yolov3 loaded successfully.")
    print("System is predicting objects.......")
    psku_obj = psku.Main()
    # Capture frame-by-frame
    img = cv2.imread(f'{test_input}')
    img = cv2.resize(img, (psku_obj.resized_height, psku_obj.resize_width))
    height, width, channels = img.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (256, 256),
                                 swapRB=True, crop=False)
    # Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    predicted_objects = ''
    predicted_objects_count = {}
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
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

    # We use NMS function in opencv to perform Non-maximum Suppression
    # we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            conf = '{:.2f}'.format(confidences[i])
            label = str(classes[class_ids[i]])

            if label not in predicted_objects_count:
                predicted_objects_count[label] = 1
            else:
                predicted_objects_count[label] = predicted_objects_count[label] + 1
            print(label + ':' + str(conf))
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, f'{label}: {str(conf)}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        # font size, color, thickness
                        0.5, (255, 0, 0), 2)

    for key in predicted_objects_count.keys():
        doc = list(collection.find({'Product SKU': key}))
        if doc:
            unit_count = predicted_objects_count.get(key)
            price = doc[0].get('Price')
            currency = doc[0].get('Currency')
            manufacturer = doc[0].get('Manufacturer')
            expiry_date = doc[0].get('Expiry Date')
            total_price = round(price * unit_count, 2)

            if predicted_objects:
                predicted_objects += ', '

            predicted_objects += '{' + '"Product" : "{0}", "Quantity" : "{1}", "Per Unit Price" : "{6}{2}", ' \
                                 '"Total Price" : "{6}{3}", "Manufacturer" : "{4}", "Expiry Date" : "{5}"'.format(
                                  key, unit_count, price, total_price, manufacturer, expiry_date, currency) + '}'

    print(predicted_objects)

    # write to output folder
    cv2.imwrite(f'{output_image_path}\\{dt.replace(":", "").replace("-", "")}.jpg', img)
    # end

    # write to a log file
    f = open(f'{log_path}\\{d1.replace("_", "")}.txt', 'a')
    f.write('{0}: test input argument: {1} | predicted objects json: {2}'.format(dt.replace('_', ' '), test_input,
                                                                                 predicted_objects))
    f.write('\n')
    f.close()
    # end

    plt.figure()
    plt.imshow(img[..., ::-1])  # RGB-> BGR
    plt.show()

# unit test cases
# predict_output('F:\\APU\Modules\\CP\\CP2\\Object Detection\\Product SKU\\NATUREL PURE OLIVE OIL 750 ML'
#                    '\\test\\TT116.jpg')
# predict_output('F:\\APU\Modules\\CP\\CP2\\Object Detection\\Product SKU\\Finalized Images\\KLEENEX ULTRA SOFT BATH '
#                'ISSUE MEGA\\unfiltered\\test\\A607.jpg')
# predict_output('F:\\APU\Modules\\CP\\CP2\\Object Detection\\Product SKU\\extra testing photos'
#                        '\\20210207_133453.jpg')
# predict_output('F:\\APU\Modules\\CP\\CP2\\Object Detection\\Product SKU\\extra testing photos'
#                        '\\20210207_134237.jpg')
# predict_output('F:\\APU\Modules\\CP\\CP2\\Object Detection\\Product SKU\\extra testing photos'
#                        '\\kleenex.jpg')
# end


parser = argparse.ArgumentParser()
parser.add_argument('--testdata', '-t', help='test data path')
args = parser.parse_args()

if args.testdata:
    predict_output(args.testdata)