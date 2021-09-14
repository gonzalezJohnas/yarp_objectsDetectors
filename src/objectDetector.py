import tensorflow as tf

import yarp
import numpy as np
import sys
import os
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

IS_RUNNING = True

yarpLog = yarp.Log()


def info(msg):
    print("[INFO] {}".format(msg))


def error(msg):
    print("[ERROR] {}".format(msg))


class ObjectDetectorModule(yarp.RFModule):
    """
    Description:
        Object to read yarp image and localise and recognize objects

    Args:
        input_port  : input port of image
        output_port : output port for streaming recognized names
        display_port: output port for image with recognized objects in bouding box
        raw_output : output the list of <bounding_box, label, probability> detected objects
    """

    def __init__(self):
        yarp.RFModule.__init__(self)

        # handle port for the RFModule
        self.handle_port = yarp.Port()
        self.attach(self.handle_port)

        # Define vars to receive an image
        self.input_port = yarp.BufferedPortImageRgb()

        # Create numpy array to receive the image and the YARP image wrapped around it
        self.input_img_array = None

        # Define vars for outputing image
        self.output_objects_port = yarp.Port()
        self.output_raw_port = yarp.Port()

        self.output_img_port = yarp.Port()
        self.display_buf_array = None
        self.display_buf_image = yarp.ImageRgb()

        self.module_name = None
        self.width_img = None
        self.height_img = None

        self.label_path = None
        self.category_index = None

        self.detection_graph = None
        self.model_path = None

        self.cap = None
        self.model = None

    def configure(self, rf):

        self.module_name = rf.check("name",
                                    yarp.Value("ObjectsDetector"),
                                    "module name (string)").asString()

        self.label_path = rf.check('label_path', yarp.Value(''),
                                   'Path to the label file').asString()

        self.width_img = rf.check('width', yarp.Value(320),
                                  'Width of the input image').asInt()

        self.height_img = rf.check('height', yarp.Value(240),
                                   'Height of the input image').asInt()

        self.model_path = rf.check('model_path', yarp.Value(),
                                   'Path to the model').asString()

        self.threshold = rf.check('threshold', yarp.Value(0.5),
                                  'Theshold detection score').asDouble()

        self.nms_iou_threshold = rf.check('filtering_distance', yarp.Value(0.8),
                                          'Filtering distance in pixels').asDouble()

        self.process = rf.check('process', yarp.Value(True),
                                'enable automatic run').asBool()

        # Create handle port to read message
        self.handle_port.open('/' + self.module_name)

        # Create a port to ouput an image
        self.output_img_port.open('/' + self.module_name + '/image:o')

        # Create a port to output the template image of the recognized face
        self.output_raw_port.open('/' + self.module_name + '/raw:o')

        # Create a port to ouput the recognize names
        self.output_objects_port.open('/' + self.module_name + '/objects:o')

        # Output
        self.display_buf_image.resize(self.width_img, self.height_img)
        self.display_buf_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8).tobytes()
        self.display_buf_image.setExternal(self.display_buf_array, self.width_img, self.height_img)

        # Create a port to receive an image
        self.input_port.open('/' + self.module_name + '/image:i')
        self.input_img_array = np.zeros((self.height_img, self.width_img, 3), dtype=np.uint8).tobytes()

        info('Model initialization ')

        if not self._read_label():
            error("Unable to load label file")
            return False

        if not self._load_graph():
            error("Unable to load model")
            return False

        # self.cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
        info('Module initialization done, Running the model')
        return True

    def _load_graph(self):
        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            error("{}".format(e))
            return False
        return True

    def _read_label(self):
        try:

            self.category_index = load_labelmap(self.label_path)

        except Exception as e:
            print("Error while loading label file {}".format(e))
            return False
        return True

    def interruptModule(self):
        print("stopping the module \n")
        self.input_port.interrupt()
        self.output_img_port.interrupt()
        self.output_raw_port.interrupt()
        self.output_objects_port.interrupt()
        self.handle_port.interrupt()

        return True

    def close(self):
        self.input_port.close()
        self.output_img_port.close()
        self.output_raw_port.close()
        self.output_objects_port.close()
        self.handle_port.close()

        return True

    def respond(self, command, reply):
        ok = False

        # Is the command recognized
        rec = False

        reply.clear()

        if command.get(0).asString() == "quit":
            reply.addString("quitting")
            return False

        elif command.get(0).asString() == "help":
            reply.addString("Object detector module command are:\n")
            reply.addString("set/get thr <double> -> to get/set the detection threshold\n")
            reply.addString("set/get filt <int> -> to get/set the filtering distance for boxes")
            ok = True
            rec = True

        elif command.get(0).asString() == "process":
            self.process = True if command.get(1).asString() == 'on' else False
            reply.addString("ok")
            ok = True
            rec = True

        elif command.get(0).asString() == "set":
            if command.get(1).asString() == 'thr' and command.get(2).isDouble():
                self.threshold = command.get(2).asDouble() if (
                            command.get(2).asDouble() > 0.0 and command.get(2).asDouble() < 1.0) else self.threshold
                reply.addString("ok")

            elif command.get(1).asString() == 'filt' and command.get(2).isDouble():
                self.nms_iou_threshold = command.get(2).asDouble() if (command.get(2).asDouble() > 0.0 and command.get(
                    2).asDouble() < 1.0) else self.nms_iou_threshold

                reply.addString("ok")
            else:
                reply.addString("nack")

        elif command.get(0).asString() == "get":
            if command.get(1).asString() == 'thr':
                reply.addDouble(self.threshold)
            elif command.get(1).asString() == 'filt':
                reply.addDouble(self.nms_iou_threshold)

            else:
                reply.addString("nack")
            ok = True
            rec = True

        return True

    def getPeriod(self):

        """
           Module refresh rate.

           Returns : The period of the module in seconds.
        """
        return 0.05

    def updateModule(self):

        # Read the data from the port into the image
        input_yarp_image = self.input_port.read(False)

        if input_yarp_image and self.process:
            input_yarp_image.setExternal(self.input_img_array, self.width_img, self.height_img)
            frame = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
                (self.height_img, self.width_img, 3)).copy()

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(frame)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]
            detections = self.model(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                          for key, value in detections.items()}
            detections['num_detections'] = num_detections
            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            # print(detections['detection_classes'])

            boxes = detections['detection_boxes']
            scores = detections['detection_scores']

            boxes, scores = self.filter_boxes(boxes, scores)
            boxes, scores = self.non_max_suppression_fast(boxes, scores, overlapThresh=self.nms_iou_threshold)

            if self.output_img_port.getOutputCount():
                # # Visualization of the results of a detection.
                visualize_boxes_and_labels_on_image_array(
                    frame,
                    boxes,
                    detections['detection_classes'],
                    scores,
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2,
                    min_score_thresh=0.1)

                self.display_buf_array = frame
                self.write_yarp_image(frame)

            if self.output_objects_port.getOutputCount():
                self.write_objects(classes, boxes, scores)

        return True

    def format_boxes_coord(self, boxes):
        new_boxes = []
        for box in boxes:
            left, top, right, bottom = get_bouding_box_coordinates(box, (self.width_img, self.height_img))
            new_boxes.append([left, top, right, bottom])

        return np.array(new_boxes)

    def non_max_suppression_fast(self, boxes_norm, scores, probs=None, overlapThresh=0.3):

        boxes = self.format_boxes_coord(boxes_norm)
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return np.array([]), np.array([])

        # if the bounding boxes are integers, convert them to floats -- this
        # is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and grab the indexes to sort
        # (in the case that no probabilities are provided, simply sort on the
        # bottom-left y-coordinate)
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = y2

        # if probabilities are provided, sort on them instead
        if probs is not None:
            idxs = probs

        # sort the indexes
        idxs = np.argsort(idxs)

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value
            # to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of the bounding
            # box and the smallest (x, y) coordinates for the end of the bounding
            # box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have overlap greater
            # than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked
        return boxes_norm[pick], scores[pick]

    def filter_boxes(self, boxes, scores):
        filt_boxes = []
        filt_scores = []
        for i, (boxe, score) in enumerate(zip(boxes, scores)):
            if score > self.threshold:
                filt_boxes.append(boxe)
                filt_scores.append(score)

        return np.array(filt_boxes), np.array(filt_scores)

    def write_objects(self, classes, boxes, scores):
        """
        Stream the recognize objects name and coordiantes on a yarp port
        :param classes: list of classe names recognized
        :param scores: list of score for each recognized object
        :param boxes: list of boxe coordiantes for each recognized objects
        :return:
        """
        list_objects_bottle = yarp.Bottle()
        list_objects_bottle.clear()
        write_bottle = False

        for boxe, score, cl in zip(boxes, scores, np.squeeze(classes)):

            if score > self.threshold:
                left, top, right, bottom = get_bouding_box_coordinates(boxe, (self.width_img, self.height_img))

                class_name = self.category_index[cl]['name']

                yarp_object_bottle = yarp.Bottle()
                yarp_object_bottle.addString(str(class_name))
                yarp_object_bottle.addDouble(float(round(score, 2)))

                yarp_coordinates = yarp.Bottle()
                yarp_coordinates.addDouble(left)
                yarp_coordinates.addDouble(top)
                yarp_coordinates.addDouble(right)
                yarp_coordinates.addDouble(bottom)

                yarp_object_bottle.addList().read(yarp_coordinates)

                list_objects_bottle.addList().read(yarp_object_bottle)
                write_bottle = True

        if write_bottle:
            self.output_objects_port.write(list_objects_bottle)

    def write_yarp_image(self, frame):
        """
        Handle function to stream the recognize faces with their bounding rectangles
        :param img_array:
        :return:
        """
        self.display_buf_image = yarp.ImageRgb()
        self.display_buf_image.resize(self.width_img, self.height_img)
        self.display_buf_image.setExternal(frame.tobytes(), self.width_img, self.height_img)
        self.output_img_port.write(self.display_buf_image)

    def write_template_face(self, template_img):
        """
        Handle function to stream the tempalte of the recognize face
        :param img_array:
        :return:
        """
        self.display_buf_image = yarp.ImageRgb()
        self.display_buf_image.resize(template_img.shape[1], template_img.shape[0])
        self.display_buf_image.setExternal(template_img.tobytes(), template_img.shape[1], template_img.shape[0])
        self.face_template.write(self.display_buf_image)


if __name__ == '__main__':

    # Initialise YARP
    if not yarp.Network.checkNetwork():
        print("Unable to find a yarp server exiting ...")
        sys.exit(1)

    yarp.Network.init()

    objectsDetectorModule = ObjectDetectorModule()

    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext('objectDetector')
    rf.setDefaultConfigFile('objectDetector.ini')

    if rf.configure(sys.argv):
        objectsDetectorModule.runModule(rf)
    sys.exit()