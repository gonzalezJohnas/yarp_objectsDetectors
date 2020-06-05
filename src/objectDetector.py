import tensorflow as tf

import yarp
import numpy as np
import sys
import os
from utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

IS_RUNNING = True

yarpLog = yarp.Log()


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
        self.input_port = yarp.Port()
        # Create numpy array to receive the image and the YARP image wrapped around it
        self.input_img_array = None
        self.input_yarp_image = None

        # Define vars for outputing image
        self.output_objects_port = yarp.Port()
        self.output_raw_port = yarp.Port()

        self.output_img_port = yarp.Port()
        self.display_buf_image = yarp.ImageRgb()
        self.display_buf_array = None

        self.module_name = None
        self.width_img = None
        self.height_img = None

        self.label_path = None
        self.category_index = None

        self.detection_graph = None
        self.model_path = None

        self.cap = None
        self.session = None

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
        self.input_yarp_image = yarp.ImageRgb()
        self.input_yarp_image.resize(self.width_img, self.height_img)

        self.input_yarp_image.setExternal(self.input_img_array, self.width_img, self.height_img)

        yarpLog.info('Model initialization ')

        if not self._read_label():
            yarpLog.error("Unable to load label file")
            return False

        if not self._load_graph():
            yarpLog.error("Unable to load model")
            return False

        # self.cap = cv2.VideoCapture(0)  # Change only if you have more than one webcams
        yarpLog.info('Module initialization done, Running the model')
        return True

    def _load_graph(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            # Load a (frozen) Tensorflow model into memory.
            od_graph_def = tf.GraphDef()
            try:
                with tf.gfile.GFile(self.model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            except Exception as e:
                print(e)
                return False

        self.session = tf.Session(graph=self.detection_graph)

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        self.session.run(
            [classes],
            feed_dict={image_tensor: np.zeros((1, 320, 240, 3))})
        return True

    def _read_label(self):
        try:
            # Loading label map
            # Label maps map indices to category names,
            # Here we use internal utility functions

            label_map = load_labelmap(self.label_path)
            max_num_classes = max(item.id for item in label_map.item)
            categories = convert_label_map_to_categories(
                label_map, max_num_classes=max_num_classes, use_display_name=True)
            self.category_index = create_category_index(categories)
        except Exception as e:
            print(e)
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
        elif command.get(0).asString() == "process":
            self.process = True if command.get(1).asString() == 'on' else False
            reply.addString("ok")
            ok = True
            rec = True
        elif command.get(0).asString() == "set":
            if command.get(1).asString() == 'thr' and command.get(2).isDouble():
                self.threshold = command.get(2).asDouble() if (command.get(2).asDouble() > 0.0 and command.get(2).asDouble() < 1.0) else self.threshold
                reply.addString("ok")
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
        self.input_port.read(self.input_yarp_image)

        if self.input_yarp_image and self.process:
            frame = np.frombuffer(self.input_img_array, dtype=np.uint8).reshape(
                (self.height_img, self.width_img, 3)).copy()

            image_np_expanded = np.expand_dims(frame, axis=0)

            # # Extract image tensor
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            (boxes, scores, classes, num_detections) = self.session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            if self.output_img_port.getOutputCount():
                # # Visualization of the results of a detection.
                visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8,
                    min_score_thresh=self.threshold)

                self.display_buf_array = frame
                self.write_yarp_image(frame)

            if self.output_objects_port.getOutputCount():
                self.write_objects(classes, boxes, scores)

        return True

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
        for boxe, score, cl in zip(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)):

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
