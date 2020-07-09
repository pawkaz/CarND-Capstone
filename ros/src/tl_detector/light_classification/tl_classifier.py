from styx_msgs.msg import TrafficLight
import keras
from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf
import os

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.model = load_model(os.path.join(base_path, "tf_classifier.h5"))
        self.model._make_predict_function()
        global graph
        graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img = cv2.resize(image, (64, 64))

        img = img[..., ::-1].astype(np.float32) / 255.
        global graph                                  # This is a workaround for asynchronous execution
        with graph.as_default():
            result = self.model.predict(img[np.newaxis, ...])[0][0]

        return TrafficLight.GREEN if result < .6 else TrafficLight.RED
