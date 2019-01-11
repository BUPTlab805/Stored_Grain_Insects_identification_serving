import ConfigParser
import os
'''
root_path = ""
serving_name = "stick_board_identify"
log_path = "log/"
detector_model_path = "model/detector/model.ckpt-928108"
classifier_model_path = "model/"
nms_threshold = 0.3
confidence_threshold = 0.9
'''
class Config:
    def __init__(self,path):

        config = ConfigParser.ConfigParser()
        config.read(path)
        self.root_path = config.get("data", "root_path")
        self.serving_name = config.get("data", "serving_name")
        self.log_path = self.root_path + config.get("data", "log_path")
        self.detector_model_path = self.root_path + config.get("data", "detector_model_path")

    def print_config(self):
        print "--------------------------------------------------------------"
        print "root_path = {}".format(self.root_path)
        print "serving_name = {}".format(self.serving_name)
        print "log_path = {}".format(self.log_path)
        print "detector_model_path = {}".format(self.detector_model_path)

        print "--------------------------------------------------------------"