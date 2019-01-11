import sys
import ConfigParser
# def sys_path_init(config_path):
def sys_path_init(config_path):
    config = ConfigParser.ConfigParser()
    config.read(config_path)
    root = config.get("data", "root_path")
    sys.path.append(root + "detector_classifier/detector/")
    sys.path.append(root + "detector_classifier/detector/datasets/")
    sys.path.append(root + "detector_classifier/detector/nets/")
    sys.path.append(root + "detector_classifier/detector/preprocessing/")
    sys.path.append(root + "detector_classifier/detector/tf_extended/")
    sys.path.append(root + "protobuff/")
    sys.path.append(root + "./config/")

