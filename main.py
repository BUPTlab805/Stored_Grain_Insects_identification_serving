# -*- coding: utf-8 -*-

import sys
import iimport
if len(sys.argv) != 4:
    print "usage : port gpu_id config_path"
    quit()
port = int(sys.argv[1])
gpu_id = sys.argv[2]
config_path = sys.argv[3]
iimport.sys_path_init(config_path)

import os
import cv2
import logging
import tornado.web
from config.config import Config
from protobuff import DATA
from detector_classifier.detector.ssd_detector import SsdDetector
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

class PredictHandler(tornado.web.RequestHandler):
    def get(self):
        pass
    def post(self):
        data = DATA.Data_Struct()
        data.successful = False
        if self.request.files:
            proto_str = self.request.files["proto_data"][0]["body"]
            data.from_pb_to_data(proto_str)
            # cv2.imwrite("./temp_file/request_img.jpg",data.train_single_img)
            if data.img_shape_width==512 and data.img_shape_high==512:
                data.identified_img_data,data.identified_result = detector.predict(data.img_data_BGR)
                data.identified_img_data = cv2.cvtColor(data.identified_img_data, cv2.COLOR_BGR2RGB)
                data.identified_result = str(data.identified_result)
                data.successful = True
            elif data.img_shape_width==2592 and data.img_shape_high==1944:
                data.identified_img_data, data.identified_result = detector.predict_full_pic(data.img_data_BGR)
                data.identified_img_data = cv2.cvtColor(data.identified_img_data, cv2.COLOR_BGR2RGB)
                data.identified_result = str(data.identified_result)
                data.successful = True
        self.set_header('Content-Type', 'application/octet-stream')
        self.set_header('Content-Disposition', 'attachment; filename=proto_response.proto')
        self.write(data.from_data_to_pb())

        self.flush()
        self.finish()



config = Config(config_path)
config.print_config()
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=config.log_path,
                filemode='w')
detector = SsdDetector(config)
application = tornado.web.Application([
    (config.serving_name, PredictHandler),
])


if __name__ == "__main__":
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()




