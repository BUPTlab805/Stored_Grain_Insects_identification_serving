# -*- coding: utf-8 -*-
import sys
sys.path.append("../")
import requests
import cv2
import numpy as np
import SGIIS_DATA_PROTO_pb2

payload = None
# 打包数据
request = SGIIS_DATA_PROTO_pb2.request()
img = cv2.imread('./full_pic.jpg')
img_encode = cv2.imencode('.jpg', img)[1]
data_encode = np.array(img_encode)
str_encode = data_encode.tostring()
request.img_data = str_encode
request.img_type = "jpg"
request.img_shape_width = 2592
request.img_shape_high = 1944
request.device_type = 0
str = request.SerializeToString()
files = {"proto_data":str}
r = requests.post("http://localhost:8008/stick_board_identify", data=payload, files=files)


rs = SGIIS_DATA_PROTO_pb2.response()
rs.ParseFromString(r.content)
if rs.successful:
    image = np.asarray(bytearray(rs.identified_img_data), dtype="uint8")
    img_data = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite("response.jpg",img_data)
    print rs.identified_result
