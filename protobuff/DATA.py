# coding: utf-8
import SGIIS_DATA_PROTO_pb2
import cv2
import numpy as np
'''
message request {
    enum Device_Type {
        long_distance_sticky_board_device = 0;
    }
    required bytes train_single_img = 1;
    required string img_type = 2;
    required int32 img_shape_width = 3;
    required int32 img_shape_high = 4;
    required Device_Type device_type = 5;
    optional float confidence_threshold = 6;
    optional float nms_threshold = 7;
    optional string debug_info = 8;
    optional string other_info = 9;

}

message response {
    required bool successful = 1;
    optional bytes identified_img_data = 2;
    optional string identified_result = 3;
}

'''

class Data_Struct:
    def __init__(self):
        # 过程中使用
        self.img_data_BGR = None

        # REQUEST
        self.img_data = None
        self.img_type = None
        self.img_shape_width = None
        self.img_shape_high = None
        self.device_type = None
        self.confidence_threshold = None
        self.nms_threshold = None
        self.debug_info = None
        self.other_info = None

        #RESPONSE
        self.successful = None
        self.identified_img_data = None
        self.identified_result = None

    def from_pb_to_data(self,pb_data_string):
        re = SGIIS_DATA_PROTO_pb2.request()
        re.ParseFromString(pb_data_string)
        image = np.asarray(bytearray(re.img_data), dtype="uint8")
        self.img_data = cv2.imdecode(image, cv2.IMREAD_COLOR)
        self.img_data_BGR = cv2.cvtColor(self.img_data, cv2.COLOR_RGB2BGR)
        self.img_type = re.img_type
        self.img_shape_width = re.img_shape_width
        self.img_shape_high = re.img_shape_high
        self.device_type = re.device_type

    def from_data_to_pb(self):
        rs = SGIIS_DATA_PROTO_pb2.response()
        img_encode = cv2.imencode('.jpg', self.identified_img_data)[1]
        data_encode = np.array(img_encode)
        str_encode = data_encode.tostring()
        rs.identified_img_data = str_encode
        rs.successful = self.successful
        rs.identified_result = self.identified_result
        str = rs.SerializeToString()
        return str

if __name__ == '__main__':

    request = SGIIS_DATA_PROTO_pb2.request()

    img = cv2.imread('./gg.jpg')
    # '.jpg'表示把当前图片img按照jpg格式编码，按照不同格式编码的结果不一样
    img_encode = cv2.imencode('.jpg', img)[1]
    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()

    request.img_data = str_encode
    request.img_type = "jpg"
    request.img_shape_width = 512
    request.img_shape_high = 512
    request.device_type = 0
    str = request.SerializeToString()

    re = SGIIS_DATA_PROTO_pb2.request()
    re.ParseFromString(str)
    image = np.asarray(bytearray(re.img_data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite("gg_out.jpg",image)
    print re.img_type
    print re.img_shape_width
    print re.img_shape_high
    print re.device_type
