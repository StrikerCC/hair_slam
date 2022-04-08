import struct
import time
import numpy as np
import codecs
import cv2

"""signal to functionality mapping"""

'''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< network command >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

'''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< network data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


# msg_type_id_2_data_type = {
#     '1': 'img',
#     '2': '1d_array',
#     '3': '2d_array',
#     '4': '3d_array',
#     '5': 'str',
# }
#
# data_type_2_msg_type_id = {}
# for key in msg_type_id_2_data_type.keys():
#     value = msg_type_id_2_data_type[key]
#     data_type_2_msg_type_id[value] = key


def encoding_1d_array(arr):
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 1, str(type(arr)) + ' shape is ' + str(arr.shape)
    arr_str = ''
    # np.array2string(arr)
    for ele in arr:
        arr_str += str(ele) + '/'
    arr_str = arr_str[:-1]
    arr_str = arr_str.encode('utf-8')
    return arr_str


def encoding_2d_array(arr):
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 2
    arr_str = b''
    for row in arr:
        row_str = encoding_1d_array(row)
        arr_str += row_str + b':'
    arr_str = arr_str[:-1]
    return arr_str


def encoding_3d_array(arr):
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 3
    arr_str = b''
    for row in arr:
        row_str = encoding_2d_array(row)
        arr_str += row_str + b','
    arr_str = arr_str[:-1]
    return arr_str


def decoding_1d_array(arr_bytes):
    assert isinstance(arr_bytes, bytes), type(arr_bytes)
    # arr_str = arr_str.decode('utf-8')
    arr = [float(ele_str) for ele_str in arr_bytes.split(b'/') if len(ele_str) > 0]
    arr = np.array(arr)
    return arr


def decoding_2d_array(arr_bytes):
    assert isinstance(arr_bytes, bytes)
    arr = [decoding_1d_array(ele_str) for ele_str in arr_bytes.split(b':') if len(ele_str) > 0]
    arr = np.asarray(arr)
    return arr


def decoding_3d_array(arr_bytes):
    assert isinstance(arr_bytes, bytes)
    arr = [decoding_2d_array(ele_str) for ele_str in arr_bytes.split(b',') if len(ele_str) > 0]
    arr = np.asarray(arr)
    return arr


'''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< local file >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


# file_path_dic = {
#     'Path_Picture_Match_Left':      "D:/Algorithm/Pic/matchLeft.jpg",
#     'Path_Picture_Match_Right':     "D:/Algorithm/Pic/matchRight.jpg",
#
#     'Path_Picture_Track_Left':      "D:/Algorithm/Pic/trackLeft.jpg",
#     'Path_Picture_Track_Right':     "D:/Algorithm/Pic/trackRight.jpg",
#
#     'Path_Xml_Match_Left':          "D:/Algorithm/Xml/Match/matchLeft.xml",
#     'Path_Xml_Match_Right':         "D:/Algorithm/Xml/Match/matchRight.xml",
#     'Path_Xml_Match_Id':            "D:/Algorithm/Xml/Match/matchId.xml",
#
#     'Path_Xml_Track_Left':          "D:/Algorithm/Xml/Track/trackLeft.xml",
#     'Path_Xml_Tracked_Left':        "D:/Algorithm/Xml/Track/trackedLeft.xml",
#
#     'Path_Xml_Track_Right':         "D:/Algorithm/Xml/Track/trackRight.xml",
#     'Path_Xml_Tracked_Right':       "D:/Algorithm/Xml/Track/trackedRight.xml",
#
#     'Path_Xml_Mask_Push_Left':      "D:/Algorithm/Xml/Mask/maskLeft.xml",
#     'Path_Xml_Mask_Tracked_Left':   "D:/Algorithm/Xml/Mask/maskTrackedLeft.xml",
#
#     'Path_Xml_Mask_Push_Right':     "D:/Algorithm/Xml/Mask/maskRight.xml",
#     'Path_Xml_Mask_Tracked_Right':  "D:/Algorithm/Xml/Mask/maskTrackedRight.xml",
# }

# xml_coordinates_decimal_precision = 3


class Msg:
    len_msg_pre_info_2_receive = 256

    def __init__(self):
        """"""
        '''msg'''
        self.data_ = None

    def set_msg(self, msg_bytes):
        if len(msg_bytes) > 0:
            self.data_ = msg_bytes

    def encoding_msg(self):
        """
        cast msg into transportable obj or list of transportable obj
        :return:
        """
        data = self.data_
        return self.encoding_msg_implement(data)

    def encoding_msg_implement(self, data):
        """
        cast msg into transportable obj or list of transportable obj
        :param data: msg, string
        :return:
        """
        if data is None:
            data = ''
        elif isinstance(data, str):
            data = data.encode('utf-8')
        elif isinstance(data, int):
            data = struct.pack('!I', data)
        elif isinstance(data, np.ndarray):
            data = data.tobytes()
        elif isinstance(data, bytes):
            data = data
        else:
            raise TypeError('Unexpect type to decode: ' + str(type(data)) + ' : ' + str(data))
        return data


class MsgGeneral(Msg):
    """"""
    '''msg pre info define'''
    # msg_len + msg_type + msg_id + msg_command + msg_data
    msg_len_len = 4
    msg_type_len = 4
    msg_id_len = 20
    msg_command_len = 2
    msg_data_len = 0
    len_msg_pre_info_2_receive = msg_len_len + msg_type_len + msg_id_len + msg_command_len

    def __init__(self):
        """"""
        super().__init__()
        '''data type mapping'''
        self.msg_type_id_2_data_type = {
            1: 'img',
            2: 'str',
            3: '1d_array',
            4: '2d_array',
            5: '3d_array',
        }
        self.data_type_2_msg_type_id = {}
        for key in self.msg_type_id_2_data_type.keys():
            value = self.msg_type_id_2_data_type[key]
            self.data_type_2_msg_type_id[value] = key

        '''msg command mapping'''
        self.msg_command_2_requests_dic = {
            "00": 'Match_Start_Image_Xml',
            '01': "Match_End_Xml",

            "20": 'Track_Start_New_Image_Xml_Left',
            "21": "Track_Start_Image_Left",
            "22": 'Track_kEnd_Xml_Left',
            "23": 'Push_Mask_Xml_Left',
            "24": 'Peek_Mask_Xml_Left',

            "30": 'Track_Start_New_Image_Xml_Right',
            "31": 'Track_Start_Image_Right',
            "32": 'Track_End_Xml_Right',
            "33": 'Push_Mask_Xml_Right',
            "34": 'Peek_Mask_Xml_Right',

            # "40": 'Mask_Start_Image_Xml_Left',
            "41": 'Mask_End_Left',
            # "42": 'Mask_Request_Left',
            #
            # "50": 'Mask_Start_Image_Xml_Right',
            "51": 'Mask_End_Right',
            # "52": 'Mask_Request_Right'
        }

        self.requests_2_msg_command_dic = {}
        for key in self.msg_command_2_requests_dic.keys():
            value = self.msg_command_2_requests_dic[key]
            self.requests_2_msg_command_dic[value] = key

        '''msg content dict'''
        self.prefix_dict_ = {}

    def set_pre_info_from_bytes(self, bytes_pre_info):
        msg_data_len = 0
        if len(bytes_pre_info) > 0:
            # msg_pre_info_str = msg_pre_info.decode('utf-8')

            #    int           int           str         str
            msg_data_len, msg_data_type, msg_data_id, msg_command = self.__decoding_msg_prefix(bytes_pre_info)
            prefix_dict = {'type': msg_data_type,
                           'id': msg_data_id,
                           'command': msg_command}
            self.prefix_dict_ = prefix_dict
        return msg_data_len

    def set_data_from_bytes(self, bytes_data):
        """receive msg data"""
        if len(bytes_data) > 0:
            '''
            no decoding 
            '''

            msg_data_array = self.__decoding_msg_data(bytes_data)

            '''
            no decoding 
            '''

            # msg_data_array = bytes_data
            self.data_ = msg_data_array

    def __decoding_msg_prefix(self, data_from_buffer):
        """
        cast msg into server readable objs
        :param data_from_buffer: bytes
        :return:
        """
        if not isinstance(data_from_buffer, bytes):
            raise TypeError('Expect bytes, but get ' + str(type(data_from_buffer)))
        elif len(data_from_buffer) != self.len_msg_pre_info_2_receive:
            raise ValueError('Expect 32 bytes for info, but get len of ' + str(len(data_from_buffer)))
        else:
            msg_data_len = data_from_buffer[:self.msg_len_len]
            msg_data_type = data_from_buffer[self.msg_len_len: self.msg_len_len + self.msg_type_len]
            msg_data_id = data_from_buffer[
                          self.msg_len_len + self.msg_type_len: self.msg_len_len + self.msg_type_len + self.msg_id_len]
            msg_command_bytes = data_from_buffer[self.msg_len_len + self.msg_type_len + self.msg_id_len:]

            msg_data_len = struct.unpack("!I", msg_data_len)[0]

            msg_data_type = struct.unpack("!I", msg_data_type)[0]
            msg_data_type = self.msg_type_id_2_data_type[msg_data_type]

            # msg_data_id.decode("UTF-8", "ignore")
            # msg_data_id = codecs.decode(msg_data_id, 'UTF-8')
            # msg_command = codecs.decode(msg_command, 'UTF-8')

            msg_data_id = msg_data_id.decode('utf-8')

            msg_command = msg_command_bytes.decode('utf-8')
            # msg_command = struct.unpack("!I", msg_command)[0]
            # msg_command = str(msg_command)
            msg_command = self.msg_command_2_requests_dic[msg_command]

            #           int           int           str          str
            return msg_data_len, msg_data_type, msg_data_id, msg_command  # type, len, id, data

    def __decoding_msg_data(self, data_bytes):
        assert isinstance(data_bytes, bytes)
        assert 'type' in self.prefix_dict_.keys()
        assert self.prefix_dict_['type'] in self.data_type_2_msg_type_id.keys()

        data = None
        if self.prefix_dict_['type'] == 'img':
            # img_size = (2064, 3088, 3)
            # img_array = np.frombuffer(data_from_buffer, dtype=np.uint8)
            # assert len(img_array) == np.prod(img_size), str(len(img_array)) + ' != ' + str(np.prod(img_size))
            # img = img_array.reshape(img_size)

            # img_array = np.frombuffer(data_bytes, dtype=np.uint8)
            # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            data = data_bytes
        elif self.prefix_dict_['type'] == '1d_array':
            data = decoding_1d_array(data_bytes)
        elif self.prefix_dict_['type'] == '2d_array':
            data = decoding_2d_array(data_bytes)
        elif self.prefix_dict_['type'] == '3d_array':
            data = decoding_3d_array(data_bytes)
        elif self.prefix_dict_['type'] == 'str':
            data = data_bytes.decode('utf-8')
        else:
            raise NotImplementedError(str(self.prefix_dict_['type']) + ' is not implemented yet')
        return data

    def encoding_msg(self):
        """
        cast msg into transportable obj or list of transportable obj
        :param data_2_buffer: msg obj, dict for sure
        :return:
        """
        data = self.data_
        return self.encoding_msg_implement(data)

    def encoding_msg_implement(self, data):
        if not isinstance(self.prefix_dict_, dict):
            return super().encoding_msg_implement(data)
        elif isinstance(self.prefix_dict_, dict):
            msg_len = 0
            msg_type = self.prefix_dict_['type']
            msg_id = self.prefix_dict_.get('id', '0' * self.msg_id_len)
            msg_command = self.prefix_dict_['command']

            if self.prefix_dict_['type'] == 'img':
                data_bytes = super().encoding_msg_implement(data)
            elif self.prefix_dict_['type'] == '1d_array':
                data_bytes = encoding_1d_array(data)
            elif self.prefix_dict_['type'] == '2d_array':
                data_bytes = encoding_2d_array(data)
            elif self.prefix_dict_['type'] == '3d_array':
                data_bytes = encoding_3d_array(data)
            elif self.prefix_dict_['type'] == 'str':
                data = str(data)
                data_bytes = data.encode('utf-8')
            else:
                raise NotImplementedError(str(self.prefix_dict_['type']) + ' is not implemented yet')

            msg_type = self.data_type_2_msg_type_id[msg_type]
            msg_id = msg_id
            msg_command = self.requests_2_msg_command_dic[msg_command]

            msg_type = super().encoding_msg_implement(msg_type)
            msg_id = super().encoding_msg_implement(msg_id)
            msg_command = super().encoding_msg_implement(msg_command)

            '''encoding accordingly'''
            if data_bytes is not None:
                msg_len = len(data_bytes)
            msg_len = super().encoding_msg_implement(msg_len)

            if len(msg_len) > 0 and len(msg_type) > 0 and len(msg_id) and len(msg_command) and len(data_bytes):
                return msg_len, msg_type, msg_id, msg_command, data_bytes  # type, len, id, data
            else:
                return None
        else:
            raise TypeError('Unexpect type to decode: ' + str(type(self.prefix_dict_)))

    def __getitem__(self, item: str):
        return self.prefix_dict_[item]

    def get_msg(self):
        return self.prefix_dict_


def test_arr_encoding_decoding():
    """"""
    '''1d'''
    arr_to_be_encoding = np.arange(0, 10).astype(float)
    arr_str = encoding_1d_array(arr_to_be_encoding)
    print(type(arr_str))
    print(arr_str)

    arr = decoding_1d_array(arr_str)
    print(type(arr))
    print(arr)
    print('encoding_decoding correct', np.allclose(arr_to_be_encoding, arr))

    '''2d'''
    arr_to_be_encoding = np.arange(0, 10).astype(float).reshape(-1, 2)
    arr_str = encoding_2d_array(arr_to_be_encoding)
    print(type(arr_str))
    print(arr_str)

    arr = decoding_2d_array(arr_str)
    print(type(arr))
    print(arr)
    print('encoding_decoding correct', np.allclose(arr_to_be_encoding, arr))

    '''3d'''
    arr_to_be_encoding = np.arange(1, 13).astype(int).reshape((2, 2, 3))
    arr_str = encoding_3d_array(arr_to_be_encoding)
    print(type(arr_str))
    print(arr_str)

    arr = decoding_3d_array(arr_str)
    print(type(arr))
    print(arr)
    print('encoding_decoding correct', np.allclose(arr_to_be_encoding, arr))


def main():
    test_arr_encoding_decoding()


if __name__ == '__main__':
    main()
