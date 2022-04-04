# coding=utf-8

import socket
import sys
import threading
import struct
import time
import warnings

import cv2
import numpy as np
import network.socket_msg
import codecs


class TcpNode:
    def __init__(self, ip, port):
        """"""
        '''socket setup'''
        self._SEND_BUF_SIZE = 256256
        self._RECV_BUF_SIZE = 256256

        '''tele setup'''
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._SEND_BUF_SIZE)  # set a new buffer size
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._RECV_BUF_SIZE)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        '''init sock'''
        self._sock_clients = []
        server_address = (ip, port)
        print("targeting ip %s, port %s" % server_address)

        '''status init'''
        self.is_connected = False
        self._msg_len_2_receive = 256

        '''output pool'''
        self.msg_class = Msg
        self._msgs = []

        '''lock'''
        self.lock = threading.Lock()

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< setter'''

    def end(self):
        self.is_connected = False
        try:
            self._socket.close()
            self._socket.detach()
        except socket.error:
            print("fail to close on port ")
            sys.exit(1)

    '''setter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< getter'''

    def peek_msg(self):
        msgs = self._msgs
        return msgs

    '''getter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< threading implementation'''

    def _listen_looping(self):
        """"""
        '''receive msg'''
        print("start listening")
        while True:
            if not self.is_connected:
                break
            self.listen()
        return True

    def _speak_looping(self):
        """"""
        '''receive msg'''
        print("start speaking")
        while True:
            if not self.is_connected:
                break
            # self.lock.acquire()  # lock self._msg here
            msgs = self.pop_msgs()
            # self.lock.release()  # unlock self._msg here
            self.speak(msgs)
        return True

    ''' threading implementation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    def reset(self):
        self.is_connected = False
        try:
            self._socket.close()
        except socket.error:
            print("fail to close on port ")
            sys.exit(1)
        self._msgs.clear()
        self._sock_clients.clear()

    def heard_something(self):
        return len(self._msgs) > 0

    def status(self):
        return 'config ' + str(self._socket.getsockname()) + \
               ' connecting: ' + str(self.is_connected) + \
               ' msg received: ' + str(self._msgs)

    ''' status getter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< utils'''
    def speak(self, msgs):
        target = self._sock_clients[0]
        if not isinstance(msgs, list) and not isinstance(msgs, tuple):
            msgs = [msgs]
        for msg in msgs:
            '''cast to Msg class'''
            if not isinstance(msg, self.msg_class):
                msgs_bytes = self.msg_class().encoding_data(msg)
            else:
                msgs_bytes = msg.encoding_content()  # make anything into transportable

            if msgs_bytes is not None:
                self.lock.acquire()

                if not isinstance(msgs_bytes, list) and not isinstance(msgs_bytes, tuple):
                    msgs_bytes = [msgs_bytes]
                for msg_bytes_ in msgs_bytes:
                    target.send(msg_bytes_)

                self.lock.release()
        return True

    def listen(self):
        """"""
        '''receive msg info'''
        msg = self.msg_class()
        bytes_received = self._keep_recv(msg.len_msg_pre_info_2_receive)
        if len(bytes_received) > 0:
            msg.set_msg(bytes_received)
            '''record msg'''
            self.lock.acquire()  # lock self._msg here
            self._msgs.append(msg)
            self.lock.release()  # unlock self._msg here

    def pop_msgs(self):
        self.lock.acquire()  # lock self._msg here
        msgs = self._msgs
        self._msgs = []
        self.lock.release()  # unlock self._msg here
        return msgs

    def add_msgs(self, msgs):
        if not isinstance(msgs, list) and not isinstance(msgs, tuple):
            msgs = [msgs]
        for msg in msgs:
            assert isinstance(msg, self.msg_class), "Expect " + str(self.msg_class) + ' get ' + str(msg.__class__) + ' insteaded'
        self.lock.acquire()
        self._msgs += msgs
        self.lock.release()

    def _keep_recv(self, length):
        time_0 = time.time()
        num_loop = 10000
        client = self._sock_clients[0]
        bytes_received = b''
        length_left = length
        for i in range(num_loop):
            bytes_receiving = client.recv(length_left)
            if len(bytes_receiving) == 0 or len(bytes_received) >= length:
                break
            bytes_received += bytes_receiving
            length_left = length - len(bytes_received)

        # warnings.warn(
        #     'cannot receive complete bytes from buffer,' + ' loops, expect ' + str(length) + ' bytes, but got ' + str(
        #         len(bytes_received)) + ' after ' + str(num_loop) + ' recv calling')
        # print('_keep_recv takes', time.time() - time_0)
        return bytes_received

    ''' utils >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


class TcpServer(TcpNode):
    def __init__(self, ip, port):
        """"""
        '''socket setup'''
        super().__init__(ip, port)
        server_address = (ip, port)
        print("starting listen on ip %s, port %s" % server_address)
        self._socket.bind(server_address)  # bind port

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< setter'''

    def stand_by(self):
        try:
            self._socket.listen(1)
        except socket.error:
            print("fail to listen on port ")
            sys.exit(1)

        '''wait connection'''
        while True:
            print("waiting for connection at")
            client, address = self._socket.accept()
            self.lock.acquire()
            self.is_connected = True
            self._sock_clients.append(client)
            self.lock.release()
            print("having a connection at", address)
            break
        return client

    '''setter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< status getter '''
    ''' status getter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< utils'''
    ''' utils >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


class TcpClient(TcpNode):
    def __init__(self, ip, port):
        """"""
        super().__init__(ip, port)
        '''socket setup'''
        self._SEND_BUF_SIZE = 256
        self._RECV_BUF_SIZE = 256

        '''tele setup'''
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._SEND_BUF_SIZE)  # set a new buffer size
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._RECV_BUF_SIZE)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        '''init sock'''
        self._sock_clients = []
        server_address = (ip, port)
        self._socket.connect(server_address)
        self._sock_clients.append(self._socket)
        self.is_connected = True


class TcpServerCommandOnly(TcpServer):
    def __init__(self, ip, port):
        super(TcpServerCommandOnly, self).__init__(ip, port)
        self._receive_len = 10
        self._msg_unit_len = 2

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< setter'''

    def start_thread_listener_and_speaker_for_vision_sensor(self):
        """"""
        self.stand_by()
        t_listener = threading.Thread(target=self._listen_looping)  # start a thread to listen
        t_listener.start()
        t_speaker = threading.Thread(target=self._speak_looping)  # start a thread to listen
        t_speaker.start()
        return True

    '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> setter'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< threading implementation '''

    # def _listen_looping(self):
    #     """"""
    #     '''receive msg'''
    #     print("start listening")
    #
    #     msg_left_over = ''
    #     while True:
    #         if not self.is_connected:
    #             break
    #         client = self._sock_clients[0]
    #         msg = client.recv(self._receive_len)
    #         msg_des_string = msg_left_over + msg.decode('utf-8')
    #         num_msg = len(msg_des_string) // self._msg_unit_len
    #         msg_left_over = msg_des_string[num_msg * self._msg_unit_len:]
    #
    #         self.lock.acquire()  # lock self._msg here
    #         for i_msg_unit in range(num_msg):
    #             self._msgs.append(msg_des_string[i_msg_unit * self._msg_unit_len:(i_msg_unit + 1) * self._msg_unit_len])
    #         self.lock.release()  # unlock self._msg here
    #     return True

    ''' threading implementation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


class Msg:
    len_msg_pre_info_2_receive = 256

    def __init__(self):
        """"""
        '''msg'''
        self.content = None

    def set_msg(self, msg_bytes):
        if len(msg_bytes) > 0:
            self.content = msg_bytes

    def encoding_content(self):
        """
        cast msg into transportable obj or list of transportable obj
        :return:
        """
        data_in_server = self.content
        return self.encoding_data(data_in_server)

    def encoding_data(self, data_node):
        """
        cast msg into transportable obj or list of transportable obj
        :param data_node: msg, string
        :return:
        """
        if data_node is None:
            data_node = ''
        elif isinstance(data_node, str):
            data_node = data_node.encode('utf-8')
        elif isinstance(data_node, int):
            data_node = struct.pack('!I', data_node)
        elif isinstance(data_node, np.ndarray):
            data_node = data_node.tobytes()
        elif isinstance(data_node, bytes):
            data_node = data_node
        else:
            raise TypeError('Unexpect type to decode: ' + str(type(data_node)) + ' : ' + str(data_node))
        return data_node


class MsgGeneral(Msg):
    """"""
    '''msg pre info define'''
    # msg_len + msg_type + msg_id + msg_command + msg_data
    msg_len_len = 4
    msg_type_len = 4
    msg_id_len = 20
    msg_command_len = 4
    msg_data_len = 0
    len_msg_pre_info_2_receive = msg_len_len + msg_type_len + msg_id_len + msg_command_len

    def __init__(self):
        """"""
        super().__init__()
        '''msg'''
        self.content = None

    def set_pre_info_from_bytes(self, bytes_pre_info):
        msg_data_len = 0
        if len(bytes_pre_info) > 0:
            # msg_pre_info_str = msg_pre_info.decode('utf-8')

            #    int           int           str         str
            msg_data_len, msg_data_type, msg_data_id, msg_command = self._decoding_msg_prefix(bytes_pre_info)
            content = {'type': msg_data_type,
                       'id': msg_data_id,
                       'command': msg_command}
            self.content = content
        return msg_data_len

    def set_data_from_bytes(self, bytes_data):
        """receive msg data"""
        if len(bytes_data) > 0:

            # msg_data_array = self._decoding_data(bytes_data)

            msg_data_array = bytes_data
            self.content['data'] = msg_data_array

    def _decoding_msg_prefix(self, data_from_buffer):
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
            msg_command = data_from_buffer[self.msg_len_len + self.msg_type_len + self.msg_id_len:]

            msg_data_len = struct.unpack("!I", msg_data_len)[0]
            msg_data_type = struct.unpack("!I", msg_data_type)[0]

            msg_data_id.decode("UTF-8", "ignore")
            msg_data_id = codecs.decode(msg_data_id, 'UTF-8')
            msg_command = codecs.decode(msg_command, 'UTF-8')

            # msg_data_id = msg_data_id.decode('utf-8')
            # msg_command = msg_command.decode('utf-8')

            #           int           int           str          str
            return msg_data_len, msg_data_type, msg_data_id, msg_command  # type, len, id, data

    def _decoding_data(self, data_from_buffer):
        # img_size = (493, 853, 3)
        # img_size = (2064, 3088, 3)
        # img_array = np.frombuffer(data_from_buffer, dtype=np.uint8)
        # assert len(img_array) == np.prod(img_size), str(len(img_array)) + ' != ' + str(np.prod(img_size))
        # img = img_array.reshape(img_size)

        img_array = np.frombuffer(data_from_buffer, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    def encoding_content(self):
        """
        cast msg into transportable obj or list of transportable obj
        :param data_2_buffer: msg obj, dict for sure
        :return:
        """
        data_2_buffer = self.content
        return self.encoding_data(data_2_buffer)

    # @staticmethod
    def encoding_data(self, data_node):
        if not isinstance(data_node, dict):
            return super().encoding_data(data_node)
        elif isinstance(data_node, dict):
            msg_len = 0
            msg_type = data_node['type']
            msg_id = data_node.get('id', '0' * self.msg_id_len)
            msg_command = data_node['command']
            msg_data = data_node['data']

            msg_type = super().encoding_data(msg_type)
            msg_id = super().encoding_data(msg_id)
            msg_command = super().encoding_data(msg_command)
            msg_data = super().encoding_data(msg_data)

            if msg_data is not None:
                msg_len = len(msg_data)
            msg_len = super().encoding_data(msg_len)

            if len(msg_len) > 0 and len(msg_type) > 0 and len(msg_id) and len(msg_command) and len(msg_data):
                return msg_len, msg_type, msg_id, msg_command, msg_data  # type, len, id, data
            else:
                return None
        else:
            raise TypeError('Unexpect type to decode: ' + str(type(data_node)))

    def __getitem__(self, item: str):
        return self.content[item]

    def get_msg(self):
        return self.content


class TcpServerGeneral(TcpServer):
    def __init__(self, ip, port):
        super().__init__(ip, port)

        self.msg_class = MsgGeneral
        assert self.msg_class.len_msg_pre_info_2_receive == 32

        self._msg_2_command_mapping = {
            '01': ''
        }
        self._msg_2_datatype_mapping = {
            '01': 'img'
        }

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< setter'''

    def start_thread_listener_and_speaker_with_vision_sensor(self):
        """"""
        self.stand_by()
        t_listener = threading.Thread(target=self._listen_looping)  # start a thread to listen
        t_listener.start()
        t_speaker = threading.Thread(target=self._speak_looping)  # start a thread to listen
        t_speaker.start()
        return True

    '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> setter'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< getter'''
    '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> getter'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< threading implementation'''

    def listen(self):
        """"""
        '''receive msg info'''
        msg = self.msg_class()
        time_0 = time.time()

        bytes_pre_info_received = self._keep_recv(msg.len_msg_pre_info_2_receive)
        if len(bytes_pre_info_received) > 0:
            msg_data_len = msg.set_pre_info_from_bytes(bytes_pre_info_received)

            print('ask data length of ', msg_data_len)

            bytes_data_received = self._keep_recv(msg_data_len)

            print('receive data length of ', len(bytes_data_received))

            msg.set_data_from_bytes(bytes_data_received)
            if msg.content is not None:
                '''record msg'''
                self.lock.acquire()  # lock self._msg here
                self._msgs.append(msg)
                self.lock.release()  # unlock self._msg here
            print('listen a msg takes', time.time() - time_0)

    ''' threading implementation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< utils'''
    ''' utils >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


# class TcpClientGeneral(TcpClient):
#     def __init__(self, ip, port):
#         """"""
#         super().__init__(ip, port)
#         self.msg_class = MsgGeneral
#
#     def start_thread_listener_and_speaker_with_vision_sensor(self):
#         """"""
#         t_listener = threading.Thread(target=self._listen_looping)  # start a thread to listen
#         t_listener.start()
#         t_speaker = threading.Thread(target=self._speak_looping)  # start a thread to listen
#         t_speaker.start()
#         return True
#
#     def listen(self):
#         """"""
#         msg = self.msg_class()
#
#         '''receive msg pre info'''
#         bytes_pre_info_received = self._keep_recv(msg.len_msg_pre_info_2_receive)
#         msg_data_len = msg.set_pre_info_from_bytes(bytes_pre_info_received)
#
#         '''receive msg data info'''
#         bytes_data_received = self._keep_recv(msg_data_len)
#         msg.set_data_from_bytes(bytes_data_received)
#
#         if msg.content is not None:
#             '''record msg'''
#             self.lock.acquire()  # lock self._msg here
#             self._msgs.append(msg)
#             self.lock.release()  # unlock self._msg here


def test_command():
    listener = TcpServer('127.0.0.1', 6000)
    listener.stand_by()
    print(listener.status())

    count = 0
    while True:
        if count > 2:
            break
        if listener.heard_something():
            print('before pop', listener.status())
            msg = listener.pop_msgs()
            print(msg)
            print('after pop', listener.status())
            count += 1
    listener.end()

    listener = TcpServerCommandOnly('127.0.0.1', 6000)
    listener.start_thread_listener_and_speaker_for_vision_sensor()
    print(listener.status())

    count = 0
    while True:
        if count > 2:
            break
        if listener.heard_something():
            print('before pop', listener.status())
            msg = listener.pop_msgs()
            print(msg)
            print('after pop', listener.status())
            count += 1
    listener.end()


def test_data_com():
    node = TcpServerGeneral('127.0.0.1', 6000)
    node.stand_by()
    print(node.status())

    img_filepath = r'../data/20210902153900.png'
    img = cv2.imread(img_filepath)
    msgs = [{
        'type': 1,
        'data': img,
        'id': '0' * 20,
        'command': '0000',
    }]
    node.speak(msgs)
    # node.speak(img)


def test_data_com_thread():
    node = TcpServerGeneral('127.0.0.1', 6000)

    print(node.status())
    node.start_thread_listener_and_speaker_with_vision_sensor()

    while True:
        if len(node.peek_msg()) > 0:
            print(node.status())

            '''receive'''
            msgs = node.peek_msg()
            for msg in msgs:
                img = msg.content['data']
                print(img.shape)
                cv2.imshow('data', img)
                cv2.waitKey(0)

            '''send back'''
            # for msg in msgs:
            node.add_msgs(msgs)

    # node.speak(msgs)
    # node.speak(img)


def main():
    test_data_com_thread()


if __name__ == '__main__':
    main()
