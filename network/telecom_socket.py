# coding=utf-8

import socket
import sys
import threading
import time
import warnings

import cv2
import numpy as np
import network.socket_msg
from network.socket_msg import Msg, MsgGeneral


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
        print("TcpNode targeting ip %s, port %s" % server_address)

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
                msgs_bytes = self.msg_class().encoding_msg_implement(msg)
            else:
                msgs_bytes = msg.encoding_msg()  # make anything into transportable

            if msgs_bytes is not None:
                self.lock.acquire()

                print(msg.data_)
                print(*msgs_bytes)
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
        print("TcpServer starting listen on ip %s, port %s" % server_address)
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


class TcpNodeGeneral(TcpNode):
    def __init__(self, ip, port):
        super().__init__(ip, port)

        '''msg '''
        self.msg_class = MsgGeneral
        assert self.msg_class.len_msg_pre_info_2_receive == 30

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
            if msg.prefix_dict_ is not None:
                '''record msg'''
                self.lock.acquire()  # lock self._msg here
                self._msgs.append(msg)
                self.lock.release()  # unlock self._msg here
            print('listen a msg takes', time.time() - time_0)
    ''' threading implementation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


class TcpServerGeneral(TcpNodeGeneral, TcpServer):
    def __init__(self, ip, port):
        server_address = (ip, port)
        super().__init__(ip, port)

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

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< utils'''
    ''' utils >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


class TcpClientGeneral(TcpNodeGeneral, TcpClient):
# class TcpClientGeneral(TcpClient):
    def __init__(self, ip, port):
        """"""
        super().__init__(ip, port)
        self.msg_class = MsgGeneral

    def start_thread_listener_and_speaker_with_vision_sensor(self):
        """"""
        t_listener = threading.Thread(target=self._listen_looping)  # start a thread to listen
        t_listener.start()
        t_speaker = threading.Thread(target=self._speak_looping)  # start a thread to listen
        t_speaker.start()
        return True


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
                img = msg.prefix_dict_['data']
                print(img.shape)
                cv2.imshow('data', img)
                cv2.waitKey(0)

            '''send back'''
            # for msg in msgs:
            node.add_msgs(msgs)

    # node.speak(msgs)
    # node.speak(img)


def test_as_client():
    msg = MsgGeneral()
    img_filepath = r'../data/20210902153900.png'
    # f = open(img_filepath, 'rb')
    # img_bytes = f.read()

    # img = cv2.imread(img_filepath)
    arr = np.arange(0, 10)
    content = {
        'type': '1d_array',
        'id': '0' * 20,
        'command': 'Track_Start_New_Image_Xml_Left',
    }
    msg.prefix_dict_ = content
    msg.data_ = arr

    node = TcpClientGeneral('127.0.0.1', 7000)
    node.add_msgs([msg])
    del msg

    print(node.status())
    node.start_thread_listener_and_speaker_with_vision_sensor()

    while True:
        if len(node.peek_msg()) > 0:
            print(node.status())

            '''receive'''
            msgs = node.pop_msgs()

            # print(msgs[0])
            for msg in msgs:
                # img_bytes = msg.prefix_dict_['data']
                # img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                # # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                # img = img_array.reshape((493, 853, 3))
                # print(img.shape)
                # cv2.imshow('data', img)
                # cv2.waitKey(0)
                arr = msg.data_
                print(arr)

            '''send back'''
            node.add_msgs(msgs)


def main():
    # test_data_com_thread()
    test_as_client()


if __name__ == '__main__':
    main()
