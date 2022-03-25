import socket
import sys
import threading
import struct
import cv2
import numpy as np


class TcpServer:
    def __init__(self, ip, port):
        """"""
        '''socket setup'''
        self._SEND_BUF_SIZE = 256
        self._RECV_BUF_SIZE = 256

        '''tele setup'''
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._SEND_BUF_SIZE)  # set a new buffer size
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._RECV_BUF_SIZE)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        '''init sock'''
        self._sock_client = []
        server_address = (ip, port)
        print("starting listen on ip %s, port %s" % server_address)
        self._socket.bind(server_address)  # bind port
        '''status init'''
        self._connected = False
        self._msg_key_len = 256
        self._heard_something = False

        '''output msgs'''
        self._msgs = []

        '''lock'''
        self.lock = threading.Lock()

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< setter'''
    def end(self):
        self._connected = False
        try:
            self._socket.close()
            self._socket.detach()
        except socket.error:
            print("fail to close on port ")
            sys.exit(1)

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
            self._connected = True
            self._sock_client.append(client)
            self.lock.release()
            print("having a connection at", address)
            break
        return client
    '''setter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< getter'''
    def peek_msg(self):
        msgs = self._msgs
        return msgs
    '''getter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< threading implementation'''
    def _listen_threading(self):
        """"""
        '''receive msg'''
        print("start listening")
        while True:
            if not self.is_connected():
                break
            client = self._sock_client[0]
            msg = client.recv(self._msg_key_len)
            msg_des_string = msg.decode('utf-8')

            self.lock.acquire()  # lock self._msg here
            self._heard_something = True
            self._msgs.append(msg_des_string)
            self.lock.release()  # unlock self._msg here
        return True

    def _speak_threading(self):
        """"""
        '''receive msg'''
        print("start listening")
        while True:
            if not self.is_connected():
                break
            self.lock.acquire()  # lock self._msg here
            self._heard_something = False
            msg = self._pop_msg()
            self.lock.release()  # unlock self._msg here
            self.speak(msg)
        return True
    ''' threading implementation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< status getter '''
    def is_connected(self):
        return self._connected

    def reset(self):
        self._connected = False
        try:
            self._socket.close()
        except socket.error:
            print("fail to close on port ")
            sys.exit(1)
        self._msgs.clear()
        self._sock_client.clear()

    def heard_something(self):
        return self._heard_something

    def status(self):
        return 'config ' + str(self._socket.getsockname()) + \
               ' connecting: ' + str(self._connected) + \
               ' new msg received: ' + str(self._heard_something)
    ''' status getter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< utils'''
    def speak(self, msgs):
        client = self._sock_client[0]
        if not isinstance(msgs, list) and not isinstance(msgs, tuple):
            msgs = [msgs]
        for msg in msgs:
            msg = self._encoding_msg(msg)   # make anything into transportable
            client.send(msg)
        return True

    def _pop_msg(self):
        self.lock.acquire()  # lock self._msg here
        msgs = self._msgs
        self._msgs = []
        self._heard_something = False
        self.lock.release()  # unlock self._msg here
        return msgs

    def _encoding_msg(self, msg):
        """
        cast msg into transportable obj or list of transportable obj
        :param msg: msg, string
        :return:
        """
        if msg is None:
            msg = ''
        if isinstance(msg, str):
            msg = msg.encode('utf-8')
        elif isinstance(msg, int):
            msg = struct.pack('!I', msg)
        elif isinstance(msg, np.ndarray):
            msg = msg.tobytes()
        elif isinstance(msg, bytes):
            msg = msg
        else:
            raise TypeError('Unexpect type to decode: ' + str(type(msg)))
        return msg
    ''' utils >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


class TcpServerCommandOnly(TcpServer):
    def __init__(self, ip, port):
        super(TcpServerCommandOnly, self).__init__(ip, port)
        self._receive_len = 10
        self._msg_unit_len = 2

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< setter'''
    def start_thread_listener_and_speaker_for_vision_sensor(self):
        """"""
        self.stand_by()
        t_listener = threading.Thread(target=self._listen_threading)  # start a thread to listen
        t_listener.start()
        t_speaker = threading.Thread(target=self._speak_threading)  # start a thread to listen
        t_speaker.start()
        return True
    '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> setter'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< threading implementation '''
    def _listen_threading(self):
        """"""
        '''receive msg'''
        print("start listening")

        msg_left_over = ''
        while True:
            if not self.is_connected():
                break
            client = self._sock_client[0]
            msg = client.recv(self._receive_len)
            msg_des_string = msg_left_over + msg.decode('utf-8')
            num_msg = len(msg_des_string) // self._msg_unit_len
            msg_left_over = msg_des_string[num_msg * self._msg_unit_len:]

            self.lock.acquire()     # lock self._msg here
            self._heard_something = True
            for i_msg_unit in range(num_msg):
                self._msgs.append(msg_des_string[i_msg_unit * self._msg_unit_len:(i_msg_unit + 1) * self._msg_unit_len])
            self.lock.release()     # unlock self._msg here

        return True

    ''' threading implementation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


class TcpServerGeneral(TcpServer):
    def __init__(self, ip, port):
        super(TcpServerGeneral, self).__init__(ip, port)
        self._msg_key_len = 4
        self._msg_id_len = 20
        self._msg_unit_len = 0
        self._msg_2_command_mapping = {
            '01': ''
        }
        self._msg_2_datatype_mapping = {
            '01': 'img'
        }

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< setter'''
    def start_thread_listener_and_speaker_for_vision_sensor(self):
        """"""
        self.stand_by()
        t_listener = threading.Thread(target=self._listen_threading)  # start a thread to listen
        t_listener.start()
        t_speaker = threading.Thread(target=self._speak_threading)  # start a thread to listen
        t_speaker.start()
        return True
    '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> setter'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< getter'''
    '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> getter'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< threading implementation'''
    def _listen_threading(self):
        """"""
        '''receive msg'''
        print("start listening")
        while True:
            if not self.is_connected():
                break
            client = self._sock_client[0]
            msg_data_len = client.recv(self._msg_key_len)
            if len(msg_data_len) == self._msg_key_len:
                self._heard_something = True
                msg_type = client.recv(self._msg_key_len)

                # decoding message length and type
                msg_data_len = struct.unpack("!I", msg_data_len)[0]
                msg_type = struct.unpack("!I", msg_type)[0]
                msg_data = b''
                while True:
                    msg_data_piece = client.recv(msg_data_len)
                    received_bytes_len = len(msg_data_piece)
                    if received_bytes_len < msg_data_len:
                        msg_data_len -= received_bytes_len
                        msg_data += msg_data_piece
                    else:
                        break

                # decoding message data
                msg_data = np.frombuffer(msg_data, dtype=np.uint8)

                self.lock.acquire()     # lock self._msg here
                self._msgs.append({'type': msg_type,
                                   'data': msg_data})
                self.lock.release()     # unlock self._msg here
        return True
    ''' threading implementation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< utils'''
    def _encoding_msg(self, msg):
        """
        cast msg into transportable obj or list of transportable obj
        :param msg: msg obj, dict for sure
        :return:
        """
        if not isinstance(msg, dict):
            return super(TcpServerGeneral, self)._encoding_msg(msg)
        elif isinstance(msg, dict):
            msg_type = msg['type']
            msg_data = msg['data']
            msg_id = msg.get('id', '')
            msg_len = len(msg_data) if msg_data is not None else 0

            msg_type = super(TcpServerGeneral, self)._encoding_msg(msg_type)
            msg_len = super(TcpServerGeneral, self)._encoding_msg(msg_len)
            msg_data = super(TcpServerGeneral, self)._encoding_msg(msg_data)
            # msg_id = self.__format_msg_peice(msg_id)
            return msg_type + msg_len + msg_data  # type, len, id, data
        else:
            raise TypeError('Unexpect type to decode: ' + str(type(msg)))
    ''' utils >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


def test_command():
    listener = TcpServer('127.0.0.1', 6000)
    listener.start_thread_listener_and_speaker_for_vision_sensor()
    print(listener.status())

    count = 0
    while True:
        if count > 2:
            break
        if listener.heard_something():
            print('before pop', listener.status())
            msg = listener._pop_msg()
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
            msg = listener._pop_msg()
            print(msg)
            print('after pop', listener.status())
            count += 1
    listener.end()


def test_data_com():
    node = TcpServerGeneral('127.0.0.1', 6000)
    # node.start_thread_listener_and_speaker()
    node.stand_by()
    print(node.status())

    # while True:
    s = 'what\'sup'
    node.speak(s)
    print(s)

    s = 0
    node.speak(s)
    print(s)

    s = b'what\'sup'
    node.speak(s)
    print(s)

    s = b'000'
    node.speak(s)
    print(s)

    s = np.arange(0, 10)
    node.speak(s)
    print(s)

    # img_filepath = r'./data/20210902153900.png'
    # img = cv2.imread(img_filepath)
    # msgs = [{
    #     'type': 1,
    #     'data': img
    # }]
    # node._send_msg(msgs)


def main():
    test_data_com()

    
if __name__ == '__main__':
    main()
