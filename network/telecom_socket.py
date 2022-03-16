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
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self._SEND_BUF_SIZE)  # set a new buffer size
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._RECV_BUF_SIZE)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        '''init sock'''
        server_address = (ip, port)
        print("starting listen on ip %s, port %s" % server_address)
        self._sock.bind(server_address)  # bind port

        '''status init'''
        self._connected = False
        self._init_receive_len = 256
        self._heard_something = False

        '''output msgs'''
        self._msgs = []

        '''lock'''
        self.lock = threading.Lock()

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< setter'''

    def start_listening(self):
        """"""
        client = self._stand_by()
        t = threading.Thread(target=self._listen, args=[client])  # start a thread to listen
        t.start()
        return True

    def end_listen(self):
        self._connected = False
        try:
            self._sock.close()
            self._sock.detach()
        except socket.error:
            print("fail to close on port ")
            sys.exit(1)

    def pop_msg(self):
        self.lock.acquire()  # lock self._msg here
        msgs = self._msgs
        self._msgs = []
        self._heard_something = False
        self.lock.release()  # unlock self._msg here
        return msgs

    '''setter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< getter'''
    def send_msg(self, msg):
        msg = msg.encode('utf-8')
        self._sock.send(msg)

    def peek_msg(self):
        msgs = self._msgs
        return msgs

    '''getter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< threading implementation'''

    def _stand_by(self):
        try:
            self._sock.listen(1)
        except socket.error:
            print("fail to listen on port ")
            sys.exit(1)

        '''wait connection'''
        while True:
            print("waiting for connection at")
            client, address = self._sock.accept()
            self._connected = True
            print("having a connection at", address)
            break
        return client

    def _listen(self, client):
        """"""
        '''receive msg'''
        print("start listening")
        while True:
            if not self.is_connected():
                break
            msg = client.recv(self._init_receive_len)
            msg_des_string = msg.decode('utf-8')

            self.lock.acquire()  # lock self._msg here
            self._heard_something = True
            self._msgs.append(msg_des_string)
            self.lock.release()  # unlock self._msg here

        return True

    ''' threading implementation >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< status getter '''

    def is_connected(self):
        return self._connected

    def reset(self):
        self._connected = False
        try:
            self._sock.close()
        except socket.error:
            print("fail to close on port ")
            sys.exit(1)
        self._msgs = []

    def heard_something(self):
        return self._heard_something

    def status(self):
        return 'config ' + str(self._sock.getsockname()) + \
               ' connecting: ' + str(self._connected) + \
               ' new msg received: ' + str(self._heard_something)

    ''' status getter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'''


class TcpServerCommandOnly(TcpServer):
    def __init__(self, ip, port):
        super(TcpServerCommandOnly, self).__init__(ip, port)
        self._receive_len = 10
        self._msg_unit_len = 2

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< threading implementation '''

    def _listen(self, client):
        """"""
        '''receive msg'''
        print("start listening")

        msg_left_over = ''
        while True:
            if not self.is_connected():
                break
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
        self._init_receive_len = 4
        self._msg_unit_len = 0

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< getter'''

    def send_msg(self, msg):
        assert isinstance(msg, dict)
        msg_type = msg['type']
        msg_data = msg['data']

        msg_type = msg_type.encode('utf-8')
        msg_data = msg_data.encode('utf-8')
        msg_len = len(msg_data)

        msg_type = format(msg_type, '0'+str(self._init_receive_len)+'d')
        msg_len = format(msg_len, '0'+str(self._init_receive_len)+'d')

        self._sock.send(msg_len)
        self._sock.send(msg_type)
        self._sock.send(msg_data)
        return

    '''>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> getter'''

    '''<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< threading implementation'''
    def _listen(self, client):
        """"""
        '''receive msg'''
        print("start listening")
        while True:
            if not self.is_connected():
                break
            msg_len = client.recv(self._init_receive_len)
            if len(msg_len) == self._init_receive_len:
                self._heard_something = True
                msg_type = client.recv(self._init_receive_len)

                # decoding message length and type
                msg_len = struct.unpack("!I", msg_len)[0]
                msg_type = struct.unpack("!I", msg_type)[0]
                msg_data = b''
                while True:
                    msg_data_piece = client.recv(msg_len)
                    received_bytes_len = len(msg_data_piece)
                    if received_bytes_len < msg_len:
                        msg_len -= received_bytes_len
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


def main():
    listener = TcpServer('127.0.0.1', 6000)
    listener.start_listening()
    print(listener.status())

    count = 0
    while True:
        if count > 2:
            break
        if listener.heard_something():
            print('before pop', listener.status())
            msg = listener.pop_msg()
            print(msg)
            print('after pop', listener.status())
            count += 1
    listener.end_listen()

    listener = TcpServerCommandOnly('127.0.0.1', 6000)
    listener.start_listening()
    print(listener.status())

    count = 0
    while True:
        if count > 2:
            break
        if listener.heard_something():
            print('before pop', listener.status())
            msg = listener.pop_msg()
            print(msg)
            print('after pop', listener.status())
            count += 1
    listener.end_listen()


if __name__ == '__main__':
    main()
