import socket
import sys
import threading


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
        self._sock.bind(server_address)                                                     # bind port

        # s_send_buffer_size = self._sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)     # get the new buffer size
        # s_recv_buffer_size = self._sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        # print("socket send buffer size[new] is %d" % s_send_buffer_size)
        # print("socket receive buffer size[new] is %d" % s_recv_buffer_size)

        '''status init'''
        self._connected = False
        self._receive_len = 256
        self._heard_something = False

        '''output msgs'''
        self._msgs = []

    def start_listening(self):
        """"""
        client = self._stand_by()
        t = threading.Thread(target=self._listen, args=[client])   # start a thread to listen
        t.start()
        return True

    def speak(self, msg):
        msg = msg.encode('utf-8')
        self._sock.send(msg)

    def end_listen(self):
        self._connected = False
        try:
            self._sock.close()
            self._sock.detach()
        except socket.error:
            print("fail to close on port ")
            sys.exit(1)

    def pop_msg(self):
        # lock self._msg here
        msgs = self._msgs
        self._msgs = []
        self._heard_something = False
        # unlock self._msg here
        return msgs

    def peek_msg(self):
        msgs = self._msgs
        return msgs

    def is_connected(self):
        return self._connected

    def heard_something(self):
        return self._heard_something

    def reset(self):
        self._connected = False
        try:
            self._sock.close()
        except socket.error:
            print("fail to close on port ")
            sys.exit(1)
        self._msgs = []

    def status(self):
        return 'config ' + str(self._sock.getsockname()) + \
               ' connecting: ' + str(self._connected) + \
               ' new msg received: ' + str(self._heard_something)

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
            msg = client.recv(self._receive_len)
            msg_des_string = msg.decode('utf-8')

            # lock self._msg here
            self._heard_something = True
            self._msgs.append(msg_des_string)
            # unlock self._msg here

        return True


class TcpServerCommandOnly(TcpServer):
    def __init__(self, ip, port):
        super(TcpServerCommandOnly, self).__init__(ip, port)
        self._receive_len = 10
        self._msg_unit_len = 2

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
            msg_left_over = msg_des_string[num_msg*self._msg_unit_len:]

            # lock self._msg here
            self._heard_something = True
            for i_msg_unit in range(num_msg):
                self._msgs.append(msg_des_string[i_msg_unit * self._msg_unit_len:(i_msg_unit+1) * self._msg_unit_len])
            # unlock self._msg here

        return True


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
