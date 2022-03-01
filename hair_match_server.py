import socket
import xml
import sys
import socket_msg

import threading

import hair_match_superglue
import tracking_roi

'''socket set up'''
Communication_Count: int = 0
receive_count: int = 0
SEND_BUF_SIZE = 256
RECV_BUF_SIZE = 256
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socks = []

'''signal to functionality mapping'''
signal_2_request = socket_msg.requests_dic
file_path_dic = socket_msg.file_path_dic

'''functionalities'''
matcher = hair_match_superglue.Matcher()
tracker_left = tracking_roi.Tracker()
tracker_right = tracking_roi.Tracker()
tracker_mask = tracking_roi.Tracker()


def start_tcp_server(ip, port):
    # create socket
    #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, port)

    # bind port
    print("starting listen on ip %s, port %s" % server_address)
    sock.bind(server_address)

    s_send_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    s_recv_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
   # print("socket send buffer size[old] is %d" % s_send_buffer_size)
    #print("socket receive buffer size[old] is %d" % s_recv_buffer_size)

    # set a new buffer size
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SEND_BUF_SIZE)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECV_BUF_SIZE)

    # get the new buffer size
    s_send_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    s_recv_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print("socket send buffer size[new] is %d" % s_send_buffer_size)
    print("socket receive buffer size[new] is %d" % s_recv_buffer_size)

    # start listening, allow only one connection
    try:
        sock.listen(1)
    except socket.error:
        print("fail to listen on port ")
        sys.exit(1)
    while True:
        print("waiting for connection")
        client, addr = sock.accept()
        socks.append(client)
        print("having a connection")
        break
    while True:
        msg = client.recv(16384)
        msg_de = msg.decode('utf-8')
        print('heard', msg_de, 'from', ip, 'at', port)
        request = signal_2_request[msg_de]

        print('in  <<<<<<<<<<<<<<<<<<<<<<<<<', msg_de, request)
        if request == 'Match_Start_Image_Xml':
            print()
        elif request == 'Match_End_Xml':
            print()
        elif request == 'Track_Start_New_Image_Xml':
            print()
        elif request == 'Track_Start_Image_Left':
            print()
        elif request == 'Track_Start_Image_Right':
            print()
        elif request == 'Track_End_Xml':
            print()
        elif request == 'Mask_Start_Image_Xml':
            print()
        elif request == 'Mask_End':
            print()
        elif request == 'Mask_Respond_Image':
            print()
        elif request == 'Mask_Resquest_Image':
            print()
        else:
            raise Warning('Invalid request')

        print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_de, request)


        print("###############################")

        send_msg(2)


def send_msg(msg):
    conn = socks[0]
    msg = ("%d" % msg)
    msg = msg.encode('utf-8')
    print("send ###############################")
    print(msg)
    print("send ###############################")
    conn.send(msg)

def closeSock():
    conn = socks[0]
    conn.close()
    sock.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_tcp_server('192.168.1.44', 6000)

