import socket
import warnings
import xml
import sys

import cv2

import socket_msg

import threading

import multithread_func

'''socket set up'''
Communication_Count: int = 0
receive_count: int = 0
SEND_BUF_SIZE = 256
RECV_BUF_SIZE = 256
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socks = []

'''signal to functionality mapping'''
signal_2_request = socket_msg.signal_2_requests_dic
request_2_signal = socket_msg.requests_2_signal_dic


def start_tcp_server(ip, port):
    # create socket
    # sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, port)

    # bind port
    print("starting listen on ip %s, port %s" % server_address)
    sock.bind(server_address)

    s_send_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    s_recv_buffer_size = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    # print("socket send buffer size[old] is %d" % s_send_buffer_size)
    # print("socket receive buffer size[old] is %d" % s_recv_buffer_size)

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

        # t = None
        print('in  <<<<<<<<<<<<<<<<<<<<<<<<<', msg_de, request)
        # ''' <<<<<<<<<<<<<<<<<<<<<<<<<<<< match >>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
        if request == 'Match_Start_Image_Xml':
            finished_msg = 'Match_End_Xml'
            t_match = threading.Thread(target=multithread_func.match_start_image_xml, name='match_start_image_xml')
            t_match.start()
            if not t_match.is_alive():
                print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_de, request)
                send_msg(request_2_signal[finished_msg])

        # ''' <<<<<<<<<<<<<<<<<<<<<<<<<<<< track left >>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
        elif request == 'Track_Start_New_Image_Xml_Left':
            finished_msg = 'Track_End_Xml_Left'
            t_track_l = threading.Thread(target=multithread_func.track_start_image_left(), name='match_start_image_xml')
            t_track_l.start()
            if not t_track_l.is_alive():
                print("out >>>>>>>>>>>>>>>>>>>>>>>>>", msg_de, request)
                send_msg(request_2_signal[finished_msg])
        elif request == 'Track_Start_Image_Left':
            finished_msg = 'Track_End_Xml_Left'

        # ''' <<<<<<<<<<<<<<<<<<<<<<<<<<<< track right >>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
        elif request == 'Track_Start_New_Image_Xml_Right':
            finished_msg = 'Track_End_Xml_Right'
        elif request == 'Track_Start_Image_Right':
            finished_msg = 'Track_End_Xml_Right'
            print()

        # ''' <<<<<<<<<<<<<<<<<<<<<<<<<<<< mask left >>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
        elif request == 'Mask_Start_Image_Xml_Left':
            finished_msg = 'Mask_End_Left'
        elif request == 'Mask_Respond_Image_Left':
            finished_msg = 'Mask_End_Left'
        elif request == 'Mask_Resquest_Image_Left':
            finished_msg = 'Mask_End_Left'

        # ''' <<<<<<<<<<<<<<<<<<<<<<<<<<<< mask right >>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
        elif request == 'Mask_Start_Image_Xml_Right':
            finished_msg = 'Mask_End_Right'
        elif request == 'Mask_Respond_Image_Right':
            finished_msg = 'Mask_End_Right'
        elif request == 'Mask_Request_Image_Right':
            finished_msg = 'Mask_End_Right'

        elif request == 'Match_End_Xml':
            warnings.warn(request, 'should be my request')
        elif request == 'Track_End_Xml_Left':
            warnings.warn(request, 'should be my request')
        elif 'Mask_End' in request:
            warnings.warn(request, 'should be my request')

        else:
            warnings.warn('Invalid request')

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
