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

    msg_left_over = ''
    while True:
        msg = client.recv(10)
        len_msg = 2
        msg_des_string = msg_left_over + msg.decode('utf-8')
        print(ip, 'at', port, 'give', msg_des_string)

        msg_des = []
        num_msg = len(msg_des_string) // len_msg
        msg_left_over = msg_des_string[num_msg*len_msg:]

        for i in range(num_msg):
            msg_des.append(msg_des_string[i*len_msg:i*len_msg+len_msg])
        print('heard', msg_des, 'from', ip, 'at', port)

        for msg_de in msg_des:
            print('heard', msg_de, 'from', ip, 'at', port)
            request = signal_2_request[msg_de]

            print('in  <<<<<<<<<<<<<<<<<<<<<<<<<', sock, '<-', msg_de, request)
            ''' <<<<<<<<<<<<<<<<<<<<<<<<<<<< match >>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
            if request == 'Match_Start_Image_Xml':
                finished_msg = 'Match_End_Xml'
                finished_signal = request_2_signal[finished_msg]
                t_match = threading.Thread(target=multithread_func.match_start_image_xml, name='match_start_image_xml',
                                           args=(socks[0], finished_signal))
                t_match.start()

            # ''' <<<<<<<<<<<<<<<<<<<<<<<<<<<< track left >>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
            elif request == 'Track_Start_New_Image_Xml_Left':
                finished_msg = 'Track_End_Xml_Left'
                finished_signal = request_2_signal[finished_msg]
                t_track_l = threading.Thread(target=multithread_func.track_start_new_image_xml_left,
                                             name='track_start_new_image_xml_left', args=(socks[0], finished_signal))
                t_track_l.start()

            elif request == 'Track_Start_Image_Left':
                finished_msg = 'Track_End_Xml_Left'
                finished_signal = request_2_signal[finished_msg]
                t_track_l = threading.Thread(target=multithread_func.track_start_image_left, name='track_start_image_left',
                                             args=(socks[0], finished_signal))
                t_track_l.start()

            # ''' <<<<<<<<<<<<<<<<<<<<<<<<<<<< track right >>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
            elif request == 'Track_Start_New_Image_Xml_Right':
                finished_msg = 'Track_End_Xml_Right'
                finished_signal = request_2_signal[finished_msg]
                t_track_l = threading.Thread(target=multithread_func.track_start_new_image_xml_right,
                                             name='track_start_new_image_xml_right', args=(socks[0], finished_signal))
                t_track_l.start()
            elif request == 'Track_Start_Image_Right':
                finished_msg = 'Track_End_Xml_Right'
                finished_signal = request_2_signal[finished_msg]
                t_track_l = threading.Thread(target=multithread_func.track_start_image_right, name='track_start_image_right',
                                             args=(socks[0], finished_signal))
                t_track_l.start()

            # ''' <<<<<<<<<<<<<<<<<<<<<<<<<<<< mask left >>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
            elif request == 'Mask_Start_Image_Xml_Left':
                finished_msg = 'Mask_End_Left'
                finished_signal = request_2_signal[finished_msg]
                t_track_l = threading.Thread(target=multithread_func.mask_start_image_xml_left,
                                             name='mask_start_image_xml_left', args=(socks[0], finished_signal))
                t_track_l.start()
            elif request == 'Mask_Request_Left':
                finished_msg = 'Mask_End_Left'
                finished_signal = request_2_signal[finished_msg]
                t_track_l = threading.Thread(target=multithread_func.request_left_mask,
                                             name='request_left_mask', args=(socks[0], finished_signal))
                t_track_l.start()

            # ''' <<<<<<<<<<<<<<<<<<<<<<<<<<<< mask right >>>>>>>>>>>>>>>>>>>>>>>>>>>>'''
            elif request == 'Mask_Start_Image_Xml_Right':
                finished_msg = 'Mask_End_Right'
                finished_signal = request_2_signal[finished_msg]
                t_track_l = threading.Thread(target=multithread_func.mask_start_image_xml_right,
                                             name='mask_start_image_xml_right', args=(socks[0], finished_signal))
                t_track_l.start()
            elif request == 'Mask_Request_Right':
                finished_msg = 'Mask_End_Right'
                finished_signal = request_2_signal[finished_msg]
                t_track_l = threading.Thread(target=multithread_func.request_right_mask,
                                             name='request_right_mask', args=(socks[0], finished_signal))
                t_track_l.start()
            else:
                warnings.warn('Invalid request')

            print("###############################")

# def send_msg(msg):
#     conn = socks[0]
#     msg = ("%d" % msg)
#     msg = msg.encode('utf-8')
#     print("send ###############################")
#     print(msg)
#     print("send ###############################")
#     conn.send(msg)


def closeSock():
    conn = socks[0]
    conn.close()
    sock.close()


def send_msg(sock, msg):
    mono = sock
    # if isinstance(msg, float):
    #     msg = int(msg)
    # if isinstance(msg, int):
    #     msg = ("%d" % msg)
    msg = msg.encode('utf-8')
    mono.send(msg)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_tcp_server('127.0.0.1', 6000)
