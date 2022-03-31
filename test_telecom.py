import copy

import cv2

from network.telecom_socket import TcpClient, TcpClientGeneral


def test():
    client = TcpClientGeneral('127.0.0.1', 6000)
    msgs_previous = []
    while True:
        client.listen()
        if client.heard_something():
            msgs = client.pop_msg()
            if len(msgs) > 0:
                print(client.status())
                msgs_previous = copy.deepcopy(msgs)
                img = msgs_previous[0]['data']

                cv2.imshow('img', img)
                cv2.waitKey(0)


def test_thread():
    client = TcpClientGeneral('127.0.0.1', 6000)
    print(client.status())
    client.start_thread_listener_and_speaker_with_vision_sensor()

    len_old = -1
    while True:
        if len(client.peek_msg()) != len_old:
            print(client.status())
            msgs = client.peek_msg()
            for msg in msgs:
                img = msg.content['img']
                cv2.imshow('img', img)
                cv2.waitKey(0)
            len_old = len(client.peek_msg())


def main():
    test_thread()


if __name__ == '__main__':
    main()
