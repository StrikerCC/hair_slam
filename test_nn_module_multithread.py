import tracking_roi
import threading


def main():
    Warning('start setup threading')
    t1 = threading.Thread(target=tracking_roi.main, name='t1', args='0')
    t2 = threading.Thread(target=tracking_roi.main, name='t2', args='1')
    # t3 = threading.Thread(target=tracking_roi.main, name='t3')

    Warning('start run threading')
    t1.start()
    t2.start()
    # t3.start()

    Warning('start wait threading')
    t1.join()
    t2.join()
    # t3.join()

    return


if __name__ == '__main__':
    main()
