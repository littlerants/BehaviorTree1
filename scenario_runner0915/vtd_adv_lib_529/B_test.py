import socket
import time

def main():
    HOST = '127.0.0.1'  # B进程的IP地址
    PORT = 65432        # B进程监听的端口号

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        i = 0
        while True:
            print("fram:",i)
            data =  str(i)+","
            # print('len data:',len(data))
            s.sendall(data.encode())
            # print("Sent data to B:", data)
            time.sleep(0.01)  # 等待20ms，即50Hz的发送频率
            i += 1
import math
import numpy as np
def trans2angle(x,y,theta):
    theta = -theta
    tmp_x = x
    tmp_y = y
    new_pos_x = tmp_x * math.cos(theta) + tmp_y * math.sin(theta)
    new_pos_y = tmp_y * math.cos(theta) - tmp_x * math.sin(theta)
    print("new value:",new_pos_x,new_pos_y)

if __name__ == "__main__":
    # main()
    trans2angle(5,10,np.pi)