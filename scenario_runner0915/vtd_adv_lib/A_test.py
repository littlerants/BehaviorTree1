import socket
import time
def main():
    HOST = '127.0.0.1'  # 监听的IP地址
    PORT = 65432        # 监听的端口号

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()

        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(4)
                # print('len data1:',len(data))
                time.sleep(0.05)
                if not data:
                    break
                print( data.decode() + '\n')

if __name__ == "__main__":
    main()