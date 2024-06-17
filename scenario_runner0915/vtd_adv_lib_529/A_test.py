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
    # import dearpygui.dearpygui as dpg
    #
    # dpg.create_context()
    #
    # dpg.show_documentation()
    # dpg.show_style_editor()
    # dpg.show_debug()
    # dpg.show_about()
    # dpg.show_metrics()
    # dpg.show_font_manager()
    # dpg.show_item_registry()
    #
    # dpg.create_viewport(title='Custom Title', width=800, height=600)
    # dpg.setup_dearpygui()
    # dpg.show_viewport()
    # dpg.start_dearpygui()
    # dpg.destroy_context()

    # import dearpygui.dearpygui as dpg
    # import dearpygui.demo as demo
    #
    # dpg.create_context()
    # dpg.create_viewport(title='Custom Title', width=600, height=600)
    #
    # demo.show_demo()
    #
    # dpg.setup_dearpygui()
    # dpg.show_viewport()
    # dpg.start_dearpygui()
    # dpg.destroy_context()