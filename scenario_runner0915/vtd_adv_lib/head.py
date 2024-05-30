# from torch._C import T
# pkg 宏定义
RDB_PKG_ID_START_OF_FRAME     =     1    # /**< sent as first package of a simulation frame                      @version 0x0100 */
RDB_PKG_ID_END_OF_FRAME       =     2  
RDB_PKG_ID_ROAD_POS           =     5  
RDB_PKG_ID_LANE_INFO          =     6
RDB_PKG_ID_OBJECT_STATE       =     9  
RDB_PKG_ID_ROAD_STATE         =    21    # /**< road state information for a given player                        @version 0x0100 */
RDB_PKG_ID_TRAFFIC_LIGHT      =    27    # /**< information about a traffic lights and their states              @version 0x0100 */
# 交通信号灯状态宏定义
LIGHT_SIGNAL = {1:'STOP', 3:'GO',5:'ATT'}
# RDB校验码
RDB_MAGIC_NO  = 35712
# RDB 头字节数
SIZE_RDB_MSG_HDR_t = 24
# 显示log开关
show_log = 0
