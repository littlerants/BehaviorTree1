
// ExampleConsole.cpp : Defines the entry point for the console application.
// //
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include<sys/epoll.h>
#include "VtdToolkit/RDBHandler.hh"
#include<iostream>
#include<ctime>
#include<unistd.h>
#include<cstdio>
using namespace std;
// #define DEFAULT_PORT        48190   /* for image port it should be 48192 */
// #define DEFAULT_PORT        48190   /* for image port it should be 48192   48185*/
// #define SENSOR_PORT        48195
// #define INFERENCE          5200


// #define DEFAULT_BUFFER      204800
using namespace Framework;
// #include<iostream>
// #include<string>
// using namespace std;
           // Server to connect to
// int   iPort     = DEFAULT_PORT;  // Port on server to connect to


// typedef struct
// {
//     float    deltaT;           /**< delta time by which to advance the simulation        @unit s                                    @version 0x0100 */
//     uint32_t frameNo;          /**< number of the simulation frame which is triggered    @unit _                                    @version 0x0100 */
//     uint16_t features;         /**< mask of features that shall be computed              @unit _                                    @version 0x011B */
//     int16_t  spare0;           /**< spare for future use                                 @unit _                                    @version 0x011B */
// } RDB_TRIGGER_t;

extern "C"{
char  szServer[128] ;  

vector<RDB_DRIVER_CTRL_t  * >driverctrl;
Framework::RDBHandler myHandler;
void ValidateArgs()
{   
    strcpy( szServer, "127.0.0.1" );
}


int connect1(int iPort = 48190){
        // tcp_server_inf = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        // # 设置端口复用，使程序退出后端口马上释放
        // tcp_server_inf.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

        // tcp_server_inf.connect(("127.0.0.1",tc_port))
    ValidateArgs();
    int    sClient;
    
    
    struct sockaddr_in server;
    struct hostent    *host = NULL;
    sClient = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if ( sClient == -1 )
    {
        fprintf( stderr, "socket() failed: %s\n", strerror( errno ) );
        return 1;
    }
    
    int opt = 1;
    setsockopt ( sClient, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof( opt ) );
    server.sin_family      = AF_INET;
    server.sin_port        = htons(iPort);
    server.sin_addr.s_addr = inet_addr(szServer);

    //
    if ( server.sin_addr.s_addr == INADDR_NONE )
    {
        host = gethostbyname(szServer);
        if ( host == NULL )
        {
            fprintf( stderr, "Unable to resolve server: %s\n", szServer );
            return 1;
        }
        memcpy( &server.sin_addr, host->h_addr_list[0], host->h_length );
    }
	// wait for connection
	bool bConnected = false;

    while ( !bConnected )
    {
        if (connect( sClient, (struct sockaddr *)&server, sizeof( server ) ) == -1 )
        {
            fprintf( stderr, "connect() failed: %s\n", strerror( errno ) );
            sleep( 1 );
        }
        else
            bConnected = true;
    }
    fprintf( stderr, "connected!\n" );
    
    return sClient;
};

void clear(){
    driverctrl.clear();
    myHandler.initMsg();
}
int get_msg_num(){
    return driverctrl.size();
}
// void sendTrigger( int sendSocket, const double  simTime, const unsigned int  simFrame )
int addPkg(const double  simTime, const unsigned int   simFrame, const double acc , const double steer,const uint32_t playerId = 1, const int flag = 0 )
{
    // flag 0: autopilot
    // flag 1: just acc
    // flag 2: just steer
    // flag 3: acc + steer
    // flag 4: gear
    printf("simTime: %f\n",simTime);
    printf("simFrame: %d\n",simFrame);

    uint32_t  validityFlags = RDB_DRIVER_INPUT_VALIDITY_ADD_ON;
    switch (flag)
    {
    case 0: /* constant-expression */
        /* code */
        break;
    case 1:
        validityFlags |= RDB_DRIVER_INPUT_VALIDITY_TGT_ACCEL;
        break;
    case 2:
        validityFlags |= RDB_DRIVER_INPUT_VALIDITY_TGT_STEERING;
        break;

    case 3:
        validityFlags |= (RDB_DRIVER_INPUT_VALIDITY_TGT_ACCEL |  RDB_DRIVER_INPUT_VALIDITY_TGT_STEERING);
        break;

    case 4:
        validityFlags |= RDB_DRIVER_INPUT_VALIDITY_GEAR;
        break;
    default:
        cout<<"flag is invalide"<<endl;
    }

    // cout<<"sendSocket:"<<sendSocket<<endl;
    // cout<<"simTime:"<<simTime<<endl;
    // cout<<"simFrame"<<simFrame<<endl;

    // RDB_TRIGGER_t *myTrigger = ( RDB_TRIGGER_t* ) myHandler.addPackage( simTime, simFrame, RDB_PKG_ID_TRIGGER );

    // if ( !myTrigger )
    //     return 0 ;

    // myTrigger->frameNo = simFrame + 1;
    // myTrigger->deltaT  = 0.043;

    RDB_DRIVER_CTRL_t *myDriver = ( RDB_DRIVER_CTRL_t* ) myHandler.addPackage( simTime, simFrame, RDB_PKG_ID_DRIVER_CTRL );

    if ( !myDriver )
        return 0;

    myDriver->playerId = playerId;
    myDriver->gear = 4;
    myDriver->accelTgt      = acc;
    myDriver->steeringTgt = steer;
    cout<<"playerId:"<<playerId<<"    accelTgt:"<<      myDriver->accelTgt <<"    steeringTgt:"<<myDriver->steeringTgt<<"    flag:"<<flag <<endl;
    myDriver->validityFlags = validityFlags;
    driverctrl.push_back(myDriver);
    return 0;
}

int sendTrigger( int sendSocket,const double  simTime, const unsigned int simFrame, int send_trigger = 1){
    if (send_trigger){
        RDB_TRIGGER_t *myTrigger = ( RDB_TRIGGER_t* ) myHandler.addPackage( simTime, simFrame, RDB_PKG_ID_TRIGGER );
        if ( !myTrigger )
            return -1;

        myTrigger->frameNo = simFrame + 1;
        myTrigger->deltaT  = 0.02;
    }
    printf("sendSocket: %d\n",sendSocket);
    // cout<<"--------------------"<<endl;
    for (int i = 0 ; i <driverctrl.size(); i++ ){
        cout<<driverctrl[i]->playerId<< "  "<<driverctrl[i]->accelTgt<<"  "<<driverctrl[i]->steeringTgt<<endl;
    }
    // cout<<"--------------------"<<endl;
    int retVal = send( sendSocket, ( const char* ) ( myHandler.getMsg() ), myHandler.getMsgTotalSize(), 0 );
    cout<<"---------send-----------"<<endl;
    if ( !retVal )
        fprintf( stderr, "sendTrigger: could not send trigger\n" );
        return -1;
    return 0;

}

}