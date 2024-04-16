import loguru
import traceback
import time
import carla
import random

client = carla.Client("localhost", 2000)  # 创建一个Carla客户端对象，连接到本地主机的2000端口
client.set_timeout(2.0)  # 设置连接超时时间为2.0秒
world = client.load_world("Town04")  # 加载一个名为"Town04"的世界

# 设置同步模式
settings = world.get_settings()  # 获取当前世界的设置对象
# settings.fixed_delta_seconds = 0.05  # 设置固定的时间步长为0.05秒
settings.synchronous_mode = True  # 启用同步模式
tm = client.get_trafficmanager()  # 获取交通管理器对象
tm.set_synchronous_mode(True)  # 设置交通管理器的同步模式为启用

blueprint_library = world.get_blueprint_library()  # 获取蓝图库对象
bp = random.choice(
    blueprint_library.filter("vehicle.tesla.model3")
)  # 从蓝图库中随机选择一个过滤条件为"vehicle.tesla.model3"的蓝图
transform = random.choice(world.get_map().get_spawn_points())  # 从世界地图的出生点中随机选择一个

spectator = world.get_spectator()  # 获取观众对象
spectator.set_transform(
    carla.Transform(
        transform.location + carla.Location(z=100), carla.Rotation(pitch=-90)
    )
)  # 设置观众的变换为给定的坐标和旋转角度

world.debug.draw_point(
    transform.location, size=0.2, color=carla.Color(0, 0, 255), life_time=1000
)  # 在给定的坐标绘制一个大小为0.2，颜色为蓝色，持续时间为1000毫秒的点

radius = 100  # 半径为100
num_vehicles = 100  # 车辆数量为100

spawn_points = world.get_map().get_spawn_points()  # 获取所有出生点的列表
spawn_points_filtered = []  # 过滤后的出生点列表
for i, spawn_point in enumerate(spawn_points):  # 遍历所有出生点
    if spawn_point.location.distance(transform.location) < radius:  # 如果出生点与给定坐标的距离小于半径
        spawn_points_filtered.append(spawn_point)  # 将出生点添加到过滤后的列表中
        world.debug.draw_point(
            spawn_point.location, size=0.2, color=carla.Color(255, 0, 0), life_time=10
        )  # 在给定的坐标绘制一个大小为0.2，颜色为红色，持续时间为10毫秒的点
    else:
        world.debug.draw_point(
            spawn_point.location, size=0.2, color=carla.Color(255, 255, 0), life_time=10
        )  # 在给定的坐标绘制一个大小为0.2，颜色为黄色，持续时间为10毫秒的点

vehicle_list = []  # 车辆列表

try:
    while True:
        start_time = time.perf_counter()  # 获取当前时间
        world.tick()  # 更新世界
        destroy_list = []
        # 移除超出范围的车辆
        for vehicle in vehicle_list:  # 遍历车辆列表
            vloc = vehicle.get_location()  # 获取车辆的位置
            if vloc.distance(transform.location) > radius:  # 如果车辆与给定坐标的距离大于半径
                # loguru.logger.debug(
                #     f"{vloc}, {transform.location}, {vloc.distance(transform.location)}, {radius}, {vehicle.id} out of range."
                # )
                vehicle_list.remove(vehicle)  # 从车辆列表中移除该车辆
                destroy_list.append(carla.command.DestroyActor(vehicle))
        client.apply_batch(destroy_list)  # 批量销毁车辆
        # 补充车辆
        if len(vehicle_list) < num_vehicles:  # 如果车辆列表中的车辆数量小于设定的数量
            random.seed(time.time())  # 使用当前时间作为随机种子
            random.shuffle(spawn_points_filtered)  # 将过滤后的出生点列表进行随机重排序
            for n, trans in enumerate(spawn_points_filtered):  # 遍历过滤后的出生点列表
                if num_vehicles > len(vehicle_list):  # 如果需要补充的车辆数量大于车辆列表中的车辆数量
                    npc = world.try_spawn_actor(bp, trans)  # 在给定的出生点尝试生成一个NPC车辆
                    if npc is not None:  # 如果生成成功
                        vehicle_list.append(npc)  # 将生成的车辆添加到车辆列表中
                        npc.set_autopilot(True)  # 设置车辆为自动驾驶模式
                        loguru.logger.debug("created %s" % npc.id)  # 记录日志：创建了车辆
        loguru.logger.debug(
            f"FPS: {1/(time.perf_counter() - start_time)}, Vehicles: {len(vehicle_list)}"
        )  # 记录日志：帧率和车辆数量

except KeyboardInterrupt:
    loguru.logger.warning("Quitting...")
except Exception:
    loguru.logger.error(traceback.format_exc())  # 记录错误日志
finally:
    settings.synchronous_mode = False  # 取消同步模式
    tm.set_synchronous_mode(False)  # 取消交通管理器的同步模式
    # settings.fixed_delta_seconds = None  # 取消固定的时间步长
    world.apply_settings(settings)  # 应用设置

    loguru.logger.info("destroying %d vehicles" % len(vehicle_list))  # 记录日志：销毁车辆数量
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])  # 批量销毁车辆
    loguru.logger.info("done.")  # 记录日志：完成
    time.sleep(0.5)  # 休眠0.5秒



