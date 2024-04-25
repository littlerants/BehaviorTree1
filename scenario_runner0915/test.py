start = time.time()
spawn_point = carla.Transform(carla.Location(self.parkinglot))
self.parking_vecs = CarlaDataProvider.request_new_batch_actors_physics_toparkinglot('vehicle.*', amount=50,
                                                                                    spawn_point=spawn_point,
                                                                                    autopilot=False, )
command = []
for i in self.parking_vecs:
    command.append(carla.command.ApplyTransform(i.id, carla.Transform(self.parkinglot)))
self.client.apply_batch(command)
dur_time = time.time() - start
print("spawn parking vecs time:", dur_time)

self.fram += 1
# if self.fram == 300 :
#     self._ego_actor.set_simulate_physics(False)
#     self._ego_actor.set_location(carla.Location(x=-510.739, y=175,z=0) )
# if self.fram == 1000 :
#     print("len(self.other_vecs:",len(self.other_vecs))
#     start = time.time()
#     command = []
#     for i in self.other_vecs:
#         # i.set_simulate_physics(False)
#         # i.set_location(carla.Location(x=-510.739, y=175,z=0) )
#         command.append( carla.command.SetSimulatePhysics(i.id, False))
#         # command.append( carla.command.ApplyTransform(i.id,carla.Transform(carla.Location(x=-510.739, y=175,z=0)) ) )
#         command.append( carla.command.ApplyTransform(i.id,carla.Transform(self.parkinglot  )   )  )
#     self.client.apply_batch(command)
#     dur_time = time.time() - start
#     print("physic false and set location:",dur_time)
if self.fram == 500:
    print("fram:", self.fram)
    ego_wp = self._map.get_waypoint(self._ego_actor.get_location())
    same_dir_wps = get_same_dir_lanes(ego_wp)
    print("same_dir_wps:", len(same_dir_wps))
    start = time.time()
    var = 0
    print("len(self.parking_vecs):", len(self.parking_vecs))
    # for i in range(len(self.other_vecs)):
    i = 0
    command = []
    while i < len(self.parking_vecs):

        # print(self.other_vecs[0])
        # -510.102" y="109.293"
        # next_wps = ego_wp.next(ego_road_spawn_dist)
        print("tm port :", self.client.get_trafficmanager().get_port())
        for j in range(len(same_dir_wps)):
            # self.other_vecs[i].set_location(same_dir_wps[j].next(self._road_spawn_dist + var)[0].transform.location)
            # self.other_vecs[i].set_simulate_physics(True)
            print('i:', i)
            command = [carla.command.SetSimulatePhysics(self.parking_vecs[i].id, True),
                       carla.command.ApplyTransform(self.parking_vecs[i].id,
                                                    same_dir_wps[j].next(self._road_spawn_dist + var)[0].transform),
                       carla.command.SetAutopilot(self.parking_vecs[i].id, True,
                                                  self.client.get_trafficmanager().get_port())]
            self.client.apply_batch(command)
            # command.append(carla.command.SetSimulatePhysics(self.parking_vecs[i].id, True))
            # command.append(carla.command.ApplyTransform(self.parking_vecs[i].id,same_dir_wps[j].next(self._road_spawn_dist + var)[0].transform)    )
            # command.append(carla.command.SetAutopilot(self.parking_vecs[i].id, True,self.client.get_trafficmanager().get_port()))
            i += 1
            if i >= len(self.parking_vecs):
                break
            # print("i,j:",i,j)
        var += 10

    # self.client.apply_batch(command)
    command = []
    # for i in self.parking_vecs:
    #     # command.append(carla.command.SetAutopilot(i.id, True))
    #     i.set_autopilot(True)
    # self.client.apply_batch(command)
    dur_time = time.time() - start
    print("physic true and set location time:", dur_time)
# if self.fram == 5500:
#     destroy_list = []
#     start = time.time()
#     for i in self.other_vecs:
#         destroy_list.append(carla.command.DestroyActor(i))
#     self.client.apply_batch(destroy_list)  # 批量销毁车辆
#     dur_time = time.time() - start
#     print("destroy vecs time:",dur_time)
# else:









