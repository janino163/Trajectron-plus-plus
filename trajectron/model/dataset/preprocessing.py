import torch
import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill
container_abcs = collections.abc
from scipy.spatial.distance import cdist
import pandas as pd
from environment import Scene, Node
import time
import copy
def augment_traversal_node(target_node, scene, t, max_ht):
    metrix = 'euclidean'
    data_columns_vehicle = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration', 'heading'], ['x', 'y']])
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°'), ('heading', 'd°')]))
    data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_product([['velocity', 'acceleration'], ['norm']]))
    data_columns_pedestrian = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name, non_aug_scene=scene)
    
    target_position = target_node.get(np.array([t]), [('position', 'x'), ('position', 'y')])
    for node in scene.nodes:
        if node.type == 'PEDESTRIAN':
            position = node.get(np.array([0, scene.timesteps]), [('position', 'x'), ('position', 'y')])
            dist_ = cdist(target_position, position)
            match_index = np.nanargmin(dist_, axis=None)
            state_ = [('position', 'x'), ('position', 'y'), ('velocity', 'x'),('velocity', 'y')]
            state_.append(('acceleration', 'x'))
            state_.append(('acceleration', 'y'))
            
            
            temp_node_data = node.get(np.array([match_index, node.last_timestep]),state_)
            
            
            x = temp_node_data[:,0]
            y = temp_node_data[:,1]
            vx = temp_node_data[:,2]
            vy = temp_node_data[:,3]
            ax = temp_node_data[:,4]
            ay = temp_node_data[:,5]
            
            
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay}
            
            node_data = pd.DataFrame(data_dict, columns=data_columns_pedestrian)
            first_valid = node_data.first_valid_index()
            
            
            node_data = node_data.iloc[first_valid:]
            assert first_valid == 0
            first_timestep = np.clip(t-max_ht,0,124)

            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=first_timestep)
        elif node.type == 'VEHICLE':
#             x = node.data.position.x.copy().reshape((-1,1))
#             y = node.data.position.y.copy().reshape((-1,1))
            
#             position = np.hstack((x, y))
            position = node.get(np.array([0, scene.timesteps]), [('position', 'x'), ('position', 'y')])

            dist_ = cdist(target_position, position)
            match_index = np.nanargmin(dist_, axis=None)
            
            state_ = [('position', 'x'), ('position', 'y'), ('velocity', 'x'), ('velocity', 'y'), ('velocity', 'norm')]
            state_.append(('acceleration', 'x'))
            state_.append(('acceleration', 'y'))
            state_.append(('acceleration', 'norm'))
            state_.append(('heading', 'x'))
            state_.append(('heading', 'y'))
            state_.append(('heading', '°'))
            state_.append(('heading', 'd°'))
#             temp_node_data = node.get(np.array([match_index - max_ht, node.last_timestep]),state_)
            temp_node_data = node.get(np.array([match_index, node.last_timestep]),state_)
            
            
            x = temp_node_data[:,0]
            y = temp_node_data[:,1]
            vx = temp_node_data[:,2]
            vy = temp_node_data[:,3]
            v_norm = temp_node_data[:,4]
            ax = temp_node_data[:,5]
            ay = temp_node_data[:,6]
            a_norm = temp_node_data[:,7]
            heading_x = temp_node_data[:,8]
            heading_y = temp_node_data[:,9]
            heading = temp_node_data[:,10]
            heading_d = temp_node_data[:,11]
            
            
            data_dict = {('position', 'x'): x,
                         ('position', 'y'): y,
                         ('velocity', 'x'): vx,
                         ('velocity', 'y'): vy,
                         ('velocity', 'norm'): v_norm,
                         ('acceleration', 'x'): ax,
                         ('acceleration', 'y'): ay,
                         ('acceleration', 'norm'): a_norm,
                         ('heading', 'x'): heading_x,
                         ('heading', 'y'): heading_y,
                         ('heading', '°'): heading,
                         ('heading', 'd°'): heading_d}
            
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)
            node_data_debug = pd.DataFrame(data_dict, columns=data_columns_vehicle)
            
            first_valid = node_data.first_valid_index()
            node_data = node_data.iloc[first_valid:]
            
            assert first_valid == 0
            first_timestep = np.clip(t-max_ht,0,124)
#             print(f'vel valid: {first_valid}, first: {first_timestep}')
            node = Node(node_type=node.type, node_id=node.id, data=node_data, first_timestep=first_timestep,
                    non_aug_node=node)
        scene_aug.nodes.append(node)
    return scene_aug


def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data


def collate(batch):
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if elem is None:
        return None
    elif isinstance(elem, container_abcs.Sequence):
        if len(elem) == 4: # We assume those are the maps, map points, headings and patch_size
            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(scene_map,
                                                                     scene_pts=torch.Tensor(scene_pts),
                                                                     patch_size=patch_size[0],
                                                                     rotation=heading_angle)
            return map
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(elem, container_abcs.Mapping):
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return dill.dumps(neighbor_dict) if torch.utils.data.get_worker_info() else neighbor_dict
    return default_collate(batch)


def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    # TODO: We will have to make this more generic if robot_type != node_type
    # Make Robot State relative to node
    _, std = env.get_standardize_params(state[robot_type], node_type=robot_type)
    std[0:2] = env.attention_radius[(node_type, robot_type)]
    robot_traj_st = env.standardize(robot_traj,
                                    state[robot_type],
                                    node_type=robot_type,
                                    mean=node_traj,
                                    std=std)
    robot_traj_st_t = torch.tensor(robot_traj_st, dtype=torch.float)

    return robot_traj_st_t


def get_node_timestep_data(env, scene, t, node, state, pred_state,
                           edge_types, max_ht, max_ft, hyperparams,
                           scene_graph=None):
    """
    Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
    as well as the neighbour data for it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node: Node
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbours are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :param scene_graph: If scene graph was already computed for this scene and time you can pass it here
    :return: Batch Element
    """

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    if hyperparams['reconstruction']:
        timestep_range_y = np.array([t + 1 - max_ht, t + max_ft-max_ht])
    else:
        timestep_range_y = np.array([t + 1, t + max_ft])
        

    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)

    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    rel_state = np.zeros_like(x[0])
    
    if hyperparams['reconstruction']:
        rel_state[0:2] = np.array(x)[first_history_index, 0:2] # is this correct?
    else:
        rel_state[0:2] = np.array(x)[-1, 0:2]
    
    x_st = env.standardize(x, state[node.type], node.type, mean=rel_state, std=std)
    if list(pred_state[node.type].keys())[0] == 'position':  # If we predict position we do it relative to current pos
        y_st = env.standardize(y, pred_state[node.type], node.type, mean=rel_state[0:2])
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None
    
    

    if hyperparams['edge_encoding']:
        if hyperparams['traversal']:
#             start = time.time()
            traversal_scene = copy.deepcopy(scene)
            temp_nodes = []
            temp_nodes.append(node)
            assert scene_graph is None
            matched_scene_names = hyperparams['matches'][scene.name]
            matched_scenes = [s for s in env.scenes if s.name in matched_scene_names]
            for traversal_scene in matched_scenes:
                traversal_scene = augment_traversal_node(node, traversal_scene, t, max_ht)
                temp_nodes.extend(traversal_scene.nodes)
            traversal_scene.nodes = temp_nodes
            scene_graph = traversal_scene.get_scene_graph(t,
                                            env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter']) if scene_graph is None else scene_graph
#             stop = time.time() # A few seconds later
#             print(f'sec: {start-stop}, num_nodes: {len(temp_nodes)}')
        # Scene Graph
        else:
            scene_graph = scene.get_scene_graph(t,
                                            env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter']) if scene_graph is None else scene_graph

        neighbors_data_st = dict()
        neighbors_edge_value = dict()
        for edge_type in edge_types:
            neighbors_data_st[edge_type] = list()
            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams['dynamic_edges'] == 'yes':
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(scene_graph.get_edge_scaling(node), dtype=torch.float)
                neighbors_edge_value[edge_type] = edge_masks

            for connected_node in connected_nodes:
                neighbor_state_np = connected_node.get(np.array([t - max_ht, t]),
                                                       state[connected_node.type],
                                                       padding=0.0)

                # Make State relative to node where neighbor and node have same state
                _, std = env.get_standardize_params(state[connected_node.type], node_type=connected_node.type)
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = env.standardize(neighbor_state_np,
                                                       state[connected_node.type],
                                                       node_type=connected_node.type,
                                                       mean=rel_state,
                                                       std=std)

                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)

    # Robot
    robot_traj_st_t = None
    if hyperparams['incl_robot_node']:
        timestep_range_r = np.array([t, t + max_ft])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(timestep_range_r, state[robot_type], padding=0.0)
        node_state = np.zeros_like(robot_traj[0])
        node_state[:x.shape[1]] = x[-1]
        robot_traj_st_t = get_relative_robot_traj(env, state, node_state, robot_traj, node.type, robot_type)

    # Map
    map_tuple = None
    if hyperparams['use_map_encoding']:
        if node.type in hyperparams['map_encoder']:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(np.array([t]), state[node.type])
            me_hyp = hyperparams['map_encoder'][node.type]
            if 'heading_state_index' in me_hyp:
                heading_state_index = me_hyp['heading_state_index']
                # We have to rotate the map in the opposit direction of the agent to match them
                if type(heading_state_index) is list:  # infer from velocity or heading vector
                    heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                                x[-1, heading_state_index[0]]) * 180 / np.pi
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]


            patch_size = hyperparams['map_encoder'][node.type]['patch_size']
            map_tuple = (scene_map, map_point, heading_angle, patch_size)

    return (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st,
            neighbors_edge_value, robot_traj_st_t, map_tuple)


def get_timesteps_data(env, scene, t, node_type, state, pred_state,
                       edge_types, min_ht, max_ht, min_ft, max_ft, hyperparams):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    if hyperparams['reconstruction']:
        nodes_per_ts = scene.present_nodes(t,
                                           type=node_type,
                                           min_history_timesteps=min_ht,
                                           min_future_timesteps=max_ft-max_ht,
                                         return_robot=not hyperparams['incl_robot_node'])
    else:
        nodes_per_ts = scene.present_nodes(t,
                                           type=node_type,
                                           min_history_timesteps=min_ht,
                                           min_future_timesteps=max_ft,
                                           return_robot=not hyperparams['incl_robot_node'])
    batch = list()
    nodes = list()
    out_timesteps = list()
    for timestep in nodes_per_ts.keys():
            if hyperparams['traversal']:
                scene_graph = None
            else:
                scene_graph = scene.get_scene_graph(timestep,
                                                env.attention_radius,
                                                hyperparams['edge_addition_filter'],
                                                hyperparams['edge_removal_filter'])
#             scene_graph = None
            present_nodes = nodes_per_ts[timestep]
            for node in present_nodes:
                nodes.append(node)
                out_timesteps.append(timestep)
                batch.append(get_node_timestep_data(env, scene, timestep, node, state, pred_state,
                                                    edge_types, max_ht, max_ft, hyperparams,
                                                    scene_graph=scene_graph))
    if len(out_timesteps) == 0:
        return None
    return collate(batch), nodes, out_timesteps
