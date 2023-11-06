import torch
from torch import nn, optim, utils
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
import warnings
from tqdm import tqdm
import visualization
import evaluation
import matplotlib.pyplot as plt
from argument_parser import args
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from model.dataset import EnvironmentDataset, collate
from tensorboardX import SummaryWriter
import collections
from scipy.stats.distributions import chi2
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os.path as osp
import json

RESULTS_SAVE_PATH = '/home/jan268/temp/anomaly_data/'
# torch.autograd.set_detect_anomaly(True)

if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        args.device = 'cuda:0'

    args.device = torch.device(args.device)

if args.eval_device is None:
    args.eval_device = torch.device('cpu')

# This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
torch.cuda.set_device(args.device)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def confidence_ellipse(cov, mean_x, mean_y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ellipse

def expand_matches(matches):
    result = dict()
    match_keys = list(matches.keys())
    for match_key in match_keys:
        scenes = matches[match_key]
        for i, scene in enumerate(scenes):
            result[scene] = scenes[:i] + scenes[i+1:]
    return result

def main():
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['edge_addition_filter'] = args.edge_addition_filter
    hyperparams['edge_removal_filter'] = args.edge_removal_filter
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['offline_scene_graph'] = args.offline_scene_graph
    hyperparams['incl_robot_node'] = args.incl_robot_node
    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['use_map_encoding'] = args.map_encoding
    hyperparams['augment'] = args.augment
    hyperparams['override_attention_radius'] = args.override_attention_radius
    hyperparams['reconstruction'] = args.reconstruction
    hyperparams['traversal'] = args.traversal

    if hyperparams['reconstruction']:
        hyperparams['prediction_horizon'] = hyperparams['prediction_horizon'] + hyperparams['maximum_history_length']
    
    match_path = "/share/campbell/lyft/info"
    file_path = osp.join(match_path, 'matched_scenes.json')
    matches = json.load(open(file_path))
    matches = expand_matches(matches)
    hyperparams['matches'] = matches
    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % args.batch_size)
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
    print('| EE state_combine_method: %s' % args.edge_state_combine_method)
    print('| EIE scheme: %s' % args.edge_influence_combine_method)
    print('| dynamic_edges: %s' % args.dynamic_edges)
    print('| robot node: %s' % args.incl_robot_node)
    print('| edge_addition_filter: %s' % args.edge_addition_filter)
    print('| edge_removal_filter: %s' % args.edge_removal_filter)
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    print('-----------------------')

    log_writer = None
#     model_dir = None
#     model_dir = os.path.join(args.log_dir, 'models_28_Sep_2022_02_34_39prediction_anomaly')
#     model_dir = os.path.join(args.log_dir, 'models_04_Oct_2022_13_56_13prediction_anomaly_120')
#     model_dir = os.path.join(args.log_dir, 'models_28_Oct_2022_08_34_59non_traversal_120')
#     model_dir = os.path.join(args.log_dir, 'models_28_Oct_2022_01_57_54traversal_120')
#     folder_ = "models_28_Oct_2022_17_03_47non_traversal_gt"
#     folder_ = "models_31_Oct_2022_16_26_18_non_traversal"
#     folder_ = "models_12_Apr_2023_02_10_07prediction_ithaca365"
#     folder_ = 'models_14_Apr_2023_02_03_01prediction_ithaca365_100'
    folder_ = 'models_13_May_2023_02_50_27half_sec_corrected_ithaca365'
    model_dir = os.path.join(args.log_dir, folder_)
    log_writer = SummaryWriter(log_dir=model_dir+'eval')
#     if not args.debug:
#         # Create the log and model directiory if they're not present.
#         model_dir = os.path.join(args.log_dir,
#                                  'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
#         pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

#         # Save config to model directory
#         with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
#             json.dump(hyperparams, conf_json)

#         log_writer = SummaryWriter(log_dir=model_dir)

    eval_scenes = []
    eval_scenes_sample_probs = None
    
    eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
    with open(eval_data_path, 'rb') as f:
        eval_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    if eval_env.robot_type is None and hyperparams['incl_robot_node']:
        eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in eval_env.scenes:
            scene.add_robot_from_nodes(eval_env.robot_type)

    eval_scenes = eval_env.scenes
    eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

    eval_dataset = EnvironmentDataset(eval_env,
                                      hyperparams['state'],
                                      hyperparams['pred_state'],
                                      scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                      node_freq_mult=hyperparams['node_freq_mult_eval'],
                                      hyperparams=hyperparams,
                                      min_history_timesteps=hyperparams['minimum_history_length'],
                                      min_future_timesteps=hyperparams['prediction_horizon'],
                                      return_robot=not args.incl_robot_node)
    eval_data_loader = dict()
    for node_type_data_set in eval_dataset:
        if len(node_type_data_set) == 0:
            continue

        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.eval_device is 'cpu' else True,
                                                     batch_size=args.eval_batch_size,
                                                     shuffle=True,
                                                     num_workers=args.preprocess_workers)
        eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Loaded evaluation data from {eval_data_path}")

    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        if hyperparams['traversal']:
            print(f"Offline calculating scene traversal graphs")
#             match_path = "/share/campbell/lyft/info"
#             file_path = osp.join(match_path, 'matched_scenes.json')
#             matches = json.load(open(file_path))
#             matches = expand_matches(matches)
            
#             for i, scene in enumerate(train_scenes):
#                 matched_scene_names = matches[scene.name]
#                 matched_scenes = [s for s in train_scenes if s.name in matched_scene_names]
#                 scene.calculate_traversal_scene_graph(train_env.attention_radius,
#                                             hyperparams['edge_addition_filter'],
#                                             hyperparams['edge_removal_filter'], 
#                                             matched_scenes)
#                 print(f"Created Scene Traversal Graph for Training Scene {i}")

            for i, scene in enumerate(eval_scenes):
                matched_scene_names = matches[scene.name]
                matched_scenes = [s for s in eval_scenes if s.name in matched_scene_names]
                scene.calculate_traversal_scene_graph(eval_env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter'],
                                            matched_scenes)
                print(f"Created Scene Traversal Graph for Evaluation Scene {i}")
#                 G =scene.get_scene_graph(8, eval_env.attention_radius, hyperparams['edge_addition_filter'], hyperparams['edge_removal_filter'])
#                 nodes_ = G.nodes
#                 for node_ in nodes_:
#                     if node_.type == "VEHICLE":
#                         neighbors = G.get_neighbors(node_, node_.type)
#                         print(neighbors)
                
        else:
#             print(f"Offline calculating scene graphs")
#             for i, scene in enumerate(train_scenes):
#                 scene.calculate_scene_graph(train_env.attention_radius,
#                                             hyperparams['edge_addition_filter'],
#                                             hyperparams['edge_removal_filter'])
#                 print(f"Created Scene Graph for Training Scene {i}")

            for i, scene in enumerate(eval_scenes):
                scene.calculate_scene_graph(eval_env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")

#         for i, scene in enumerate(eval_scenes):
#             scene.calculate_scene_graph(eval_env.attention_radius,
#                                         hyperparams['edge_addition_filter'],
#                                         hyperparams['edge_removal_filter'])
#             print(f"Created Scene Graph for Evaluation Scene {i}")
    
    # load saved model
    model_registrar = ModelRegistrar(model_dir, args.device)
    model_registrar.load_models(iter_num=49)
    
    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            args.eval_device)

    trajectron.set_environment(eval_env)
    trajectron.set_annealing_params()
    print('Created Model.')
    
    epoch = 0
    max_hl = hyperparams['maximum_history_length']
    ph = hyperparams['prediction_horizon']
    model_registrar.to(args.eval_device)
    with torch.no_grad():
        # Calculate evaluation loss
#         for node_type, data_loader in eval_data_loader.items():
#             eval_loss = []
#             print(f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")
#             pbar = tqdm(data_loader, ncols=80)
#             for batch in pbar:
#                 eval_loss_node_type = trajectron.eval_loss(batch, node_type)
#                 pbar.set_description(f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}")
#                 eval_loss.append({node_type: {'nll': [eval_loss_node_type]}})
#                 del batch

#             evaluation.log_batch_errors(eval_loss,
#                                         log_writer,
#                                         f"{node_type}/eval_loss",
#                                         epoch)

        # Predict batch timesteps for evaluation dataset evaluation
        eval_batch_errors = []
        PF = 0.95
        alpha = 1-PF
        results_save_path = os.path.join(RESULTS_SAVE_PATH, folder_+'_510_910')
        for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
            if scene.name != 'c226366bd789ffbd1e38f8f276be9413':
                continue
            scene_dict = dict()
#             timesteps = scene.sample_timesteps(args.eval_batch_size)
            
#             timesteps = scene.sample_timesteps(scene.timesteps)
            timesteps = np.arange(510,910)
            
            
            if hyperparams['history_head']:
                predictions, mean_predictions, cov_predictions, y_t, log_pis_dict, y_dist,
                history_predictions, mean_r_predictions, cov_r_predictions, x_t, log_pis_r_dict, y_dist_r = trajectron.full_predict(scene,
                                                          timesteps,
                                                          ph,
                                                          num_samples=10,
                                                          min_future_timesteps=ph,
                                                          z_mode=False,
                                                          gmm_mode=False,
                                                          full_dist=True)
            else:
                predictions, mean_predictions, cov_predictions, y_t, log_pis_dict, y_dist  = trajectron.full_predict(scene,
                                                              timesteps,
                                                              ph,
                                                              num_samples=10,
                                                              min_future_timesteps=ph,
                                                              z_mode=False,
                                                              gmm_mode=False,
                                                              full_dist=True)

            mean_predictions = collections.OrderedDict(sorted(mean_predictions.items()))
            y_t = collections.OrderedDict(sorted(y_t.items()))
            predictions = collections.OrderedDict(sorted(predictions.items()))
            cov_predictions = collections.OrderedDict(sorted(cov_predictions.items()))
            log_pis = collections.OrderedDict(sorted(log_pis_dict.items()))
            
            if hyperparams['history_head']:
                mean_r_predictions = collections.OrderedDict(sorted(mean_r_predictions.items()))
                x_t = collections.OrderedDict(sorted(x_t.items()))
                history_predictions = collections.OrderedDict(sorted(history_predictions.items()))
                cov_r_predictions = collections.OrderedDict(sorted(cov_r_predictions.items()))
                log_pis_r = collections.OrderedDict(sorted(log_pis_r_dict.items()))
            
            scene_dict.update({'mean_predictions': mean_predictions})
            scene_dict.update({'y_t': y_t})
            scene_dict.update({'predictions': predictions})
            scene_dict.update({'cov_predictions': cov_predictions})
            scene_dict.update({'log_pis': log_pis})
            scene_dict.update({'scene': scene})
            scene_dict.update({'Scene_GMM': y_dist})
            
            if hyperparams['history_head']:
                scene_dict.update({'mean_r_predictions': mean_r_predictions})
                scene_dict.update({'x_t': x_t})
                scene_dict.update({'history_predictions': history_predictions})
                scene_dict.update({'cov_r_predictions': cov_r_predictions})
                scene_dict.update({'log_pis_r': log_pis_r})
                scene_dict.update({'Scene_GMM_r': y_dist_r})
            
            if not os.path.exists(results_save_path):
                os.makedirs(f'{results_save_path}')

            with open(os.path.join(results_save_path, scene.name), 'wb') as f:
                dill.dump(scene_dict, f, protocol=dill.HIGHEST_PROTOCOL)
            
            
            
            
            
            
            if False:
                fig, ax = plt.subplots(figsize=(6, 6))
                for t in y_t:
                    for node in y_t[t]:
                        test = m_dist[t][node]>=chi2.ppf(1-alpha, 2)
                        if any(test):
                            y = y_t[t][node]
                            p = mean_predictions[t][node]
                            cov = cov_predictions[t][node]
    #                         ax.plot(y[:,0],y[:,1],'k',label='gt')

    #                         ax.plot(p[:,0],p[:,1],'g', label='prediction')
    #                         ax.set_ylabel('East [m]')
    #                         ax.set_ylabel('North [m]')

    #                         ax.legend()
    #                         plt.savefig(f'./img/time_{t}_node_{node.__str__().split("/")[-1]}.png')
    #                         ax.clear() 
                            for i, anon in enumerate(test):
                                ellipse = confidence_ellipse(cov[i], p[i,0], p[i,1], ax, n_std=3.0, edgecolor='red')

    #                             if anon:
    #                                 ellipse = confidence_ellipse(cov[i], p[i,0], p[i,1], ax, n_std=3.0, edgecolor='red')

    #                             lower_index = np.clip(i-0, 0, 29)
    #                             upper_index = np.clip(i+2, 0, 29)
    #                             ax.plot(p[lower_index:upper_index,0], p[lower_index:upper_index,1], '*g', label='prediction')
    #                             ax.plot(y[lower_index:upper_index,0], y[lower_index:upper_index,1], '*k', label='gt')
                                ax.plot(y[i,0], y[i,1], '*k', label='gt')
                                if anon:
                                    ax.plot(p[i,0], p[i,1], 'r*')
                                else:
                                    ax.plot(p[i,0], p[i,1], 'g*')
                                ax.add_patch(ellipse)

    #                         ax.legend()
                            ax.set_ylabel('East [m]')
                            ax.set_ylabel('North [m]')
                            plt.savefig(f'./img/time_{t}_node_{node.__str__().split("/")[-1]}_cov.png')
                            ax.clear()
                        
                        
                        
                        
#             eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
#                                                                          scene.dt,
#                                                                          max_hl=max_hl,
#                                                                          ph=ph,
#                                                                          node_type_enum=eval_env.NodeType,
#                                                                          map=scene.map))

#         evaluation.log_batch_errors(eval_batch_errors,
#                                     log_writer,
#                                     'eval',
#                                     epoch,
#                                     bar_plot=['kde'],
#                                     box_plot=['ade', 'fde'])

        # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
#         eval_batch_errors_ml = []
#         for scene in tqdm(eval_scenes, desc='MM Evaluation', ncols=80):
#             timesteps = scene.sample_timesteps(scene.timesteps)

#             predictions, mean_predictions, cov_predictions = trajectron.full_predict(scene,
#                                                   timesteps,
#                                                   ph,
#                                                   num_samples=1,
#                                                   min_future_timesteps=ph,
#                                                   z_mode=True,
#                                                   gmm_mode=True,
#                                                   full_dist=False)

#             eval_batch_errors_ml.append(evaluation.compute_batch_statistics(predictions,
#                                                                             scene.dt,
#                                                                             max_hl=max_hl,
#                                                                             ph=ph,
#                                                                             map=scene.map,
#                                                                             node_type_enum=eval_env.NodeType,
#                                                                             kde=False))

#         evaluation.log_batch_errors(eval_batch_errors_ml,
#                                     log_writer,
#                                     'eval/ml',
#                                     epoch)



        


if __name__ == '__main__':
    main()
