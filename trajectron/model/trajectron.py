import torch
import numpy as np
from model.mgcvae import MultimodalGenerativeCVAE
from model.dataset import get_timesteps_data, restore
from scipy.spatial import distance

class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAE(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node_str, model in self.node_models_dict.items():
            model.set_annealing_params()

    def step_annealers(self, node_type=None):
        if node_type is None:
            for node_type in self.node_models_dict:
                self.node_models_dict[node_type].step_annealers()
        else:
            self.node_models_dict[node_type].step_annealers()

    def train_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph)

        return loss

    def eval_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        nll = model.eval_loss(inputs=x,
                              inputs_st=x_st_t,
                              first_history_indices=first_history_index,
                              labels=y,
                              labels_st=y_st_t,
                              neighbors=restore(neighbors_data_st),
                              neighbors_edge_value=restore(neighbors_edge_value),
                              robot=robot_traj_st_t,
                              map=map,
                              prediction_horizon=self.ph)

        return nll.cpu().detach().numpy()

    def predict(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):

        predictions_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            predictions = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep)

            predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict

    def full_predict(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):

        predictions_dict = {}
        mean_predictions_dict = {}
        cov_predictions_dict = {}
        m_dist_dict = {}
        log_pis_dict = {}
        y_dict = {}
        corrs_dict = {}
        log_sigmas_dict = {}
        x_dict = dict()
        if self.hyperparams['history_head']:
            history_predictions_dict = dict()
            mean_r_predictions_dict = dict()
            cov_r_predictions_dict = dict()
#             x_dict = dict()
            log_pis_r_dict = dict()
            corrs_r_dict = dict()
            log_sigmas_r_dict = dict()
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            if self.hyperparams['history_head']:
                y_dist, predictions, y_dist_r, history_predictions = model.full_predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep)
            else:
                y_dist, predictions = model.full_predict(inputs=x,
                                            inputs_st=x_st_t,
                                            first_history_indices=first_history_index,
                                            neighbors=neighbors_data_st,
                                            neighbors_edge_value=neighbors_edge_value,
                                            robot=robot_traj_st_t,
                                            map=map,
                                            prediction_horizon=ph,
                                            num_samples=num_samples,
                                            z_mode=z_mode,
                                            gmm_mode=gmm_mode,
                                            full_dist=full_dist,
                                            all_z_sep=all_z_sep)
            
            
            predictions_np = predictions.cpu().detach().numpy()
            mus = y_dist.mus.cpu().detach().numpy()
            log_sigmas = y_dist.sigmas.cpu().detach().numpy()
            cov_matrix = y_dist.get_covariance_matrix().cpu().detach().numpy()
            y_t = y_t.cpu().detach().numpy()
            log_pis = y_dist.log_pis.cpu().detach().numpy()
            corrs = y_dist.corrs.cpu().detach().numpy()
            x_t = torch.flip(x_t[:, :-1, 0:2], [1]).cpu().detach().numpy()
            if self.hyperparams['history_head']:
#                 x_t = torch.flip(x_t[:, :-1, 0:2], [1]).cpu().detach().numpy()
                history_predictions_np = history_predictions.cpu().detach().numpy()
                mus_r = y_dist_r.mus.cpu().detach().numpy()
                log_sigmas_r = y_dist_r.sigmas.cpu().detach().numpy()
                cov_matrix_r = y_dist_r.get_covariance_matrix().cpu().detach().numpy()
                log_pis_r = y_dist_r.log_pis.cpu().detach().numpy()
                corrs_r = y_dist_r.corrs.cpu().detach().numpy()
            
            # Assign predictions to node
            
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                    mean_predictions_dict[ts] = dict()
                    cov_predictions_dict[ts] = dict()
                    y_dict[ts] = dict()
                    log_pis_dict[ts] = dict()
                    corrs_dict[ts] = dict()
                    log_sigmas_dict[ts] = dict()
                    x_dict[ts] = dict()
                    if self.hyperparams['history_head']:
                        history_predictions_dict[ts] = dict()
                        mean_r_predictions_dict[ts] = dict()
                        cov_r_predictions_dict[ts] = dict()
#                       x_dict[ts] = dict()
                        log_pis_r_dict[ts] = dict()
                        corrs_r_dict[ts] = dict()
                        log_sigmas_r_dict[ts] = dict()
                        
                predictions = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
                mean_predictions = np.squeeze(np.transpose(mus[:, [i]], (1, 0, 2, 3, 4)))
                cov_predictions = np.squeeze(np.transpose(cov_matrix[:, [i]], (1, 0, 2, 3, 4, 5)))
                log_pis_predictions = np.squeeze(np.transpose(log_pis[:, [i]], (1, 0, 2, 3)))
                corrs_predictions = np.squeeze(np.transpose(corrs[:, [i]], (1, 0, 2, 3)))
                log_sigmas_predictions = np.squeeze(np.transpose(log_sigmas[:, [i]], (1, 0, 2, 3, 4)))
                y = np.squeeze(y_t[i])
                x = np.squeeze(x_t[i])
                if self.hyperparams['history_head']:
                    history_predictions = np.transpose(history_predictions_np[:, [i]], (1, 0, 2, 3))
                    mean_r_predictions = np.squeeze(np.transpose(mus_r[:, [i]], (1, 0, 2, 3, 4)))
                    cov_r_predictions = np.squeeze(np.transpose(cov_matrix_r[:, [i]], (1, 0, 2, 3, 4, 5)))
                    log_pis_r_predictions = np.squeeze(np.transpose(log_pis_r[:, [i]], (1, 0, 2, 3)))
                    corrs_r_predictions = np.squeeze(np.transpose(corrs_r[:, [i]], (1, 0, 2, 3)))
                    log_sigmas_r_predictions = np.squeeze(np.transpose(log_sigmas_r[:, [i]], (1, 0, 2, 3, 4)))
                    
#                     x = np.squeeze(x_t[i])
                    
                    
                predictions_dict[ts][nodes[i]] = predictions
                mean_predictions_dict[ts][nodes[i]] = mean_predictions
                cov_predictions_dict[ts][nodes[i]] = cov_predictions
                y_dict[ts][nodes[i]] = y
                log_pis_dict[ts][nodes[i]] = log_pis_predictions
                corrs_dict[ts][nodes[i]] = corrs_predictions
                log_sigmas_dict[ts][nodes[i]] = log_sigmas_predictions
                x_dict[ts][nodes[i]] = x
                if self.hyperparams['history_head']:
                    history_predictions_dict[ts][nodes[i]] = history_predictions
                    mean_r_predictions_dict[ts][nodes[i]] = mean_r_predictions
                    cov_r_predictions_dict[ts][nodes[i]] = cov_r_predictions
#                     x_dict[ts][nodes[i]] = x
                    log_pis_r_dict[ts][nodes[i]] = log_pis_r_predictions
                    corrs_r_dict[ts][nodes[i]] = corrs_r_predictions
                    log_sigmas_r_dict[ts][nodes[i]] = log_sigmas_r_predictions
                    
        prediction_returns = {'predictions_dict': predictions_dict,
                              'mean_predictions_dict': mean_predictions_dict,
                              'cov_predictions_dict': cov_predictions_dict,
                              'log_pis_dict': log_pis_dict,
                              'corrs_dict': corrs_dict,
                              'log_sigmas_dict': log_sigmas_dict,
                              'y_dist': y_dist}
        

        gt_returns = {'y_dict': y_dict,
                      'x_dict': x_dict}

        
        
        if self.hyperparams['history_head']:
            reconstruction_returns = {'history_predictions': history_predictions,
                                      'mean_r_predictions_dict': mean_r_predictions_dict,
                                      'cov_r_predictions_dict': cov_r_predictions_dict,
                                      'log_pis_r_dict': log_pis_r_dict,
                                      'corrs_r_dict': corrs_r_dict,
                                      'log_sigmas_r_dict': log_sigmas_r_dict,
                                      'y_dist_r': y_dist_r}
            
            return prediction_returns, gt_returns, reconstruction_returns
        
        return prediction_returns, gt_returns, None