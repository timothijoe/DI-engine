import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb
from torch.distributions import Normal, Independent
# import torch.distributions.multivariate_normal.MultivariateNormal as MultivariateNormal
from easydict import EasyDict
def kl_divergence_with_unit_gaussian(mu, sigma):
	"""Here self=q and other=p and we compute KL(q, p)"""
	KL_Div = - torch.mean(torch.sum((1 + torch.log(sigma**2) - mu**2 - sigma**2), 1) / 2, 0)
	return  KL_Div

def KL_divergence_between_two_gaussian(last_latent_mu, last_latent_sigma, current_latent_mu, current_latent_sigma):
	return torch.mean((torch.log((last_latent_sigma / (current_latent_sigma + 1e-9))**2) + (current_latent_sigma / (last_latent_sigma + 1e-9))**2 + ((last_latent_mu - current_latent_mu) / (last_latent_sigma + 1e-9))**2 - 1) * 1/2, 1)


class hyper_parameter(object):
    def __init__(self, args):

        # dataset parameter
        self.batch_size = args.batch_size
        self.split_ratio = 0.8
        # 1 mean [0], 2 means [0.05], 3 means [0.1], 4 means [0, 0.05, 0.1]
        if args.noise_mode == 1:
            self.noise_lst = [0]
        elif args.noise_mode == 2:
            self.noise_lst = [0.05]
        elif args.noise_mode == 3:
            self.noise_lst = [0.1]
        elif args.noise_mode == 4:
            self.noise_lst = [0, 0.05, 0.1]
        # self.track_lst = args.track_lst
        self.seed = 1
        self.restore_epoch = 0
        # training parameter
        self.train_the_model = args.train_the_model
        self.evaluate_model = args.evaluate_model
        self.roll_out_test = args.roll_out_test
        self.visualize_data_distribution = args.visualize_data_distribution
        self.restore_model = args.restore_model
        self.restore_epoch = args.restore_epoch
        self.n_epochs = args.n_epochs
        self.print_every_epoch = 1
        self.learning_rate = 0.001
        self.adam_weight_decay = 5e-4
        self.lr_decay_step_size = 5 
        self.lr_decay_gamma = 0.6
        self.latent_compact_beta = 1
        self.latent_continuous_beta = 0.5
        self.model_name = 'Naive_Model'
        self.load_model_name = 'Naive_Model'
        self.exp_name = 'christmas105'
        self.test = True
        self.embedding_dim = 64
        self.h_dim = 64
        self.latent_dim = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128 
        self.seq_len = 30
        self.use_relative_pos = True
        self.kld_weight = 0.01
        self.fde_weight = 0.1
        self.cum_theta_weight = 1



        self.val_freq = 1



class VAELSTM(nn.Module):
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        embedding_dim = 64,
        h_dim = 64,
        latent_dim = 100,
        seq_len = 30,
        use_relative_pos = True,
        kld_weight = 0.01,
        fde_weight = 0.1,
        dt = 0.03,
        )
    def __init__(self):
        self._cfg = EasyDict(self.config)
        super(VAELSTM, self).__init__()
        self.embedding_dim = self._cfg.embedding_dim
        self.h_dim = self._cfg.h_dim 
        self.num_layers = 1
        self.latent_dim = self._cfg.latent_dim
        self.seq_len = self._cfg.seq_len 
        self.use_relative_pos = self._cfg.use_relative_pos
        self.kld_weight = self._cfg.kld_weight
        self.fde_weight = self._cfg.fde_weight
        self.dt = self._cfg.dt


        # input: x, y, v,   output: embedding
        self.spatial_embedding = nn.Linear(4, self.embedding_dim)

        # input: h_dim, output: x,y,v
        self.hidden2control = nn.Linear(self.h_dim, 2)
        dec_mid_dims = [self.h_dim, self.h_dim, self.h_dim, 3]
        modules = []
        in_channels = self.h_dim

        for m_dim in dec_mid_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = m_dim        
        # self.hidden2pos = nn.Sequential(*modules)

        enc_mid_dims = [self.h_dim, self.h_dim, self.h_dim, self.latent_dim]
        mu_modules = []
        sigma_modules = []
        in_channels = self.h_dim 
        for m_dim in enc_mid_dims:
            mu_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            sigma_modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, m_dim),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = m_dim  
        self.mean = nn.Sequential(*mu_modules) 
        self.log_var = nn.Sequential(*sigma_modules)
        self.encoder = nn.LSTM(self.embedding_dim, self.h_dim, self.num_layers)
        self.decoder = nn.LSTM(self.embedding_dim, self.h_dim, self.num_layers)
        self.init_hidden_decoder = torch.nn.Linear(in_features = self.latent_dim, out_features = self.h_dim * self.num_layers)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.h_dim).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.h_dim).to(self.device)
        )

    def get_relative_position(self, abs_traj):
        # abs_traj shape: batch_size x seq_len x 3
        # rel traj shape: batch_size x seq_len -1 x 2
        rel_traj = abs_traj[:, 1:, :2] - abs_traj[:, :-1, :2]
        rel_traj = torch.cat([abs_traj[:, 0, :2].unsqueeze(1), rel_traj], dim = 1)
        rel_traj = torch.cat([rel_traj, abs_traj[:,:,2:]],dim=2)
        #rel_traj = torch.cat([rel_traj, abs_traj[:,:,2:].unsqueeze(2)],dim=2)
        # rel_traj shape: batch_size x seq_len x 3
        return rel_traj
    
    def encode(self, input):
        # input meaning: a trajectory len 25 and contains x, y , v
        # input shape: batch x seq_len x 3
        #data_traj shape: seq_len x batch x 3
        if self.use_relative_pos:
            input = self.get_relative_position(input)
        data_traj = input.permute(1, 0, 2).contiguous()
        traj_embedding = self.spatial_embedding(data_traj.view(-1, 4))
        traj_embedding = traj_embedding.view(self.seq_len, -1, self.embedding_dim)
        # Here we do not specify batch_size to self.batch_size because when testing maybe batch will vary
        batch_size = traj_embedding.shape[1]
        hidden_tuple = self.init_hidden(batch_size)
        output, encoder_h = self.encoder(traj_embedding, hidden_tuple)
        mu = self.mean(encoder_h[0])
        log_var = self.log_var(encoder_h[0])
        #mu, log_var = torch.tanh(mu), torch.tanh(log_var)
        return mu, log_var

    def decode(self, z, init_state):
        generated_traj = []
        prev_state = init_state 
        # decoder_input shape: batch_size x 3
        decoder_input = self.spatial_embedding(prev_state)
        decoder_input = decoder_input.view(1, -1 , self.embedding_dim)
        decoder_h = self.init_hidden_decoder(z)
        if len(decoder_h.shape) == 2:
            decoder_h = torch.unsqueeze(decoder_h, 0)
            #decoder_h.unsqueeze(0)
        decoder_h = (decoder_h, decoder_h)
        for _ in range(self.seq_len):
            # output shape: 1 x batch x h_dim
            output, decoder_h = self.decoder(decoder_input, decoder_h)
            # rel_state shape: batch x 3
            control = self.hidden2control(output.view(-1, self.h_dim))
            curr_state = self.plant_model_batch(prev_state, control[:,0], control[:,1], self.dt)
            generated_traj.append(curr_state)
            decoder_input = self.spatial_embedding(curr_state)
            decoder_input = decoder_input.view(1, -1, self.embedding_dim)
            prev_state = curr_state 
        generated_traj = torch.stack(generated_traj, dim = 1)
        return generated_traj

    def plant_model_batch(self, prev_state_batch, pedal_batch, steering_batch, dt = 0.03):
        #import copy
        prev_state = prev_state_batch
        x_t = prev_state[:,0]
        y_t = prev_state[:,1]
        psi_t = prev_state[:,2]
        v_t = prev_state[:,3]
        #pedal_batch = torch.clamp(pedal_batch, -5, 5)
        steering_batch = torch.clamp(steering_batch, -0.5, 0.5)
        beta = steering_batch
        a_t = pedal_batch
        v_t_1 = v_t + a_t * dt 
        v_t_1 = torch.clamp(v_t_1, 0, 30)
        x_dot = v_t * torch.cos(psi_t)
        y_dot = v_t * torch.sin(psi_t)
        psi_dot = v_t * torch.tan(beta) / 2.5
        psi_dot = torch.clamp(psi_dot, -3.14 /2,3.14 /2)
        x_t_1 = x_dot * dt + x_t 
        y_t_1 = y_dot * dt + y_t
        psi_t_1 = psi_dot*dt + psi_t 
        #psi_t = self.wrap_angle_rad(psi_t)
        current_state = torch.stack([x_t_1, y_t_1, psi_t_1, v_t_1], dim = 1)
        #current_state = torch.FloatTensor([x_t, y_t, psi_t, v_t_1])
        return current_state

    def reparameterize(self, mu, logvar):
        # mu shape: batch size x latent_dim
        # sigma shape: batch_size x latent_dim
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
        #return mu

    def forward(self, expert_traj, init_state):
        mu, log_var = self.encode(expert_traj)
        z = self.reparameterize(mu, log_var)
        z = torch.tanh(z)
        recons_traj = self.decode(z, init_state)
        #recons_traj = recons_traj[:,:,[0,1,3]]
        return [recons_traj, expert_traj, mu.squeeze(0), log_var.squeeze(0)]
    
    def loss_function(self, *args):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        epoch = 0
        if len(args) > 4:
            epoch = args[4]
        kld_weight = self.kld_weight
        recon_loss = 0
        # reconstruction loss
        recons_loss = F.mse_loss(recons[:,:,:2], input[:,:,:2])
        #recons_loss += F.mse_loss(recons[:,:,3], input[:,:,3]) * 0.01
        vel_loss = F.mse_loss(recons[:,:,3], input[:,:,3]) * 0.01
        #final displacement loss
        final_displacement_error = F.mse_loss(recons[:,-1, :2], input[:, -1, :2])
        theta_error = F.mse_loss(recons[:,:,2], input[:,:,2]) * 0.01
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_weight = 0.01
        loss = recons_loss  + kld_weight * kld_loss + self.fde_weight * final_displacement_error + theta_error  + vel_loss
        # print('kld_weight: {}'.format(kld_weight))
        # print('epoch: {} '.format(epoch))
        return {'loss': loss, "reconstruction_loss": recons_loss, 'KLD': kld_loss, 'final_displacement_error' : final_displacement_error, 
        'theta_error':theta_error, 'mu':mu[0][0], 'log_var': log_var[0][0]}

    def sample(self, num_samples: int, init_state):
        z = torch.randn(1, num_samples, self.latent_dim)
        z = z.to(self.device)
        samples = self.decode(z, init_state)
        return samples

    def generate(self, x, init_state):
        return self.forward(x, init_state)[0]