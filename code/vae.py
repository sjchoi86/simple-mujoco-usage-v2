import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from util import ikdpp,np2torch,torch2np

class ConditionalVariationalAutoEncoderClass(nn.Module):
    def __init__(
        self,
        name     = 'CVAE',              
        x_dim    = 784,              # input dimension
        c_dim    = 10,               # condition dimension
        z_dim    = 16,               # latent dimension
        h_dims   = [64,32],          # hidden dimensions of encoder (and decoder)
        actv_enc = nn.LeakyReLU(),   # encoder activation
        actv_dec = nn.LeakyReLU(),   # decoder activation
        actv_out = None,             # output activation
        var_max  = None,             # maximum variance
        device   = 'cpu'
        ):
        """
            Initialize
        """
        super(ConditionalVariationalAutoEncoderClass,self).__init__()
        self.name = name
        self.x_dim    = x_dim
        self.c_dim    = c_dim
        self.z_dim    = z_dim
        self.h_dims   = h_dims
        self.actv_enc = actv_enc
        self.actv_dec = actv_dec
        self.actv_out = actv_out
        self.var_max  = var_max
        self.device   = device
        # Initialize layers
        self.init_layers()
        self.init_params()
        # Test
        self.test()
        
    def init_layers(self):
        """
            Initialize layers
        """
        self.layers = {}
        
        # Encoder part
        h_dim_prev = self.x_dim + self.c_dim
        for h_idx,h_dim in enumerate(self.h_dims):
            self.layers['enc_%02d_lin'%(h_idx)]  = \
                nn.Linear(h_dim_prev,h_dim,bias=True)
            self.layers['enc_%02d_actv'%(h_idx)] = \
                self.actv_enc
            h_dim_prev = h_dim
        self.layers['z_mu_lin']  = nn.Linear(h_dim_prev,self.z_dim,bias=True)
        self.layers['z_var_lin'] = nn.Linear(h_dim_prev,self.z_dim,bias=True)
        
        # Decoder part
        h_dim_prev = self.z_dim + self.c_dim
        for h_idx,h_dim in enumerate(self.h_dims[::-1]):
            self.layers['dec_%02d_lin'%(h_idx)]  = \
                nn.Linear(h_dim_prev,h_dim,bias=True)
            self.layers['dec_%02d_actv'%(h_idx)] = \
                self.actv_dec
            h_dim_prev = h_dim
        self.layers['out_lin'] = nn.Linear(h_dim_prev,self.x_dim,bias=True)
        
        # Append parameters
        self.param_dict = {}
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                self.param_dict[key+'_w'] = layer.weight
                self.param_dict[key+'_b'] = layer.bias
        self.cvae_parameters = nn.ParameterDict(self.param_dict)
        
    def xc_to_z_mu(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x and c to z_mu
        """
        if c is not None:
            net = torch.cat((x,c),dim=1)
        else:
            net = x
        for h_idx,_ in enumerate(self.h_dims):
            net = self.layers['enc_%02d_lin'%(h_idx)](net)
            net = self.layers['enc_%02d_actv'%(h_idx)](net)
        z_mu = self.layers['z_mu_lin'](net)
        return z_mu
    
    def xc_to_z_var(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x and c to z_var
        """
        if c is not None:
            net = torch.cat((x,c),dim=1)
        else:
            net = x
        for h_idx,_ in enumerate(self.h_dims):
            net = self.layers['enc_%02d_lin'%(h_idx)](net)
            net = self.layers['enc_%02d_actv'%(h_idx)](net)
        net = self.layers['z_var_lin'](net)
        if self.var_max is None:
            net = torch.exp(net)
        else:
            net = self.var_max*torch.sigmoid(net)
        z_var = net
        return z_var
    
    def zc_to_x_recon(
        self,
        z = torch.randn(2,16),
        c = torch.randn(2,10)
        ):
        """
            z and c to x_recon
        """
        if c is not None:
            net = torch.cat((z,c),dim=1)
        else:
            net = z
        for h_idx,_ in enumerate(self.h_dims[::-1]):
            net = self.layers['dec_%02d_lin'%(h_idx)](net)
            net = self.layers['dec_%02d_actv'%(h_idx)](net)
        net = self.layers['out_lin'](net)
        if self.actv_out is not None:
            net = self.actv_out(net)
        x_recon = net
        return x_recon
    
    def xc_to_z_sample(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            x and c to z_sample
        """
        z_mu,z_var = self.xc_to_z_mu(x=x,c=c),self.xc_to_z_var(x=x,c=c)
        eps_sample = torch.randn(
            size=z_mu.shape,dtype=torch.float32).to(self.device)
        z_sample   = z_mu + torch.sqrt(z_var+1e-10)*eps_sample
        return z_sample
    
    def xc_to_x_recon(
        self,
        x             = torch.randn(2,784),
        c             = torch.randn(2,10), 
        STOCHASTICITY = True
        ):
        """
            x and c to x_recon
        """
        if STOCHASTICITY:
            z_sample = self.xc_to_z_sample(x=x,c=c)
        else:
            z_sample = self.xc_to_z_mu(x=x,c=c)
        x_recon = self.zc_to_x_recon(z=z_sample,c=c)
        return x_recon
    
    def sample_x(
        self,
        c        = torch.randn(5,10),
        n_sample = 5
        ):
        """
            Sample x
        """
        z_sample = torch.randn(
            size=(n_sample,self.z_dim),dtype=torch.float32).to(self.device)
        return self.zc_to_x_recon(z=z_sample,c=c),z_sample
    
    def init_params(self):
        """
            Initialize parameters
        """
        for key in self.layers.keys():
            layer = self.layers[key]
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,mean=0.0,std=0.01)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.constant_(layer.weight,1.0)
                nn.init.constant_(layer.bias,0.0)
            elif isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
                
    def test(
        self,
        batch_size = 4
        ):
        """
            Unit tests
        """
        x_test   = torch.randn(batch_size,self.x_dim)
        if self.c_dim > 0:
            c_test = torch.randn(batch_size,self.c_dim)
        else:
            c_test = None
        z_test   = torch.randn(batch_size,self.z_dim)
        z_mu     = self.xc_to_z_mu(x=x_test,c=c_test)
        z_var    = self.xc_to_z_var(x=x_test,c=c_test)
        x_recon  = self.zc_to_x_recon(z=z_test,c=c_test)
        z_sample = self.xc_to_z_sample(x=x_test,c=c_test)
        x_recon  = self.xc_to_x_recon(x=x_test,c=c_test)
    
    def loss_recon(
        self,
        x               = torch.randn(2,784),
        c               = torch.randn(2,10),
        LOSS_TYPE       = 'L1',
        recon_loss_gain = 1.0,
        STOCHASTICITY   = True
        ):
        """
            Recon loss
        """
        x_recon = self.xc_to_x_recon(x=x,c=c,STOCHASTICITY=STOCHASTICITY)
        if (LOSS_TYPE == 'L1') or (LOSS_TYPE == 'MAE'):
            errs = torch.mean(torch.abs(x-x_recon),axis=1)
        elif (LOSS_TYPE == 'L2') or (LOSS_TYPE == 'MSE'):
            errs = torch.mean(torch.square(x-x_recon),axis=1)
        elif (LOSS_TYPE == 'L1+L2') or (LOSS_TYPE == 'EN'):
            errs = torch.mean(
                0.5*(torch.abs(x-x_recon)+torch.square(x-x_recon)),axis=1)
        else:
            raise Exception("VAE:[%s] Unknown loss_type:[%s]"%
                            (self.name,LOSS_TYPE))
        return recon_loss_gain*torch.mean(errs)
    
    def loss_kl(
        self,
        x = torch.randn(2,784),
        c = torch.randn(2,10)
        ):
        """
            KLD loss
        """
        z_mu     = self.xc_to_z_mu(x=x,c=c)
        z_var    = self.xc_to_z_var(x=x,c=c)
        z_logvar = torch.log(z_var)
        errs     = 0.5*torch.sum(z_var + z_mu**2 - 1.0 - z_logvar,axis=1)
        return torch.mean(errs)
        
    def loss_total(
        self,
        x               = torch.randn(2,784),
        c               = torch.randn(2,10),
        LOSS_TYPE       = 'L1+L2',
        recon_loss_gain = 1.0,
        STOCHASTICITY   = True,
        beta            = 1.0
        ):
        """
            Total loss
        """
        loss_recon_out = self.loss_recon(
            x               = x,
            c               = c,
            LOSS_TYPE       = LOSS_TYPE,
            recon_loss_gain = recon_loss_gain,
            STOCHASTICITY   = STOCHASTICITY
        )
        loss_kl_out    = beta*self.loss_kl(x=x,c=c)
        loss_total_out = loss_recon_out + loss_kl_out
        info           = {'loss_recon_out' : loss_recon_out,
                          'loss_kl_out'    : loss_kl_out,
                          'loss_total_out' : loss_total_out,
                          'beta'           : beta}
        return loss_total_out,info
    
    def debug_plot_img(
        self,
        x_train_np     = np.zeros((60000,784)),  # to plot encoded latent space 
        y_train_np     = np.zeros((60000)),      # to plot encoded latent space 
        c_train_np     = np.zeros((60000,10)),   # to plot encoded latent space
        x_test_np      = np.zeros((10000,784)),
        c_test_np      = np.zeros((10000,10)),
        c_vecs         = np.eye(10,10),
        n_sample       = 10,
        img_shape      = (28,28),
        img_cmap       = 'gray',
        figsize_image  = (10,3.25),
        figsize_latent = (10,3.25),
        DPP_GEN        = False,
        dpp_hyp        = {'g':1.0,'l':0.1}
        ):
        """
            Debug plot
        """
        n_train            = x_train_np.shape[0]
        z_prior_np         = np.random.randn(n_train,self.z_dim)
        x_train_torch      = np2torch(x_train_np)
        c_train_torch      = np2torch(c_train_np)
        z_mu_train_np      = torch2np(self.xc_to_z_mu(x_train_torch,c_train_torch))
        z_sample_train_out = torch2np(
            self.xc_to_z_sample(x_train_torch,c_train_torch))
        
        # Reconstruct
        x_test_torch       = np2torch(x_test_np)
        c_test_torch       = np2torch(c_test_np)
        n_test             = x_test_np.shape[0]
        rand_idxs          = np.random.permutation(n_test)[:n_sample]
        if self.c_dim > 0:
            x_recon = self.xc_to_x_recon(
                x = x_test_torch[rand_idxs,:],
                c = c_test_torch[rand_idxs,:]).detach().cpu().numpy()
        else:
            x_recon = self.xc_to_x_recon(
                x = x_test_torch[rand_idxs,:],
                c = None).detach().cpu().numpy()
        
        # Plot images to reconstruct
        fig = plt.figure(figsize=figsize_image)
        for s_idx in range(n_sample):
            plt.subplot(1,n_sample,s_idx+1)
            plt.imshow(x_test_np[rand_idxs[s_idx],:].reshape(img_shape),
                       vmin=0,vmax=1,cmap=img_cmap)
            plt.axis('off')
        fig.suptitle("Images to Reconstruct",fontsize=15);plt.show()
        
        # Plot reconstructed images
        fig = plt.figure(figsize=figsize_image)
        for s_idx in range(n_sample):
            plt.subplot(1,n_sample,s_idx+1)
            plt.imshow(x_recon[s_idx,:].reshape(img_shape),
                       vmin=0,vmax=1,cmap=img_cmap)
            plt.axis('off')
        fig.suptitle("Reconstructed Images",fontsize=15);plt.show()
        
        # Plot conditioned generated images
        if DPP_GEN:
            n_sample_total = 100
            z_sample_total = np.random.randn(n_sample_total,self.z_dim)
            z_sample,_ = ikdpp(
                xs_total = z_sample_total,
                qs_total = None,
                n_select = n_sample,
                n_trunc  = n_sample_total,
                hyp      = dpp_hyp)
        else:
            z_sample = np.random.randn(n_sample,self.z_dim)
        z_sample_torch = np2torch(z_sample)
        c_vecs_torch   = np2torch(c_vecs)
        
        # Plot (conditioned) generated images
        if self.c_dim > 0:
            for r_idx in range(c_vecs.shape[0]):
                c_torch  = c_vecs_torch[r_idx,:].reshape((1,-1))
                c_np     = c_vecs[r_idx,:]
                fig      = plt.figure(figsize=figsize_image)
                for s_idx in range(n_sample):
                    z_torch = z_sample_torch[s_idx,:].reshape((1,-1))
                    x_recon = self.zc_to_x_recon(z=z_torch,c=c_torch)
                    x_reocn_np = torch2np(x_recon)
                    plt.subplot(1,n_sample,s_idx+1)
                    plt.imshow(x_reocn_np.reshape(img_shape),vmin=0,vmax=1,cmap=img_cmap)
                    plt.axis('off')
                fig.suptitle("Conditioned Generated Images c:%s"%
                             (c_np),fontsize=15);plt.show()
        else:
            fig = plt.figure(figsize=figsize_image)
            for s_idx in range(n_sample):
                z_torch = z_sample_torch[s_idx,:].reshape((1,-1))
                x_recon = self.zc_to_x_recon(z=z_torch,c=None)
                x_reocn_np = torch2np(x_recon)
                plt.subplot(1,n_sample,s_idx+1)
                plt.imshow(x_reocn_np.reshape(img_shape),vmin=0,vmax=1,cmap=img_cmap)
                plt.axis('off')
            fig.suptitle("Generated Images",fontsize=15);plt.show()
            
        # Plot latent space of training inputs
        fig = plt.figure(figsize=figsize_latent)
        # Plot samples from the prior
        plt.subplot(1,3,1) # z prior
        plt.scatter(z_prior_np[:,0],z_prior_np[:,1],marker='.',s=0.5,c='k',alpha=0.5)
        plt.title('z prior',fontsize=13)
        plt.xlim(-3,+3); plt.ylim(-3,+3)
        plt.gca().set_aspect('equal', adjustable='box')
        # Plot encoded mean
        plt.subplot(1,3,2) # z mu
        plt.scatter(
            x      = z_mu_train_np[:,0],
            y      = z_mu_train_np[:,1],
            c      = y_train_np,
            cmap   = 'rainbow',
            marker = '.',
            s      = 0.5,
            alpha  = 0.5)
        plt.title('z mu',fontsize=13)
        plt.xlim(-3,+3); plt.ylim(-3,+3)
        plt.gca().set_aspect('equal', adjustable='box')
        # Plot samples
        plt.subplot(1,3,3) # z sample
        sc = plt.scatter(
            x      = z_sample_train_out[:,0],
            y      = z_sample_train_out[:,1],
            c      = y_train_np,
            cmap   = 'rainbow',
            marker = '.',
            s      = 0.5,
            alpha  = 0.5)
        plt.plot(z_sample[:,0],z_sample[:,1],'o',mec='k',mfc='none',ms=10)
        colors = [plt.cm.rainbow(a) for a in np.linspace(0.0,1.0,10)]
        for c_idx in range(10):
            plt.plot(10,10,'o',mfc=colors[c_idx],mec=colors[c_idx],ms=6,
                     label='%d'%(c_idx)) # outside the axis, only for legend
        plt.legend(fontsize=8,loc='upper right')
        plt.title('z sample',fontsize=13)
        plt.xlim(-3,+3); plt.ylim(-3,+3)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show() # plot latent spaces
        
print ("Done.")