import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

# 데이타로더
from torchvision import datasets, transforms



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    
    def loss_function(self, recon_x, x, mu, logvar): # im inside model so i pass the 'self'
        BCE = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def training_step(self, batch, batch_idx):
        '''
        original code에서 지운것 #로 표시
        
        data = data.to(device)                         # lt에서는 gpu cpu알아서(?)
        optimizer.zero_grad()                          # update optimize automatically
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()                                # do backward automatically
        train_loss += loss.item()
        optimizer.step()                               # update optimize automatically
        '''
        # model 안이기 때문에 self라고함(?)
        x,y = batch
        recon_batch, mu, logvar = self(x)          # 왜 self.forward가 아니지?
        loss = self.loss_function(recon_batch, x, mu, logvar)
        # loss.backward()  lightning에서는 이것도 자동으로 해줌.
        return {'loss':loss} # lightning에다가 optimize할 것을 알려줌
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default = 32)
    parser.add_argument("--cuda", default = False)
    args = parser.parse_args()
    kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('./data', train = True, download = False, transform = transforms.ToTensor(),)
                                #   batch_size = args.batch_size, shuffle = True, **kwargs)
                    )
    
    
    vae = VAE()
    trainer = pl.Trainer(fast_dev_run = True) # fast_dev_run : single batch for training loop, to check if they have any error
    trainer.fit(vae, train_dataloader = train_loader) # -> training step 만들라하네.
    print(vae(torch.rand(4,784)))
    
    

#     def validation_step(self, batch, batch_idx):
#         def save_image(self, data, filename):
#             img = data.clone().clamp(0, 255).numpy()
#             img = img[0].transpose(1, 2, 0)
#             img = Image.fromarray(img, mode='RGB')
#             img.save(filename)
            
#         '''
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#                 save_image(comparison.cpu(),
#                          'results/reconstruction_' + str(epoch) + '.png', nrow=n)
#         '''
        
#         x,y = batch
#         recon_batch, mu, logvar = self(x)          # 왜 self.encode가 아니지?
#         val_loss = self.loss_function(recon_batch, x, mu, logvar).item()

#         if batch_idx == 0:
            
        
#         return {'loss':loss} # lt에다가 optimize할 것을 알려줌    

#     # 지금까지 optimizer을 추가하지 않았음. Let's add
#     def configure_optimizer(self):
        
#         return torch.optim.Adam(self.parameters(), lr = le-3)
    
    
# if __name__ == '__main__':
#     from argparse import ArgumentParser

#     parser = ArgumentParser()
#     parser = pl.Trainer.add_argparse_args(parser)
#     parser.add_argument('--batch_size', default=32, type=int)
#     parser.add_argument('--learning_rate', default=1e-3, type=float)

#     args = parser.parse_args()
#     kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    
#     ### 원본
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                        transform=transforms.ToTensor()),
#         batch_size=args.batch_size, shuffle=True, **kwargs)
#     val_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#         batch_size=args.batch_size, shuffle=True, **kwargs)    
    

#     vae = VAE(hparams=args)
#     # trainer = pl.Trainer(fast_dev_run = True) # one epoch
#     trainer = pl.Trainer() # full run
#     trainer.fit(vae, train_dataloader = train_loader, val_dataloader = val_loader)