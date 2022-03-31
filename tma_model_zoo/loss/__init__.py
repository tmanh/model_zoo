import os

from .adversarial import *
from .basics import *
from .synthesis import *


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super().__init__()
        print('Preparing loss function:')

        self.n_gpus = args.n_gpus
        self.batch_size = args.batch_size

        self.generate_loss_modules(args)

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.log = torch.zeros(len(self.loss), device=device)
        self.loss_module.to(device)

        if not args.cpu and args.n_gpus > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(args.n_gpus))

        if args.load != '.':
            self.load(ckp.dir, cpu=args.cpu)

    def generate_loss_modules(self, args):
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'Fusion':
                loss_function = SynthesisLoss(args.mode)
            elif loss_type == 'LIP':
                module = import_module('src.loss.DataFusion.Basic')
                loss_function = getattr(module, 'LIPLoss')(args.mode)
            elif loss_type == 'VGG':
                loss_function = VGGPerceptualLoss(args.mode)
            else:
                loss_function = Adversarial(args)
            
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )

            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

    def forward(self, deep_images, target, depths, poses, valid_mask):
        total_loss = 0

        for i, l in enumerate(self.loss):
            if l['type'] in ['GAN', 'VGG']:
                loss = l['function'](deep_images[0], target[:, 0])
                effective_loss = l['weight'] * loss
                if l['type'] == 'GAN':
                    self.log[i + 1] += (l['function'].loss_d * loss).item()
            elif l['type'] not in ['Total', 'DIS']:
                loss = l['function'](deep_images, target, depths, poses, valid_mask)
                effective_loss = l['weight'] * loss
    
            self.log[i] += effective_loss
            total_loss += effective_loss

        return total_loss

    def step(self):
        for loss in self.get_loss_module():
            if hasattr(loss, 'scheduler'):
                loss.scheduler.step()

    def display_loss(self, batch):
        n_samples = (batch + 1) * self.batch_size
        log = ['[{}: {:.4f}]'.format(l['type'], c / n_samples) for l, c in zip(self.loss, self.log)]

        return ''.join(log)

    def get_loss_module(self):
        return self.loss_module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        kwargs = {'map_location': lambda storage, loc: storage} if cpu else {}
        self.load_state_dict(torch.load(os.path.join(apath, 'loss.pt'), **kwargs))
        
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.loss_module:
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()
