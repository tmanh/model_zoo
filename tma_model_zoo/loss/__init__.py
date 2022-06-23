import os
import numpy as np

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
        self.log = np.zeros(len(self.loss) + 2)
        self.last_loss = 1e8
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
            elif loss_type == 'PRJ':
                loss_function = ProjectionLoss()
            elif loss_type == 'VGG':
                loss_function = VGGPerceptualLoss(args.mode)
            elif loss_type == 'SR':
                loss_function = DSRLoss()
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

    # refine, deep_dst_color, deep_prj_colors, deep_depths, dst_color, dst_depth, src_colors, src_depths, dst_intrinsic, dst_extrinsic, src_inrinsics, src_extrinsics, valid_mask
    def forward(self, tensors):
        total_loss = 0

        for i, l in enumerate(self.loss):
            effective_loss = 0
            if l['type'] not in ['DIS', 'Total']:
                loss = l['function'](tensors)
                effective_loss = l['weight'] * loss

            if l['type'] == 'GAN':
                self.log[i + 1] += (l['function'].loss_d * loss).detach().cpu().numpy()

            if not isinstance(effective_loss, int) and not isinstance(effective_loss, float):
                self.log[i] += effective_loss.detach().cpu().numpy()

            total_loss += effective_loss

        self.last_loss = total_loss.detach().cpu().numpy()

        self.log[-1] += total_loss.detach().cpu().numpy()

        return total_loss

    def reset_log(self):
        self.log = np.zeros(len(self.loss) + 2)

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
