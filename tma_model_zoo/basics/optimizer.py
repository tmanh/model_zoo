import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def make_optimizer(args, target):
    """
    make optimizer and scheduler together
    """
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'ADAMW':
        optimizer_class = optim.AdamW
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    if args.decay_type == 'step':
        # scheduler
        kwargs_scheduler = {'step_size': args.milestone_distance, 'gamma': args.gamma}
        scheduler_class = lrs.StepLR
    elif args.decay_type == 'multi_step':
        milestones = args.schedule
        kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
        scheduler_class = lrs.MultiStepLR
    elif args.decay_type == 'exponential':
        kwargs_scheduler = {'gamma': args.gamma}
        scheduler_class = lrs.ExponentialLR
    elif args.decay_type == 'lambda':
        kwargs_scheduler = {'lr_lambda': args.lambda_funcs, 'gamma': args.gamma}
        scheduler_class = lrs.LambdaLR
    elif args.decay_type == 'cosine':
        kwargs_scheduler = {'T_max': args.T_max, 'eta_min': args.eta_min}
        scheduler_class = lrs.CosineAnnealingLR
    elif args.decay_type == 'reduce':
        kwargs_scheduler = {'reduce_mode': args.reduce_mode, 'factor': args.factor, 'patience': args.patience}
        scheduler_class = lrs.ReduceLROnPlateau

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch):
                    self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer
