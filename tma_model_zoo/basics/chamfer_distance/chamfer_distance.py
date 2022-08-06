import torch
import os
from torch.utils.cpp_extension import load


# TODO: there is a chance that you will face "nvcc fatal: Unknown option '-generate-dependencies-with-compile'" error. To prevent
# this error, go to the source file of torch.utils.cpp_extension and change the version that is required for building the dependencies
# at this line "required_cuda_version = packaging.version.parse('10.2')"
# E.g. If your cuda version is 11.1, then change it to 10.2 ==> 11.2. So, it will skip building and generating the dependencies simultaneously
f_path = os.path.dirname(os.path.realpath(__file__))
cd = load(name='cfd', sources=[f'{f_path}/cfd/chamfer_distance.cpp', f'{f_path}/cfd/chamfer_distance.cu'], verbose=True)


class ChamferDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2, device):
        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        idx2 = torch.zeros(batchsize, m, dtype=torch.int)

        if not xyz1.is_cuda:
            cd.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.to(device)
            dist2 = dist2.to(device)
            idx1 = idx1.to(device)
            idx2 = idx2.to(device)
            cd.forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, device):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        if not graddist1.is_cuda:
            cd.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            gradxyz1 = gradxyz1.to(device)
            gradxyz2 = gradxyz2.to(device)
            cd.backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


class ChamferDistance(torch.nn.Module):
    def forward(self, xyz1, xyz2):
        assert(xyz1.device == xyz2.device)
        return ChamferDistanceFunction.apply(xyz1, xyz2, xyz1.device)
