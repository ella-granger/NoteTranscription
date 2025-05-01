import numpy as np
import torch
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
import math

from matplotlib import pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_cuda(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    """
    :param seq_len: The length of the sequence (both inputs are assumed to be of the same size)
    :param n_passes: 2 * seq_len - 1 (The number of anti-diagonals)
    """
    # Each block processes one pair of examples
    b = cuda.blockIdx.x
    # We have as many threads as seq_len, because the most number of threads we need
    # is equal to the number of elements on the largest anti-diagonal
    tid = cuda.threadIdx.x

    # Compute I, J, the indices from [0, seq_len)

    # The row index is always the same as tid
    I = tid

    inv_gamma = 1.0 / gamma

    # Go over each anti-diagonal. Only process threads that fall on the current on the anti-diagonal
    for p in range(n_passes):

        # The index is actually 'p - tid' but need to force it in-bounds
        J = max(0, min(p - tid, max_j - 1))

        # For simplicity, we define i, j which start from 1 (offset from I, J)
        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == p and (I < max_i and J < max_j):
            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
@cuda.jit
def compute_softdtw_backward_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # Indexing logic is the same as above, however, the anti-diagonal needs to
    # progress backwards
    I = tid

    for p in range(n_passes):
        # Reverse the order to make the loop go backward
        rev_p = n_passes - p - 1

        # convert tid to I, J, then i, j
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        # Only compute if element[i, j] is on the current anti-diagonal, and also is within bounds
        if I + J == rev_p and (I < max_i and J < max_j):

            if math.isinf(R[k, i, j]):
                R[k, i, j] = -math.inf

            # Don't compute if outside bandwidth
            if not (abs(i - j) > bandwidth > 0):
                a = math.exp((R[k, i + 1, j] - R[k, i, j] - D[k, i + 1, j]) * inv_gamma)
                b = math.exp((R[k, i, j + 1] - R[k, i, j] - D[k, i, j + 1]) * inv_gamma)
                c = math.exp((R[k, i + 1, j + 1] - R[k, i, j] - D[k, i + 1, j + 1]) * inv_gamma)
                E[k, i, j] = E[k, i + 1, j] * a + E[k, i, j + 1] * b + E[k, i + 1, j + 1] * c

        # Wait for other threads in this block
        cuda.syncthreads()

# ----------------------------------------------------------------------------------------------------------------------
def jacobean_product_squared_euclidean(X, Y, Bt):
    '''
    jacobean_product_squared_euclidean(X, Y, Bt):
    
    Jacobean product of squared Euclidean distance matrix and alignment matrix.
    See equations 2 and 2.5 of https://arxiv.org/abs/1703.01541
    '''
    # print(X.shape, Y.shape, Bt.shape)
    
    ones = torch.ones(Y.shape).to('cuda' if Bt.is_cuda else 'cpu')
    return 2 * (ones.matmul(Bt) * X - Y.matmul(Bt))


def jacobean_product_norm_nll(z, mu, logs, D, B):
    # print("--------------J mat--------------")
    # print(z.size())
    # print(mu.size())
    # print(logs.size())
    # print(D.size())
    # print(B.size())
    # print("---------------------------------")

    en2logs = torch.exp(-2 * logs)
    Bt = B.transpose(1, 2)
    pn1 = torch.ones(z.shape).to('cuda' if B.is_cuda else 'cpu')

    Jz = en2logs.matmul(Bt) * z - mu.matmul(Bt)
    Jmu = pn1.matmul(B) * mu * en2logs - z.matmul(B)
    Jlogs = 2 * pn1.matmul(B) * logs + pn1.matmul(1 + math.log(2 * math.pi) - 2 * D)

    return Jz.transpose(1,2), Jmu.transpose(1,2), Jlogs.transpose(1,2)


class _SoftDTWCUDA(Function):
    """
    CUDA implementation is inspired by the diagonal one proposed in https://ieeexplore.ieee.org/document/8400444:
    "Developing a pattern discovery method in time series data and its GPU acceleration"
    """

    @staticmethod
    def forward(ctx, z, mu, logs, D, gamma, bandwidth):
        dev = D.device
        dtype = D.dtype
        gamma = torch.cuda.FloatTensor([gamma])
        bandwidth = torch.cuda.FloatTensor([bandwidth])

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        # Prepare the output array
        R = torch.ones((B, N + 2, M + 2), device=dev, dtype=dtype) * math.inf
        R[:, 0, 0] = 0

        # Run the CUDA kernel.
        # Set CUDA's grid size to be equal to the batch size (every CUDA block processes one sample pair)
        # Set the CUDA block size to be equal to the length of the longer sequence (equal to the size of the largest diagonal)
        compute_softdtw_cuda[B, threads_per_block](cuda.as_cuda_array(D.detach()),
                                                   gamma.item(), bandwidth.item(), N, M, n_passes,
                                                   cuda.as_cuda_array(R))
        ctx.save_for_backward(D, z, mu, logs, R, gamma, bandwidth)
        return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        # print("------------Backward------------")
        # print(grad_output.size())
        dev = grad_output.device
        dtype = grad_output.dtype
        # D, X, Y, R, gamma, bandwidth = ctx.saved_tensors
        D, z, mu, logs, R, gamma, bandwidth = ctx.saved_tensors
        # print("------------In size-------------")
        # print(z.size())
        # print(mu.size())
        # print(logs.size())

        B = D.shape[0]
        N = D.shape[1]
        M = D.shape[2]
        threads_per_block = max(N, M)
        n_passes = 2 * threads_per_block - 1

        D_ = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        D_[:, 1:N + 1, 1:M + 1] = D

        R[:, :, -1] = -math.inf
        R[:, -1, :] = -math.inf
        R[:, -1, -1] = R[:, -2, -2]

        E = torch.zeros((B, N + 2, M + 2), dtype=dtype, device=dev)
        E[:, -1, -1] = 1

        # Grid and block sizes are set same as done above for the forward() call
        compute_softdtw_backward_cuda[B, threads_per_block](cuda.as_cuda_array(D_),
                                                            cuda.as_cuda_array(R),
                                                            1.0 / gamma.item(), bandwidth.item(), N, M, n_passes,
                                                            cuda.as_cuda_array(E))
        E = E[:, 1:N + 1, 1:M + 1]
        # print("--------------E--------------")
        # print(E.size())
        plt.clf()
        plt.matshow(E[0].detach().cpu(), origin="lower")
        plt.colorbar()
        plt.savefig("E.png")
        Jz, Jmu, Jlogs = jacobean_product_norm_nll(z.transpose(1,2),
                                                   mu.transpose(1,2),
                                                   logs.transpose(1,2),
                                                   D, E)
        # G = jacobean_product_squared_euclidean(X.transpose(1,2), Y.transpose(1,2), E.transpose(1,2)).transpose(1,2)
        # print(G.size())
        # print(Jz.size())
        # print(Jmu.size())
        # print(Jlogs.size())
        # print("------------J fin------------")

        Jz = grad_output.view(-1, 1, 1).expand_as(Jz) * Jz
        Jmu = grad_output.view(-1, 1, 1).expand_as(Jmu) * Jmu
        Jlogs = grad_output.view(-1, 1, 1).expand_as(Jlogs) * Jlogs

        return Jz, Jmu, Jlogs, E, None, None

# ----------------------------------------------------------------------------------------------------------------------
class SoftDTW(torch.nn.Module):
    """
    The soft DTW implementation that optionally supports CUDA
    """

    def __init__(self, use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None):
        """
        Initializes a new instance using the supplied parameters
        :param use_cuda: Flag indicating whether the CUDA implementation should be used
        :param gamma: sDTW's gamma parameter
        :param normalize: Flag indicating whether to perform normalization
                          (as discussed in https://github.com/mblondel/soft-dtw/issues/10#issuecomment-383564790)
        :param bandwidth: Sakoe-Chiba bandwidth for pruning. Passing 'None' will disable pruning.
        :param dist_func: Optional point-wise distance function to use. If 'None', then a default Euclidean distance function will be used.
        """
        super(SoftDTW, self).__init__()

        assert use_cuda, "Only the CUDA version is supported."

        self.normalize = normalize
        self.gamma = gamma
        self.bandwidth = 0 if bandwidth is None else float(bandwidth)
        self.use_cuda = use_cuda

        # Set the distance function
        if dist_func is not None:
            if dist_func == "nll":
                self.dist_func = SoftDTW._norm_nll_func
            else:
                self.dist_func = dist_func
        else:
            self.dist_func = SoftDTW._euclidean_dist_func

    def _get_func_dtw(self, z, mu, logs):
        """
        Checks the inputs and selects the proper implementation to use.
        """
        # print(z.shape)
        # print(mu.shape)
        # print(logs.shape)
        bx, lx, dx = z.shape
        by, ly, dy = mu.shape
        bs, ls, ds = logs.shape
        # Make sure the dimensions match
        assert bx == by  # Equal batch sizes
        assert bx == bs
        assert dx == dy  # Equal feature dimensions
        assert dx == ds
        assert ly == ls

        use_cuda = self.use_cuda

        if use_cuda and (lx > 1024 or ly > 1024):  # We should be able to spawn enough threads in CUDA
                print("SoftDTW: Cannot use CUDA because the sequence length > 1024 (the maximum block size supported by CUDA)")
                use_cuda = False

        # Finally, return the correct function
        return _SoftDTWCUDA.apply

    @staticmethod
    def _euclidean_dist_func(x, y):
        """
        Calculates the Euclidean distance between each element in x and y per timestep
        """
        n = x.size(1)
        m = y.size(1)
        d = x.size(2)
        x = x.unsqueeze(2).expand(-1, n, m, d)
        y = y.unsqueeze(1).expand(-1, n, m, d)
        return torch.pow(x - y, 2).sum(3)

    @staticmethod
    def _norm_nll_func(z, mu, logs):
        # print("----------------------------------")
        z = z.transpose(1,2)
        # print(z.size())
        # print(mu.size())
        # print(logs.size())
        # print("----------------------------------")
        nll = torch.sum(0.5 * math.log(2 * math.pi) + logs, -1, keepdim=True)
        # print(nll.size())
        factor = 0.5 * torch.exp(-2 * logs)
        # print(factor.size())
        nll = nll + torch.matmul(factor, z**2)
        # print(nll.size())
        nll -= 2 * torch.matmul(factor * mu, z)
        # print(nll.size())
        nll += torch.sum(factor * mu**2, -1, keepdim=True)
        # print(nll.size())
        nll = torch.permute(nll, (0, 2, 1)) # (B, LN, LM)
        # print(nll.size())
        nll = nll.contiguous()
        # print("------------D fin----------------")
        return nll

    def forward(self, z, mu, logs):
        """
        Compute the soft-DTW value between X and Y
        :param X: One batch of examples, batch_size x seq_len x dims
        :param Y: The other batch of examples, batch_size x seq_len x dims
        :return: The computed results
        """

        # Check the inputs and get the correct implementation
        func_dtw = self._get_func_dtw(z, mu, logs)

        if self.normalize:
            # Stack everything up and run
            x = torch.cat([X, X, Y])
            y = torch.cat([Y, X, Y])
            D = self.dist_func(x, y)
            out = func_dtw(X, Y, D, self.gamma, self.bandwidth)
            out_xy, out_xx, out_yy = torch.split(out, X.shape[0])
            return out_xy - 1 / 2 * (out_xx + out_yy)
        else:
            D = self.dist_func(z, mu, logs)
            plt.clf()
            plt.matshow(D[0].detach().cpu(), origin="lower")
            plt.colorbar()
            plt.savefig("D.png")
            return func_dtw(z, mu, logs, D, self.gamma, self.bandwidth)


if __name__ == "__main__":
    batch_size, len_x, len_y, dims = 8, 15, 12, 5 # 8, 15, 12, 5
    # z = [1,1,1,3,3,3,5,5,5,3,3,3,1,1,1]
    # mu = [1,1,3,3,4,5,5,4,3,3,1,1]
    # mu = [1,1,1,3,3,3,5,5,5,3,3,3,1,1,1]
    # z = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5]
    # mu = [2,2,2,3,3,3,4,4,4,5,5,5,6,6,6]
    z = torch.rand((batch_size, len_x, dims), requires_grad=True).cuda()
    mu = torch.rand((batch_size, len_y, dims), requires_grad=True).cuda()
    # z = torch.tensor(z).unsqueeze(0).unsqueeze(-1).cuda().to(torch.float32)
    # mu = torch.tensor(mu).unsqueeze(0).unsqueeze(-1).cuda().to(torch.float32) + 10
    # logs = torch.abs(torch.rand((batch_size, len_y, dims), requires_grad=True)).cuda()
    logs = torch.zeros_like(mu, requires_grad=True).cuda() - 1.0

    # Create the "criterion" object
    sdtw = SoftDTW(use_cuda=True, gamma=0.1, dist_func="nll")

    # Compute the loss value
    loss = sdtw(z, mu, logs)  # Just like any torch.nn.xyzLoss()
    loss = loss.mean()
    print(loss.item())

    # Aggregate and call backward()
    loss.backward()
