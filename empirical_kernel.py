import numpy as np
import os,sys,time
import gc

from utils import *
from model import resnet


def empirical_K(model, data, number_samples, device,
        sigmaw=np.sqrt(2), sigmab=1.0, n_gpus=1,
        empirical_kernel_batch_size=256,
        truncated_init_dist=False,
        store_partial_kernel=True,
        partial_kernel_n_proc=1,
        partial_kernel_index=0):

    # Here model should be the coresponding CNN poped the lase fc layer.


    #number_samples = data.shape[0] # Number of MC samples
    num_tasks = number_samples

    if store_partial_kernel:
        size = partial_kernel_n_proc
        rank = partial_kernel_index
        num_tasks_per_job = num_tasks//size
        tasks = list(range(int(rank*num_tasks_per_job),int((rank+1)*num_tasks_per_job)))

        if rank < num_tasks%size:
            tasks.append(size*num_tasks_per_job+rank)
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(rank)
        num_tasks_per_job = num_tasks//size
        tasks = list(range(int(rank*num_tasks_per_job),int((rank+1)*num_tasks_per_job)))

        if rank < num_tasks%size:
            tasks.append(size*num_tasks_per_job+rank)

    print("Doing task %d of %d" % (rank, size))

    m = len(data)
    if device == 'cuda':
        covs = torch.zeros(m, m)
    else:
        covs = np.zeros((m,m), dtype=np.float32)
    local_index = 0
    update_chunk = 10000
    num_chunks = covs.shape[0]//update_chunk
    print("num_chunks",num_chunks)

    for index in tasks:
        start_time = time.time()
        print("sample for kernel", index)

        model.apply(he_init)
        # Guillermo chose to use Keras model, while I do everything in Pytorch.
        # The principle is in every MC sample the weights and biases are chosen from 
        # He-normal distribution (In lots of GP papers they use forward-version of Xavier
        # but for He-normal you only need to change the 'gain' from 1 to sqrt{2}. The term
        # 'gain' is used in Pytorch docs: https://pytorch.org/docs/stable/nn.init.html)

        # Also, Guillermo's 'reset_weights' will re-initialize weights and biases but keep
        # BatchNorm layer unchanged. While in Pytorch case the default initialization for 
        # BatchNorm is constant already.
        X = model_predict(model, data, empirical_kernel_batch_size, 4, device)
        print('X.shape:', X.shape)








if __name__ == '__main__':
    model = resnet.ResNet_pop_fc_50(num_classes=1) # Actually num_classes doesn't matter
                                                   # because the fc layer was removed.
    from main import trainset, device
    empirical_K(model, trainset, 1, device,
        sigmaw=np.sqrt(2), sigmab=1.0, n_gpus=1,
        empirical_kernel_batch_size=256,
        truncated_init_dist=False,
        store_partial_kernel=True,
        partial_kernel_n_proc=1,
        partial_kernel_index=0)








































