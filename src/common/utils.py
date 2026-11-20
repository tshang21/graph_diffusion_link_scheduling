import sys as _sys
import torch
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from scipy.spatial import distance
import os
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected, degree as pyg_degree

# Back-compat: the saved datasets (data/sawl_dataset_*.pt) pickle custom classes
# (SAWLData / SAWLDataset) under the old top-level module name ``utils``.  After
# the move to ``common.utils`` those pickles would fail with
# "No module named 'utils'"; alias this module so torch.load still resolves them.
_sys.modules.setdefault("utils", _sys.modules[__name__])

import numpy as np
from collections import defaultdict
from tqdm import tqdm
from torch_geometric.loader import DataLoader
                #from torch.utils.data import TensorDataset, DataLoader be careful using 
                # the corresponding dataloader for pytorch geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import ToUndirected

##########################################################
##########################################################
##########################################################
##########################################################
################## other function ########################
##########################################################
##########################################################
##########################################################
##########################################################

class SAWLDataset(Dataset):
    # Define a basic dataset
    def __init__(self, data_list):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx], idx

##########################################################
##########################################################
##########################################################
##########################################################
################## other function ########################
##########################################################
##########################################################
##########################################################
##########################################################

class SAWLData(Data):
    def __init__(self,
                 x=None,
                 edge_index=None,
                 A=None,
                 Delta=None,
                 K=None):
        super().__init__()
        self.x = x
        self.edge_index = edge_index
        self.A = A
        self.Delta = Delta
        self.K = K # I add the number of nodes in the conflict graph (that is,
                   # the amount of edges in the inital graph) for simplicity

##########################################################
##########################################################
##########################################################
##########################################################
################## other function ########################
##########################################################
##########################################################
##########################################################
##########################################################

def sort_tuple(tup):
    '''
    This function gets a tuple called tup containing two integers
    and it will sort it. This is needed so that a node called (0,1) and
    one called (1, 0) are considered the same.
    '''
    sorted_tuple = sorted(tup)
    return tuple(sorted_tuple)

##########################################################
##########################################################
##########################################################
##########################################################
################## other function ########################
##########################################################
##########################################################
##########################################################
##########################################################
# ------------------------------------------------------------

def create_conflict_graph(graph):
    # Create a dual graph from the input graph
    dual_graph = nx.line_graph(graph)
    dual_edges = list(dual_graph.edges())
    np.random.shuffle(dual_edges)
    tuples_as_numbers = {}
    next_number = 0
    for tuple_pair in dual_edges:
        for tup in tuple_pair:
            st = sort_tuple(tup)
            if st not in tuples_as_numbers:
                tuples_as_numbers[st] = next_number
                next_number += 1
    K = len(list(dual_graph.nodes()))
    A = torch.zeros(K, K)
    degrees = torch.zeros(K, dtype=torch.long)
    fixed_edges = []
    for tuple_pair in dual_edges:
        new_tuple_pair = []
        for tup in tuple_pair:
            st = sort_tuple(tup)
            transformed_value = tuples_as_numbers[st]
            new_tuple_pair.append(transformed_value)
        u, v = new_tuple_pair[0], new_tuple_pair[1]
        A[u, v] = 1
        A[v, u] = 1
        degrees[u] += 1
        degrees[v] += 1
        fixed_edges.append((u, v))
    if len(fixed_edges) > 0:
        edges = torch.tensor(fixed_edges, dtype=torch.long).t().contiguous()
    else:
        edges = torch.empty((2, 0), dtype=torch.long)
    dual_data = Data(edge_index=edges)
    transform = ToUndirected()
    dual_data = transform(dual_data)
    return dual_data, A, K, edges.shape[1], degrees
    
def get_data(data, batch_size):
    edge_index = data.edge_index # edges
    K = data.K # nodes in conflict graph
    K = K[0].cpu().numpy()
    A = data.A # A matrix
    A = A.view(batch_size, -1, K)
    Delta = data.Delta # minimum transmission requirement
    Delta = Delta.view(-1, K)
    Delta = Delta[0].view(K)
    return(edge_index, K, A, Delta)

def sample_lambda(epoch, T, option, scale, device, K, dist_lim):
    if option < dist_lim[0] or epoch == 0: # 15% of time I do uniform distribution
        lambda_dual = scale*torch.rand(K, 1).to(device)
    elif dist_lim[0] <= option and option < dist_lim[1]: # 15% of time I do uniform with 30% of zeros
        rand_vals = scale * torch.rand(K, 1, device=device)
        mask = (torch.rand(K, 1, device=device) >= 0.3).float()
        lambda_dual = (rand_vals * mask).to(device)
    elif dist_lim[1] <= option and option < dist_lim[2]: # 15% of time I do uniform with 25% of ones
        rand_vals = scale * torch.rand(K, 1)
        mask = torch.rand(K, 1) < 0.25
        lambda_dual = torch.where(mask, torch.tensor(1.0), rand_vals).to(device)
    return lambda_dual

def plot_lambdas(epoch, delta, lr_phi, lr_lambda, dual_variable, exp_name, weight_decay):
    plt.figure(figsize=(16,9))
    plt.title('Evolution of dual variable in epoch={}'.format(epoch))
    plt.suptitle('Delta={}, lr_primal={}, lr_dual={}, weight_decay={}'.format(delta, lr_phi, lr_lambda, weight_decay), fontsize=14, fontweight='bold')
    plt.ylabel('lambda in epoch {}'.format(epoch))
    plt.xlabel('Time step')
    plt.plot(np.linalg.norm(dual_variable.detach().cpu().numpy(), axis=1))
    image_name = 'results_average' + exp_name + 'lambda' + str(epoch) + '.png'
    image_path = os.path.join('results/lambdas', image_name)
    plt.savefig(image_path)
    return


def remove_collisions(schedule, A, threshold=0.5, rng=None, zero_if_all_active=False):
    """Greedy collision resolution for ONE timestep -> collision-free 0/1 mask.

    Given a schedule (continuous or binary) for a single timestep and a conflict
    matrix ``A`` (``A[i,j] != 0`` iff links i and j interfere), repeatedly turn
    off the more-conflicting endpoint of a collision until the still-active links
    form an independent set in ``A`` (no two active links conflict).

    Rule (matches the SAGNN expert pipeline): take the first colliding pair
    ``(i, j)`` in ascending index order; turn off whichever of i, j currently has
    MORE active conflicting neighbors; on a tie, turn one of the two off at random.

    Args:
        schedule: ``(K,)`` array/tensor. Raw schedules are binarized at ``threshold``.
        A:         ``(K, K)`` conflict matrix (nonzero = edge; diagonal ignored).
        threshold: raw -> active cutoff (default 0.5).
        rng:       optional ``np.random.Generator`` used only for tie-breaks, so
                   masks are reproducible when a seeded generator is passed.
        zero_if_all_active: when True, a slot with EVERY link scheduled (all
                   ``>= threshold``) is treated as degenerate and masked to all
                   zeros instead of resolved to a survivor set. Used for the masked
                   SAGNN, whose cold/low-lambda policy collapses to "schedule
                   everything"; that all-on output carries no scheduling decision,
                   so it should count as scheduling nothing. (Leave False for MIS /
                   UGNN, where an all-active slot on a sparse graph is a legitimate
                   full independent set.)

    Returns:
        ``(K,)`` float32 mask; ``1`` = active and survived, ``0`` = inactive/removed.
    """
    if rng is None:
        rng = np.random.default_rng()

    if torch.is_tensor(schedule):
        schedule = schedule.detach().cpu().numpy()
    schedule = np.asarray(schedule).reshape(-1)
    K = schedule.shape[0]
    active = schedule >= threshold

    if zero_if_all_active and active.all():
        return np.zeros(K, dtype=np.float32)                 # all links scheduled -> degenerate, mask to nothing

    if torch.is_tensor(A):
        A = A.detach().cpu().numpy()
    Adj = (np.asarray(A).reshape(K, K) != 0)
    np.fill_diagonal(Adj, False)

    # Incremental conflict counts: conf[k] = #active conflicting neighbors of k.
    # One link is dropped per iteration, so instead of recomputing the active
    # submatrix each round (O(|act|^2) -- pathological on dense schedules, e.g.
    # an untrained model emitting all-on slots), decrement conf for the dropped
    # link's neighbors.  Decision sequence (first colliding pair, drop the
    # higher-conflict endpoint, rng tie-break) is IDENTICAL to the original.
    conf = (Adj & active).sum(axis=1).astype(np.int64)   # active-neighbor counts
    while True:
        cand = np.flatnonzero(active & (conf > 0))       # active links with a collision
        if cand.size == 0:
            break                                        # active set is collision-free
        i = cand[0]                                      # first colliding link
        j = np.flatnonzero(Adj[i] & active)[0]           # its first active conflicting neighbor
        ci, cj = int(conf[i]), int(conf[j])
        if ci > cj:
            drop = i
        elif cj > ci:
            drop = j
        else:
            drop = rng.choice([i, j])                    # tie -> random
        active[drop] = False
        conf[Adj[drop]] -= 1                             # neighbors lose one active conflict

    return active.astype(np.float32)


def mask_schedule_trajectory(schedule, A, threshold=0.5, rng=None, zero_if_all_active=False):
    """Apply :func:`remove_collisions` to every timestep of a ``(T, K)`` schedule.

    Returns a ``(T, K)`` float32 collision-free mask (or ``(K,)`` if a single
    timestep is passed). Pass a seeded ``np.random.Generator`` as ``rng`` for
    reproducible tie-breaking across the whole trajectory. ``zero_if_all_active``
    is forwarded per-timestep (see :func:`remove_collisions`) -- use it for the
    masked SAGNN so all-on slots count as scheduling nothing.
    """
    if rng is None:
        rng = np.random.default_rng()
    if torch.is_tensor(schedule):
        schedule = schedule.detach().cpu().numpy()
    schedule = np.asarray(schedule, dtype=np.float32)
    if schedule.ndim == 1:
        return remove_collisions(schedule, A, threshold=threshold, rng=rng,
                                 zero_if_all_active=zero_if_all_active)
    return np.stack([
        remove_collisions(schedule[t], A, threshold=threshold, rng=rng,
                          zero_if_all_active=zero_if_all_active)
        for t in range(schedule.shape[0])
    ]).astype(np.float32)


def effective_success_trajectory(schedule, A, threshold=0.5):
    """Per-timestep 'unmasked' successful-transmission mask for a ``(T, K)`` schedule.

    Unlike :func:`mask_schedule_trajectory` (which resolves each collision by keeping
    one survivor link), here a link counts as a successful transmission ONLY if it is
    active AND none of its conflicting neighbours are active -- so when two links
    collide, NEITHER counts. This is the ``succ = s_binary * conflicts_binary``
    semantics of the SAGNN expert objective. The result is collision-free by
    construction, so it can feed the same metrics as a masked schedule.

    Returns a ``(T, K)`` float32 mask (or ``(K,)`` for a single timestep).
    """
    if torch.is_tensor(schedule):
        schedule = schedule.detach().cpu().numpy()
    schedule = np.asarray(schedule, dtype=np.float32)
    single = schedule.ndim == 1
    if single:
        schedule = schedule[None, :]
    if torch.is_tensor(A):
        A = A.detach().cpu().numpy()
    K = schedule.shape[1]
    Adj = (np.asarray(A).reshape(K, K) != 0).astype(np.float32)
    np.fill_diagonal(Adj, 0.0)
    active = (schedule >= threshold).astype(np.float32)             # (T, K)
    conflict_count = active @ Adj                                   # (T, K) active conflicting neighbours
    eff = (active * (conflict_count == 0)).astype(np.float32)

    # SAGNN expert objective: a link counts as successful only if it is active AND none of its conflicting neighbours are active. This is the succ = s_binary * conflicts_binary semantics of the SAGNN expert objective. The result is collision-free by construction, so it can feed the same metrics as a masked schedule.
    s_binary = (schedule >= threshold).astype(np.float32)
    conflicts_binary = (conflict_count == 0).astype(np.float32)
    succ = s_binary * conflicts_binary
    #return eff[0] if single else eff

    return succ

