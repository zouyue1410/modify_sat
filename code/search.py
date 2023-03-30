import pdb
import random

import scipy.sparse as sparse
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli

from data_search import Batch, DataSample, init_edge_attr, init_tensors, to_sparse_tensor
from util import normalize


class LocalSearch:
    def __init__(self, policy, device, config):
        self.policy = policy
        self.device = device
        if config['method'] == 'reinforce':
            self.generate_episode = self._generate_episode_reinforce
        elif config['method'] == 'reinforce_multi':
            self.generate_episode = self._generate_episode_reinforce_multi
        elif config['method'] == 'pg':
            self.generate_episode = self._generate_episode_pg
        elif config['method'] == 'a2c':
            self.generate_episode = self._generate_episode_a2c

    def eval(self):
        self.generate_episode = self._eval_generate_episode_a2c

    def train(self):
        self.generate_episode = self._generate_episode_a2c



    # @profile




    def _select_variable_a2c(self, data):
        logit, value = self.policy(data)

        prob = F.softmax(logit, dim=0)
        log_prob = F.log_softmax(logit, dim=0)
        entropy = -(log_prob * prob).sum()

        dist = Categorical(prob.view(-1))
        v = dist.sample()

        return v, dist.log_prob(v), value, entropy
    '''
        def eval_select_variable_a2c(self,data):


        logit, value = self.policy(data)
        v = logit.argmax(dim=0)
        return v
    '''

    def _generate_episode_a2c(self, sample, max_flips,walk_prob):
        f = sample.formula
        data = init_tensors(sample, self.device)
        true_lit_count = compute_true_lit_count(f.clauses, data.sol)

        log_probs = []
        values = []
        entropies = []

        flip = 0
        flipped = set()
        backflipped = 0
        unsat_clauses = []
        entropy=0
        while flip < max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if true_lit_count[k] == 0]
            unsat_clauses.append(len(unsat_clause_indices))
            sat = not unsat_clause_indices
            if sat:
                break

            v, log_prob, value, entropy = self._select_variable_a2c(data)
            if v.item() not in flipped:
                flipped.add(v.item())
            else:
                backflipped += 1
            flip_(data.x[0], data.sol, true_lit_count, v, f.occur_list)
            flip += 1
            #entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)


        return sat, (flip, backflipped, unsat_clauses), (log_probs, values, entropies)
    '''
        def _eval_generate_episode_a2c(self,sample, max_flips,walk_prob):
        f = sample.formula
        data = init_tensors(sample, self.device)
        true_lit_count = compute_true_lit_count(f.clauses, data.sol)

        #log_probs = []
        #values = []
        #entropies = []

        flip = 0
        while flip < max_flips:
            unsat_clause_indices = [k for k in range(len(f.clauses)) if true_lit_count[k] == 0]
            sat = not unsat_clause_indices
            if sat:
                break
            v, log_prob, value, entropy = self.eval_select_variable_a2c(data)
            flip_(data.x[0], data.sol, true_lit_count, v, f.occur_list)
            flip += 1

            #log_probs.append(log_prob)
            #values.append(value)
            #entropies.append(entropy)

        return sat, flip,None
    '''


def flip_multi_(xv, sol, true_lit_count, vs, occur_list):
    for v in vs.int().cpu().numpy().flatten():
        lit_false = (v + 1) * normalize(int(xv[v, 0].item() == 1))
        for i in occur_list[lit_false]:
            true_lit_count[i] += 1
        for i in occur_list[-lit_false]:
            true_lit_count[i] -= 1
        xv[v, :2] = 1 - xv[v, :2]
        sol[v + 1] *= -1

#xv:n*3表示变量的vector
def flip_(xv, sol, true_lit_count, v, occur_list):
    lit_false = (v + 1) * normalize(int(xv[v, 0].item() == 1))
    for i in occur_list[lit_false]:
        true_lit_count[i] += 1
    for i in occur_list[-lit_false]:
        true_lit_count[i] -= 1
    xv[v, :2] = 1 - xv[v, :2]
    sol[v + 1] *= -1


def compute_true_lit_count(clauses, sol):
    n_clauses = len(clauses)
    true_lit_count = [0] * n_clauses
    for i in range(n_clauses):
        for lit in clauses[i]:
            if sol[abs(lit)] == lit:
                true_lit_count[i] += 1
    return true_lit_count
