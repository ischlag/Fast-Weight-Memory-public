# based on backpropamine code by miconi

import argparse
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from numpy import random
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
import random
import sys
import pickle
import time
import os
import platform

import numpy as np

from myfastweights import FWMRNN

np.set_printoptions(precision=4)


class AgentLSTM(nn.Module):

    def __init__(self, p):
        super(AgentLSTM, self).__init__()
        self.l1 = nn.LSTM(p['in_size'], p['hidden'], 1, dropout=0)
        
        self.h_c = nn.Linear(p['hidden'], p['n_actions'])
        self.h_v = nn.Linear(p['hidden'], 1)

        self.p = p
        
    def reset(self):
        device = next(self.parameters()).device
        self.hidden = (torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device),
                            torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device))

    def unchain(self):
        if self.hidden:
            self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
    def __call__(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        h, self.hidden = self.l1(x, self.hidden)
        h = h.reshape(-1, h.size(-1))
        act = self.h_c(h)
        val = self.h_v(h)
        return act, val

class AgentFWM(nn.Module):

    def __init__(self, p):
        super(AgentFWM, self).__init__()
        self.l1 = FWMRNN(p['in_size'], p['hidden'], s_size=p['s_size'])
        
        self.h_c = nn.Linear(p['hidden'], p['n_actions'])
        self.h_v = nn.Linear(p['hidden'], 1)

        self.p = p
        self.nhid = p['hidden']
        
    def reset(self):
        device = next(self.parameters()).device
        lstm_hidden = (torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device),
                       torch.zeros(1, self.p['batch_size'], self.p['hidden']).to(device))
        fwm_hidden = torch.zeros(self.p['batch_size'], 
                                 self.p['s_size'],
                                 self.p['s_size'], 
                                 self.p['s_size']).to(device)
        self.hidden = (lstm_hidden, fwm_hidden)

    def unchain(self):
        if self.hidden:
            lstm_hidden, fwm_hidden = self.hidden
            lstm_hidden = (lstm_hidden[0].detach(), lstm_hidden[1].detach())
            fwm_hidden = fwm_hidden.detach()
            self.hidden = (lstm_hidden, fwm_hidden)

    def __call__(self, x):
        if len(x.shape) < 3:
            x = x.unsqueeze(0)
        h, self.hidden = self.l1(x, self.hidden)
        h = h.reshape(-1, h.size(-1))
        act = self.h_c(h)
        val = self.h_v(h)
        return act, val


def sample_adjacency_matrix(n_actions, n_islands, random_state):
    while True:
        A = np.zeros((n_actions, n_islands, n_islands))

        # every island has to be leavable by at least one means of transportation (and not not be ambiguous)
        for from_island in range(n_islands):      
            to_island = random_state.choice([i for i in range(n_islands) if i != from_island])
            transport = random_state.randint(0, n_actions)
            A[transport, from_island, to_island] = 1

        # every island has to be reachable by one or more from-islands
        for to_island in range(n_islands):
            # only select from the islands that don't have any neighbours for a certain transportation
            transport_list, from_list = np.where(A.sum(2) == 0)
            # remove self from the selection
            options = np.asarray(list(filter(lambda x: x[0] != to_island, zip(from_list, transport_list))))
            indecies = np.arange(options.shape[0])
            chosen_idx = random_state.choice(indecies)
            from_island, transport = options[chosen_idx]
            A[transport, from_island, to_island] = 1

        # reject if they are not all connected
        Q = A.sum(0)
        Q[Q > 0] = 1
        for _ in range(n_islands):
            Q = np.matmul(Q,Q)
        if (Q == 0).sum() == 0:
            return A


def generate_graphs(n_graphs, n_agents, n_actions, n_states, not_allowed_graphs, random_state):
    graphs = []
    for _ in range(n_graphs):
        while True:
            new_g = sample_adjacency_matrix(n_actions=n_actions, 
                                            n_islands=n_states,
                                            random_state=random_state)
            if not_allowed_graphs:
                if not any([(new_g == test_g).all() for test_g in not_allowed_graphs]):
                    break
            else:
            	break
        graphs += [new_g for _ in range(n_agents)]
    return graphs


def run_episode(p, graphs, net):
    batch_size = p['batch_size']
    n_actions = p['n_actions']
    n_islands = p['n_islands']

    # prepare to perform steps in the environment to train
    loss = 0
    lossv = 0        
    last_action = np.zeros(batch_size, dtype='int32')

    rewards = []
    total_reward = np.zeros(batch_size, dtype='float32')
    values = []
    logprobs = []

    reward_reset_timer = np.zeros(batch_size, dtype='int32')

    # sample new agent and reward position
    state = []
    for batch_j in range(batch_size):
        # select the reward location
        reward_island = np.random.randint(0, n_islands)

        # select the agent location different from the reward location
        agent_island = np.random.choice([i for i in range(n_islands) if i != reward_island])

        state.append([reward_island, agent_island])

    # run steps
    for period_l in range(1, p['periods']+1):
        # reset for every period the reward timer and last action
        for batch_j in range(batch_size):
            last_action[batch_j] = n_actions  # there is an additional hidden action
            reward_reset_timer[batch_j] = 0

        for step_k in range(p['n_steps']):
            # construct the input to the model
            inputs = np.zeros((batch_size, p['in_size']), dtype='float32')
            curr_reward = np.zeros(batch_size, dtype='float32')
            for batch_j in range(batch_size):
                reward_idx = state[batch_j][0]
                agent_idx = state[batch_j][1]

                # set the respective one-hot bits
                # ... for destination
                inputs[batch_j, reward_idx] = 1
                # ... for current location
                inputs[batch_j, n_islands + agent_idx] = 1
                # ... for last action
                inputs[batch_j, 2*n_islands + last_action[batch_j] + 1] = 1
                # ... and for other
                inputs[batch_j, 2*n_islands + n_actions + 1] = 1.0  # bias
                inputs[batch_j, 2*n_islands + n_actions + 2] = step_k / p['n_steps']  # episode length
                inputs[batch_j, 2*n_islands + n_actions + 3] = 1.0 * curr_reward[batch_j]
            inputs = torch.from_numpy(inputs).cuda()
            
            # run through the network
            action_logits, value = net(inputs)

            action_probs = F.softmax(action_logits, dim=1)
            # we add noise to action_probs because otherwise it can trigger cuda error due to 
            # value which are negative or all values being zero. (BUG)
            action_dist = torch.distributions.Categorical(action_probs + 1e-8) 
            action_samples = action_dist.sample()
            logprobs.append(action_dist.log_prob(action_samples))

            # state transitions
            action_samples = action_samples.data.cpu().numpy()  # Turn to scalar
            for batch_j in range(batch_size):
                action_j = action_samples[batch_j]

                reward_idx = state[batch_j][0]
                agent_idx = state[batch_j][1]

                graph_j = graphs[batch_j * period_l]
                neighbours = np.where(graph_j[action_j, agent_idx] == 1)[0]

                last_action[batch_j] = int(action_j)  # remember the action taken

                if len(neighbours) == 0:
                    # invalid action, this island does not provide this mode of transportation                    
                    curr_reward[batch_j] -= p['penalty']  # add penalty
                    reward_reset_timer[batch_j] += 1 # increase timer due to no reward
                elif len(neighbours) == 1:
                    # valid action
                    new_agent_idx = neighbours[0]  # travel to the new island

                    if reward_idx == new_agent_idx:
                        # destination is reached
                        curr_reward[batch_j] += p['reward']  # add reward for reaching the destination
                        # randomly select a new destination
                        state[batch_j][0] = np.random.choice([i for i in range(n_islands) if i != reward_idx])
                        # randomly select a new position for the agent
                        state[batch_j][1] = np.random.choice([i for i in range(n_islands) if i != state[batch_j][0]])
                    else:
                        # destination not reached yet
                        reward_reset_timer[batch_j] += 1 # increase timer due to no reward                    
                        state[batch_j][1] = new_agent_idx  # set the state to the new location
                else:
                   raise Exception("Graph is faulty. Should not have multiple neighbours with the same transportation") 
                
                # reset agent and reward if timer exceeds maximum number of steps
                if reward_reset_timer[batch_j] > n_islands:
                    # randomly select a new destination
                    state[batch_j][0] = np.random.choice([i for i in range(n_islands) if i != reward_idx])
                    # randomly select a new position for the agent
                    state[batch_j][1] = np.random.choice([i for i in range(n_islands) if i != state[batch_j][0]])
                    # randomly select a new position for the agent
                    last_action[batch_j] = n_actions  # there is an additional hidden action
                    reward_reset_timer[batch_j] = 0

            rewards.append(curr_reward)
            values.append(value)
            total_reward += curr_reward

            # This is the "entropy bonus" of A2C, except that since our version
            # of PyTorch doesn't have an entropy() function, we implement it as
            # a penalty on the sum of squares instead. The effect is the same:
            # we want to penalize concentration of probabilities, i.e.
            # encourage diversity of actions.
            # loss += ( p['bent'] * y.pow(2).sum() / BATCHSIZE )  
            action_entropy = -action_dist.entropy().mean()
            loss += p['entropy_coef'] * action_entropy
            # end of episode

    # compute loss based on episode steps
    R = Variable(torch.zeros(batch_size).cuda(), requires_grad=False)
    gammaR = p['gamma']
    for numstepb in reversed(range(p['n_steps'])) :
        R = gammaR * R + Variable(torch.from_numpy(rewards[numstepb]).cuda(), requires_grad=False)
        ctrR = R - values[numstepb]
        lossv += ctrR.pow(2).mean()
        loss -= (logprobs[numstepb] * ctrR.detach()).mean()

    loss += p['value_coef'] * lossv
    loss /= p['n_steps']*p['periods']

    return loss, total_reward


def run(args_dict):
    p = {}
    p.update(args_dict)

    # Initialize random seeds
    global_seed = random.randint(0, 100000000)
    print ("Setting global random seed: ", global_seed)
    p['global_seed'] = global_seed
    np.random.seed(p['global_seed'])
    torch.manual_seed(p['global_seed'])
    torch.cuda.manual_seed_all(p['global_seed'])

    # some constants
    batch_size = p['batch_size'] = p['batch_envs'] * p['batch_agents']
    n_actions = p['n_actions'] # means of transportation
    n_islands = p['n_islands']
    extra_input = 4 # 1 for the previous reward, 1 for numstep, 1 "Bias" inputs, 1 for no-action
    p['in_size'] = n_islands + n_islands + extra_input + n_actions 

    suffix = "{}_h{}_m{}_graph{},{}_r{}_p{}_lr{}_steps{}_periods{}_batch{},{}_dataseed{}_modelseed{}{}{}".format(
        p['model'], p['hidden'], p['s_size'], 
        p['n_islands'], p['n_actions'], p['reward'], p['penalty'], p['lr'], p['n_steps'], p['periods'], 
        p['batch_envs'], p['batch_agents'], p['data_seed'], p['global_seed'],
        "_randTrain" if p['randomize_train'] else "",
        "_keepHidden" if p['keep_hidden'] else "",
        )
    # construct model
    print("Build network ...")
    if p['model'] == 'lstm':
        net = AgentLSTM(p)
    elif p['model'] == 'fwm':
        net = AgentFWM(p)
    net = net.to("cuda")
    print (net)
    print ("shape of trainable objects:", [x.size() for x in net.parameters()])
    allsizes = [torch.numel(x.data.cpu()) for x in net.parameters()]
    print ("size of trainable objects:", allsizes)
    print ("trainable parameters in total:", sum(allsizes))
    print("Environments: ", p['batch_envs'])
    print("Agents per environment: ", p['batch_agents'])
    print("batch_size: ", batch_size)


    print("Initializing optimizer ...")
    optimizer = torch.optim.Adam(net.parameters(), lr=p['lr'], eps=1e-4, weight_decay=p['l2'])

    print("Experiment config: \n", p)
    print(platform.uname())

    #if p['periods'] > 1 and not p['randomize_train']:
    #    print("(!) WARNING: You train with multiple periods but don't randomize train. " + 
    #            "This makes the graph sequences deterministic which means that " + 
    #            "the agent doesn't have to learn to adapt to a new graph")

    print("Begin training")
    all_losses = []
    all_losses_objective = []
    all_total_rewards = []
    all_test_total_rewards = []
    all_losses_v = []
    timestamp = time.time()
    graphs = []

    # init model hidden states
    net.reset()

    # generate data
    random_data_state = np.random.RandomState(p['data_seed'])
    test_graphs = generate_graphs(n_graphs=p['batch_envs']*p['periods'], 
                                  n_agents=p['batch_agents'],
                                  n_actions=p['n_actions'], 
                                  n_states=p['n_islands'], 
                                  not_allowed_graphs=None, 
                                  random_state=random_data_state)

    train_graphs = generate_graphs(n_graphs=p['batch_envs']*p['periods'], 
                                   n_agents=p['batch_agents'],
                                   n_actions=p['n_actions'], 
                                   n_states=p['n_islands'], 
                                   not_allowed_graphs=test_graphs, 
                                   random_state=random_data_state)

    # start training
    for episode_i in range(p['n_episodes']):

        if p['randomize_train']:
            train_graphs = generate_graphs(n_graphs=p['batch_envs']*p['periods'], 
                                           n_agents=p['batch_agents'],
                                           n_actions=p['n_actions'], 
                                           n_states=p['n_islands'], 
                                           not_allowed_graphs=test_graphs, 
                                           random_state=random_data_state)
        
        if not p['randomize_train'] and p['periods'] > 1:
        	# if we don't want to generate new training graphs but 
        	# we run multiple periods then we should make sure the 
        	# sequence of train graphs is not deterministic. So we 
        	# simply shuffle the list of graphs
        	random.shuffle(train_graphs)

        optimizer.zero_grad()
        net.unchain()
        if not p['keep_hidden']:
        	net.reset()

        loss, total_reward = run_episode(p, train_graphs, net)

        loss.backward()
        optimizer.step()

        all_losses_objective.append(float(loss))
        all_total_rewards.append(total_reward.mean())

        # terminal plot
        if episode_i % p['log_every_n_episode'] == 0:
            curr_time = time.time()

            # avg reward since the last terminal log
            avg_train_reward = np.mean(all_total_rewards[-p['log_every_n_episode']:])
            avg_train_loss = np.mean(all_losses_objective[-p['log_every_n_episode']:])

            # test            
            train_state = net.hidden
            net.unchain()  # save train state
            if not p['keep_hidden']:
                net.reset()
            test_loss, test_total_reward = run_episode(p, test_graphs, net)
            all_test_total_rewards.append(test_total_reward.mean())
            avg_test_reward = test_total_reward.mean()

            net.hidden = train_state  # reset train state


            print("episode {:3}: train R={:6.3f} loss={:.4f} | test R={:6.3f} loss={:.4f}  ({:.1f}s)".format(
                episode_i, 
                avg_train_reward, avg_train_loss, 
                avg_test_reward, test_loss, 
                curr_time - timestamp))
            timestamp = curr_time

        # save 
        if episode_i % p['save_every_n_episode'] == 0 and episode_i > 0:
            print("\nSaving files...")
            net.zero_grad()
            torch.save(net, p['save_dir'] + '/model_'+ suffix +'.pt')
            torch.save(optimizer, p['save_dir'] + '/optimizer_'+ suffix +'.pt')
            with open(p['save_dir'] + '/trainrewards_' + suffix + '.txt', 'w') as thefile:
                for item in all_total_rewards[::p['log_every_n_episode']]:
                        thefile.write("%s\n" % item)
            with open(p['save_dir'] + '/testrewards_' + suffix + '.txt', 'w') as thefile:
                for item in all_test_total_rewards:
                        thefile.write("%s\n" % item)
            with open(p['save_dir'] + '/params_' + suffix + '.dat', 'wb') as fo:
                pickle.dump(p, fo)
            print("Done!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # experiment
    parser.add_argument("--lr", type=float, help="learning rate (Adam optimizer)", default=1e-3)
    parser.add_argument("--periods", type=int, help="number of different graphs per run", default=3)    
    parser.add_argument("--batch_envs", type=int, help="number of environments in a batch", default=30)
    parser.add_argument("--batch_agents", type=int, help="number agents per environment", default=10)
    parser.add_argument("--n_steps", type=int, help="length of episode", default=20)
    parser.add_argument("--n_episodes", type=int, help="length of trial", default=500)
    parser.add_argument("--save_every_n_episode", type=int, help="number of episodes between successive save points", default=1000)
    parser.add_argument("--log_every_n_episode", type=int, help="number of episodes before terminal print", default=100)
    parser.add_argument("--save_dir", help="log dir", default='logs_new')
    parser.add_argument('--randomize_train', help='resample train graphs for every episode', action='store_true')
    parser.add_argument('--keep_hidden', help='keep the hidden state between batches', action='store_true')

    # model
    parser.add_argument("--model", help="model: lstm | fwm", default='lstm')
    parser.add_argument("--hidden", type=int, help="size of the recurrent (hidden) layer", default=64)
    parser.add_argument("--s_size", type=int, help="size of fw tensor", default=32)

    # environment
    parser.add_argument("--n_islands", type=int, help="number of islands", default=5)
    parser.add_argument("--n_actions", type=int, help="number of actions/transportation modes", default=3)
    parser.add_argument("--reward", type=float, help="reward value (reward increment for taking correct action after correct stimulus)", default=10.0)
    parser.add_argument("--penalty", type=float, help="penalty for choosing the wrong action", default=.05)

    # other    
    parser.add_argument("--data_seed", type=int, help="random seed for the train and test graphs", default=666)
    parser.add_argument("--entropy_coef", type=float, help="coefficient for the entropy reward (really Simpson index concentration measure)", default=0.03)
    parser.add_argument("--value_coef", type=float, help="coefficient for value prediction loss", default=.1)
    parser.add_argument("--gamma", type=float, help="gammaR: discounting factor for rewards", default=.9)
    parser.add_argument("--l2", type=float, help="coefficient of L2 norm (weight decay)", default=0)

    args = parser.parse_args()
    argvars = vars(args)
    args_dict = { k : argvars[k] for k in argvars if argvars[k] != None }
    run(args_dict)
