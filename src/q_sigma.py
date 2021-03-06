from game import *
from learningAgents import ReinforcementAgent
import fileio
import numpy as np

import random,util,math
from collections import defaultdict

class QSigmaAgent(ReinforcementAgent):
    def __init__(self, discount, alpha, epsilon, actionFn, n, terminalFn, sigma, numEpisodes):
        ReinforcementAgent.__init__(self, gamma=discount, alpha=alpha, epsilon=epsilon, actionFn=actionFn)
        
        assert n > 0
        self.n = n
        self.terminalFn = terminalFn
        self.numEpisodes = numEpisodes
        self.cur_episode = -1
        
        # initialize q-values to be 0
        self.qvals = defaultdict(lambda: 0.1) # keys: (state, action), value: qval
        self.e_greedy_probs = dict() # key: (state, action), value: probs
        self.greedy_probs = dict()
        
        self.anneal_sigma = sigma >= 2
        self.anneal_type = 'linear' if sigma == 2 else 'exp'
        self.sigma = sigma if sigma < 1 else 1
    
    def startEpisode(self, initialState):
        super().startEpisode()
        
        self.cur_episode += 1
        
        self.T = math.inf
        self.cur_timestep = 0
        self.tau = 0
        
        # storage for states, actions, and rewards
        self.states = dict()
        self.actions = dict()
        self.deltas = dict()
        self.qvals_time = defaultdict(int) # key: (time, state, action), value: qval
        self.greedy_probs_time = dict()
        self.sigmas = dict()
        self.rhos = dict()

        self.states[0] = initialState
        self.actions[0] = self.get_e_greedy_action(self.states[0])
        self.qvals_time[0] = self.qvals[(self.states[0], self.actions[0])]
        self.sigmas[0] = 0
        self.rhos[0] = 0
    
    #def getSigma(self):
    #    return np.random.rand()
    
    def getQValue(self, state, action):
        """ Returns Q(state,action)
            Should return 0.0 if we have never seen a state or the Q node value otherwise """
        return self.qvals[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return 0
        
        qvals = [(self.getQValue(state, legal_act), legal_act) for legal_act in legalActions]
        highest_qval_action = max(qvals)
        return highest_qval_action[0]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        
        qvals = [(self.getQValue(state, legal_act), legal_act) for legal_act in legalActions]
       
        max_qval = max(qvals)[0] 
        all_max = [action for qval, action in qvals if (math.isnan(max_qval) and math.isnan(qval)) or qval == max_qval]
        return random.choice(all_max) # random max

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        
        if self.cur_timestep >= self.T:
            # don't do anything else
            return None
        
        return self.actions[self.cur_timestep]
        
    def get_e_greedy_action(self, state):
        legalActions = self.getLegalActions(state)        
        if len(legalActions) == 0:
            return None
        
        return np.random.choice(legalActions, p=[self.get_e_greedy_prob(state, act) for act in legalActions])
    
    def update(self, state, action, nextState, reward, update_vals):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
        """
        
        if not update_vals:
            # always choose greedy action
            self.states[self.cur_timestep+1] = nextState
            self.actions[self.cur_timestep+1] = self.get_e_greedy_action(self.states[self.cur_timestep+1])
            #print(self.actions[self.cur_timestep+1])
            self.cur_timestep += 1
            return
        
        if self.cur_timestep < self.T:
            # action was taken, so store next state and reward
            assert reward is not None and nextState is not None
            
            r = reward
            self.states[self.cur_timestep+1] = nextState # store next state s_t+1
            
            if self.terminalFn(self.states[self.cur_timestep+1]):
                self.T = self.cur_timestep + 1
                self.deltas[self.cur_timestep] = r - self.qvals_time[self.cur_timestep]
                self.sigmas[self.cur_timestep+1] = 0
                self.greedy_probs_time[self.cur_timestep+1] = 0
                
            else:
                # select & store an action A_t+1 ~ u(* | S_t+1)
                self.actions[self.cur_timestep+1] = self.get_e_greedy_action(self.states[self.cur_timestep+1])
                
                # select & store sigma_t+1
                self.sigmas[self.cur_timestep+1] = self.sigma
                
                # store Q(S_t+1, A_t+1) as Q_t+1
                self.qvals_time[self.cur_timestep+1] = self.qvals[(self.states[self.cur_timestep+1], self.actions[self.cur_timestep+1])]
                
                # store d_t = R + (gamma * sigma_t+1 * Q_t+1) + (gamma * (1 - sigma_t+1) * SUM_a(PI(a|S_t+1) * Q(S_t+1, a))) - Q_t
                delta = reward + (self.discount * self.sigmas[self.cur_timestep+1] * self.qvals_time[self.cur_timestep+1])
                sum_acts = 0
                for act in self.getLegalActions(self.states[self.cur_timestep+1]):
                    sum_acts += (self.get_greedy_prob(self.states[self.cur_timestep+1], act) * self.qvals[(self.states[self.cur_timestep+1], act)])
                delta += (self.discount * (1 - self.sigmas[self.cur_timestep+1]) * sum_acts) - self.qvals_time[self.cur_timestep]
                self.deltas[self.cur_timestep] = delta
                
                # store PI(A_t+1 | S_t+1) as PI_t+1
                self.greedy_probs_time[self.cur_timestep+1] = self.get_greedy_prob(self.states[self.cur_timestep+1], self.actions[self.cur_timestep+1])
                
                # store PI(A_t+1 | S_t+1) / u(A_t+1 | S_t+1) as rho_t+1
                e_greedy_prob = self.get_e_greedy_prob(self.states[self.cur_timestep+1], self.actions[self.cur_timestep+1])
                #if e_greedy_prob > self.epsilon / len(self.getLegalActions(self.states[self.cur_timestep+1])):
                #    e_greedy_prob = (1 - self.epsilon) + (self.epsilon / len(self.getLegalActions(self.states[self.cur_timestep+1])))
                self.rhos[self.cur_timestep+1] = self.greedy_probs_time[self.cur_timestep+1] / e_greedy_prob
                        
        # tau: time whose estimate is being updated
        self.tau = self.cur_timestep - self.n + 1
        
        if self.tau >= 0:
            rho = 1
            e = 1
            g = self.qvals_time[self.tau]
            
            #print("Range: ", self.tau, "to min(", self.tau+self.n-1, "+1,", self.T-1+1, "+1)")
            for k in range(self.tau, min(self.tau+self.n-1, self.T-1)+1):
                g += (e * self.deltas[k])
                
                e = self.discount * e * ( ((1 - self.sigmas[k+1])*self.greedy_probs_time[k+1]) + self.sigmas[k+1] )
                rho = rho * (1 - self.sigmas[k] + (self.sigmas[k] * self.rhos[k]))
                #print(rho, "\t", self.sigmas[k%self.n], self.rhos[k%self.n])
            
            # update q-value in direction of g
            prev_qv = self.qvals[(self.states[self.tau], self.actions[self.tau])]
            self.qvals[(self.states[self.tau], self.actions[self.tau])] = prev_qv + (self.alpha * rho * (g - prev_qv))
            
            self.update_probs(self.states[self.tau])
        
        self.cur_timestep += 1
    
    def update_probs(self, state):
        # for all actions, choose them with prob. epsilon/#legalactions
        legalActions = self.getLegalActions(state)        
        if len(legalActions) == 0:
            return
        
        qvals = [(self.getQValue(state, legal_act), legal_act) for legal_act in legalActions]
        
        max_qval = max(qvals)[0]
        all_max = [action for qval, action in qvals if (math.isnan(max_qval) and math.isnan(qval)) or qval == max_qval]
        
        # otherwise, set max actions...
        for act in legalActions:
            if act in all_max:
                self.e_greedy_probs[(state, act)] = ((1 - self.epsilon)/len(all_max)) + (self.epsilon/len(legalActions))
            else:
                self.e_greedy_probs[(state, act)] = (self.epsilon/len(legalActions))
        
        sum_probs = 0
        for action in legalActions:
            sum_probs += self.e_greedy_probs[(state, action)]
        #print(sum_probs)
        assert math.isclose(sum_probs, 1)
        
        # compute greedy probs
        
        for act in legalActions:
            if act in all_max:
                self.greedy_probs[(state, act)] = 1/len(all_max)
            else:
                self.greedy_probs[(state, act)] = 0
        
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
    
    def should_end_episode(self):
        return True if self.tau == self.T - 1 else False
    
    def get_greedy_prob(self, state, action):
        legalActions = self.getLegalActions(state)
        
        if all((state, legalAct) not in self.greedy_probs for legalAct in legalActions):
            # return uniform probability
            return 1/len(legalActions)
        else:
            return self.greedy_probs[(state, action)]
        
    def get_e_greedy_prob(self, state, action):
        legalActions = self.getLegalActions(state)
        
        if all((state, legalAct) not in self.e_greedy_probs for legalAct in legalActions):
            # return uniform probability
            #print("1/l", 1/len(legalActions))
            return 1/len(legalActions)
        else:
            #print(self.e_greedy_probs[(state, action)])
            return self.e_greedy_probs[(state, action)]
    
    def stopEpisode(self):
        super().stopEpisode()
        
        if self.anneal_sigma:
            if self.anneal_type == 'linear':
                self.sigma -= 1/self.numEpisodes
            elif self.anneal_type == 'exp':
                self.sigma = 1 * math.exp(-self.cur_episode / self.numEpisodes)
            else:
                raise Exception("unknown anneal type")