from game import *
from learningAgents import ReinforcementAgent
import fileio
import numpy as np

import random,util,math
from collections import defaultdict

class NStepSarsaAgent(ReinforcementAgent):
    def __init__(self, discount, alpha, epsilon, actionFn, n, terminalFn):
        ReinforcementAgent.__init__(self, gamma=discount, alpha=alpha, epsilon=epsilon, actionFn=actionFn)
        
        assert n > 0
        self.n = n
        self.terminalFn = terminalFn
        
        # initialize q-values to be 0
        self.qvals = defaultdict(lambda: 0.1) # key: (state, action), value: qvals
        self.e_greedy_probs = dict() # key: (state, action), value: probs
        self.greedy_probs = dict()
    
    def startEpisode(self, initialState):
        super().startEpisode()
        
        self.T = math.inf
        self.cur_timestep = 0
        self.tau = 0
        
        # storage for states, actions, and rewards
        self.states = dict()
        self.actions = dict()
        self.rewards = dict()

        self.states[0] = initialState
        self.actions[0] = self.get_e_greedy_action(self.states[0])
    
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
        
        probs = [self.get_e_greedy_prob(state, act) for act in legalActions]
        #print(probs)
        return np.random.choice(legalActions, p=probs)
    
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
            self.cur_timestep += 1
            return
        
        if self.cur_timestep < self.T:
            # action was taken, so store next state and reward
            assert reward is not None and nextState is not None
            
            self.rewards[self.cur_timestep+1] = reward
            #print("Reward: ", self.rewards[self.cur_timestep+1])
            self.states[self.cur_timestep+1] = nextState
            
            if self.terminalFn(self.states[self.cur_timestep+1]):
                self.T = self.cur_timestep + 1
            else:
                # select and store action A_t+1 ~ PI(* | S_t+1)
                self.actions[self.cur_timestep+1] = self.get_e_greedy_action(self.states[self.cur_timestep+1])
        
        # tau: time whose estimate is being updated
        self.tau = self.cur_timestep - self.n + 1
        if self.tau >= 0:
            rho = 1
            for i in range(self.tau+1, self.get_max_rho()+1):
                rho *= self.get_greedy_prob(self.states[i], self.actions[i]) / self.get_e_greedy_prob(self.states[i], self.actions[i])
            
            g = 0
            for i in range(self.tau+1, min(self.tau+self.n, self.T)+1):
                g += (self.discount ** (i - self.tau - 1)) * self.rewards[i]
            if self.tau + self.n < self.T:
                g += self.get_last_return_segment()
            
            # update q-value in direction of G
            prev_qv = self.qvals[(self.states[self.tau], self.actions[self.tau])]
            self.qvals[(self.states[self.tau], self.actions[self.tau])] = prev_qv + (self.alpha * rho * (g - prev_qv))
            
            #if self.qvals[(self.states[self.tau], self.actions[self.tau])] > 1 or self.qvals[(self.states[self.tau], self.actions[self.tau])] < 0:
            #    print(self.qvals[(self.states[self.tau], self.actions[self.tau])])
            #    print(self.alpha, rho, g, prev_qv)
            #    print("update: ", (self.alpha * rho * (g - prev_qv)))
            
            self.update_probs(self.states[self.tau])
        
        self.cur_timestep += 1
    
    def get_max_rho(self):
        return min(self.tau+self.n-1, self.T-1)
    
    def update_probs(self, state):
        # for all actions, choose them with prob. epsilon/#legalactions
        legalActions = self.getLegalActions(state)        
        if len(legalActions) == 0:
            return
        
        qvals = [(self.getQValue(state, legal_act), legal_act) for legal_act in legalActions]
        #print(qvals)
        
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
            #print(action, self.e_greedy_probs[(state, action)])
            sum_probs += self.e_greedy_probs[(state, action)]
        #print(sum_probs)
        assert math.isclose(sum_probs, 1)
        
        # compute greedy probs
        
        for act in legalActions:
            if act in all_max:
                self.greedy_probs[(state, act)] = 1/len(all_max)
            else:
                self.greedy_probs[(state, act)] = 0
        
    def get_last_return_segment(self):
        return (self.discount ** self.n) * self.getQValue(self.states[self.tau+self.n], self.actions[self.tau+self.n])
    
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
            return 1/len(legalActions)
        else:
            return self.e_greedy_probs[(state, action)]

class NStepExpectedSarsaAgent(NStepSarsaAgent):
    def __init__(self, **args):
        super().__init__(**args)
    
    def get_max_rho(self):
        return min(self.tau+self.n-1, self.T-1)
    
    def get_last_return_segment(self):
        sum_acts = 0
        for action in self.getLegalActions(self.states[self.tau+self.n]):
            sum_acts += (self.get_greedy_prob(self.states[self.tau+self.n], action) * self.qvals[(self.states[self.tau+self.n], action)])
        
        return (self.discount ** self.n) * sum_acts