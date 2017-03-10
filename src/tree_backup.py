from game import *
from learningAgents import ReinforcementAgent
import fileio
import numpy as np

import random,util,math
from collections import defaultdict

class NStepTreeBackupAgent(ReinforcementAgent):
    def __init__(self, discount, alpha, epsilon, actionFn, n, terminalFn):
        ReinforcementAgent.__init__(self, gamma=discount, alpha=alpha, epsilon=epsilon, actionFn=actionFn)
        
        assert n > 0
        self.n = n
        self.terminalFn = terminalFn
        
        # initialize q-values to be 0
        self.qvals = defaultdict(lambda: 0.1) # keys: (state, action), value: qval
        self.probs = dict() # key: (state, action), value: probs
    
    def startEpisode(self, initialState):
        super().startEpisode()
        
        self.T = math.inf
        self.cur_timestep = 0
        self.tau = 0
        
        # storage for states, actions, and rewards
        self.states = dict()
        self.actions = dict()
        self.deltas = dict()
        self.qvals_time = defaultdict(int) # key: (time, state, action), value: qval
        self.probs_time = dict()

        self.states[0] = initialState
        self.actions[0] = self.get_e_greedy_action(self.states[0])
        self.qvals_time[0] = self.qvals[(self.states[0], self.actions[0])]
    
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
        
        return np.random.choice(legalActions, p=[self.get_prob(state, act) for act in legalActions])
    
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
            
            r = reward
            #print("Reward: ", reward)
            self.states[self.cur_timestep+1] = nextState # store next state s_t+1
            
            if self.terminalFn(self.states[self.cur_timestep+1]):
                self.T = self.cur_timestep + 1
                self.deltas[self.cur_timestep] = r - self.qvals_time[self.cur_timestep]
                self.probs_time[self.cur_timestep+1] = 0
                #print(self.cur_timestep, self.deltas[self.cur_timestep])
                
                #if reward == 1:
                #    print("Reward 1, terminal, d[", self.cur_timestep, "]:", self.deltas[self.cur_timestep])
            else:
                sum_acts = 0
                for act in self.getLegalActions(self.states[self.cur_timestep+1]):
                    sum_acts += (self.get_prob(self.states[self.cur_timestep+1], act) * self.qvals[(self.states[self.cur_timestep+1], act)])
                self.deltas[self.cur_timestep] = r + (self.discount * sum_acts) - self.qvals_time[self.cur_timestep]
                
                # select arbitrarily and store an action as a_t+1
                self.actions[self.cur_timestep+1] = self.get_e_greedy_action(self.states[self.cur_timestep+1]) #random.choice(self.getLegalActions(self.states[self.cur_timestep+1]))
                
                # store Q(S_t+1, A_t+1) as Q_t+1
                self.qvals_time[self.cur_timestep+1] = self.qvals[(self.states[self.cur_timestep+1], self.actions[self.cur_timestep+1])]
                
                # store PI(A_t+1 | S_t+1) as PI_t+1
                self.probs_time[self.cur_timestep+1] = self.get_prob(self.states[self.cur_timestep+1], self.actions[self.cur_timestep+1])
                #print("**T+1:", self.cur_timestep+1)
                
                #if reward == 1:
                #    print("Reward 1, non-terminal, d[", self.cur_timestep, "]:", self.deltas[self.cur_timestep])
                
        # tau: time whose estimate is being updated
        self.tau = self.cur_timestep - self.n + 1
        
        if self.tau >= 0:
            e = 1
            g = self.qvals_time[self.tau]
            
            for k in range(self.tau, min(self.tau+self.n-1, self.T-1)+1):
                g += (e * self.deltas[k])                
                e = self.discount * e * self.probs_time[k+1]

            # update q-value in direction of g
            self.qvals[(self.states[self.tau], self.actions[self.tau])] = self.qvals[(self.states[self.tau], self.actions[self.tau])] + (self.alpha * (g - self.qvals[(self.states[self.tau], self.actions[self.tau])]))
            
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
                self.probs[(state, act)] = ((1 - self.epsilon)/len(all_max)) + (self.epsilon/len(legalActions))
            else:
                self.probs[(state, act)] = (self.epsilon/len(legalActions))
        
        sum_probs = 0
        for action in legalActions:
            sum_probs += self.probs[(state, action)]
        #print(sum_probs)
        assert math.isclose(sum_probs, 1)
    
    def get_last_return_segment():
        return (self.discount ** self.n) * self.getQValue(self.states[self.tau+self.n], self.actions[self.tau+self.n])
    
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
    
    def should_end_episode(self):
        return True if self.tau == self.T - 1 else False
    
    def get_prob(self, state, action):
        legalActions = self.getLegalActions(state)
        
        if all((state, legalAct) not in self.probs for legalAct in legalActions):
            # return uniform probability
            return 1/len(legalActions)
        else:
            return self.probs[(state, action)]
