from game import *
from learningAgents import ReinforcementAgent

import random,util,math
import numpy as np
from collections import defaultdict

class QLearningAgent(ReinforcementAgent):
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.qvals = defaultdict(lambda: 0) # key: (state, action), value: qvals

    def getQValue(self, state, action):
        return self.qvals[(state, action)]

    def computeValueFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return 0
        
        qvals = [(self.getQValue(state, legal_act), legal_act) for legal_act in legalActions]
        highest_qval_action = max(qvals)
        return highest_qval_action[0]
    
    def computeActionFromQValues(self, state):
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        
        qvals = [(self.getQValue(state, legal_act), legal_act) for legal_act in legalActions]
       
        max_qval = max(qvals)[0] 
        all_max = [action for qval, action in qvals if (math.isnan(max_qval) and math.isnan(qval)) or qval == max_qval]
        return random.choice(all_max) # random max
    
    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        action = None
    
        if len(legalActions) == 0:
        	return None
        
        if util.flipCoin(self.epsilon):
        	action = random.choice(legalActions)
        else:
        	action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward, update_vals=True):
        if not update_vals:
            return
        
        legal_actions = self.getLegalActions(nextState)
        max_qval = self.computeValueFromQValues(nextState)
        target = reward + (self.discount * max_qval)
        
        cur_qval = self.qvals[(state, action)]
        new_qval = cur_qval + (self.alpha * (target - cur_qval))
        self.qvals[(state, action)] = new_qval

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
    
    def startEpisode(self, initialState):
        super().startEpisode()

global optionid
optionid = 0

class Option():
    def __init__(self, policyFn, terminationFn, initiationSet, primitive=False, primitiveName=None, target=None):
        self.getAction = policyFn
        self.shouldTerminate = terminationFn
        self.initiationSet = initiationSet
        self.primitive = primitive
        self.primitiveName = primitiveName
        self.target = target
        
        global optionid
        self.idnum = optionid
        optionid += 1
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.idnum == other.idnum
        if isinstance(other, str):
            return self.primitiveName == other
        return False

class SMDPQLearningAgent(ReinforcementAgent):
    def __init__(self, options, **args):
        ReinforcementAgent.__init__(self, **args)
        self.options = options
        self.qvals = defaultdict(lambda: 0) # key: (state, option), value: qvals

    def startEpisode(self, initialState):
        super().startEpisode()
        
        self.cur_option = None # list of actions
        self.cur_option_action_count = -1
        self.start_option_state = None
        self.cur_total_reward = 0

    def getQValue(self, state, option):
        opt_index = self.options.index(option)
        return self.qvals[(state, opt_index)]
    
    def setQValue(self, state, option, qval):
        opt_index = self.options.index(option)
        self.qvals[(state, opt_index)] = qval
    
    def computeValueFromQValues(self, state):
        legalOptions = self.getLegalOptions(state)
        if len(legalOptions) == 0:
            return 0
        
        qvals = [self.getQValue(state, legal_option) for legal_option in legalOptions]
        return max(qvals)
    
    def computeOptionFromQValues(self, state):
        legalOptions = self.getLegalOptions(state)
        if len(legalOptions) == 0:
            return None
        
        qvals = [(self.getQValue(state, legal_option), legal_option) for legal_option in legalOptions]
       
        max_qval = max([qv for qv, opt in qvals]) 
        all_max = [option for qval, option in qvals if (math.isnan(max_qval) and math.isnan(qval)) or qval == max_qval]
        return random.choice(all_max) # random max
    
    def getAction(self, state):
        if self.cur_option is None or self.cur_option.shouldTerminate(state):
            # we don't have an option yet, or we are done the current option, so get a new one e-greedily
            legalOptions = self.getLegalOptions(state)
            if len(legalOptions) == 0:
            	return None
        
            option = None
            if util.flipCoin(self.epsilon):
            	option = random.choice(legalOptions)
            else:
            	option = self.computeOptionFromQValues(state)
            
            self.cur_option = option
            self.cur_option_action_count = 0
            self.cur_total_reward = 0
            self.start_option_state = state
        
        #print("At state", state, ", should terminate? ", self.cur_option.shouldTerminate(state))
        
        self.cur_option_action_count += 1
        #print("Current option says go:", self.cur_option.getAction(state))
        return self.cur_option.getAction(state)

    def update(self, state, action, nextState, reward, update_vals=True):
        if not update_vals:
            return
        
        # only update if we have an option and have just finished it.
        if self.cur_option is not None:
            self.cur_total_reward += (self.discount**(self.cur_option_action_count-1)) * reward
            if self.cur_option.shouldTerminate(nextState):
                #max_qval = self.computeValueFromQValues(nextState)
            
                legalOptions = self.getLegalOptions(nextState)
                qvals = [self.getQValue(nextState, legal_option) for legal_option in legalOptions]
                max_qval = 0 if len(legalOptions) == 0 else max(qvals)
            
                timestep_delta = self.cur_option_action_count-1
                target = self.cur_total_reward + ((self.discount**timestep_delta) * max_qval)
            
                cur_qval = self.getQValue(self.start_option_state, self.cur_option)
                new_qval = cur_qval + (self.alpha * (target - cur_qval))
                self.setQValue(self.start_option_state, self.cur_option, new_qval)
                #print("Updating", self.start_option_state, self.cur_option, new_qval, max_qval, qvals, legalOptions)                

    def getPolicy(self, state):
        option_for_state = self.computeOptionFromQValues(state)
        if option_for_state is None:
            return None
        return option_for_state.getAction(state)
        #return self.computeActionFromQValues(state)

    def getValue(self, state):
        val = self.computeValueFromQValues(state)
        return val

class OneStepIntraOptionQLearningAgent(ReinforcementAgent):
    def __init__(self, options, **args):
        ReinforcementAgent.__init__(self, **args)
        self.options = options
        self.qvals = defaultdict(lambda: 0) # key: (state, option), value: qvals

    def startEpisode(self, initialState):
        super().startEpisode()
        
        self.cur_option = None # list of actions
        self.cur_option_action_count = -1
        self.start_option_state = None

    def getQValue(self, state, option):
        opt_index = self.options.index(option)
        return self.qvals[(state, opt_index)]
    
    def setQValue(self, state, option, qval):
        opt_index = self.options.index(option)
        self.qvals[(state, opt_index)] = qval
    
    def computeValueFromQValues(self, state):
        legalOptions = self.getLegalOptions(state)
        if len(legalOptions) == 0:
            return 0
        
        qvals = [self.getQValue(state, legal_option) for legal_option in legalOptions]
        return max(qvals)
    
    def computeOptionFromQValues(self, state):
        legalOptions = self.getLegalOptions(state)
        if len(legalOptions) == 0:
            return None
        
        qvals = [(self.getQValue(state, legal_option), legal_option) for legal_option in legalOptions]
       
        max_qval = max([qv for qv, opt in qvals]) 
        all_max = [option for qval, option in qvals if (math.isnan(max_qval) and math.isnan(qval)) or qval == max_qval]
        return random.choice(all_max) # random max
    
    def getAction(self, state):
        '''
        legalOptions = self.getLegalOptions(state)
        if len(legalOptions) == 1:
            return 'exit'
        
        return np.random.choice(['north', 'south', 'east', 'west'])
    
    def getActionGreedy(self, state):
        '''
        if self.cur_option is None or self.cur_option.shouldTerminate(state):
        
            # get the best option e-greedily.        
            legalOptions = self.getLegalOptions(state)
            if len(legalOptions) == 0:
            	return None
    
            option = None
            if util.flipCoin(self.epsilon):
            	option = random.choice(legalOptions)
            else:
            	option = self.computeOptionFromQValues(state)
        
            self.cur_option = option
            self.cur_option_action_count = 0
            self.start_option_state = state
        
            #print("At state", state, ", should terminate? ", self.cur_option.shouldTerminate(state))
        
        self.cur_option_action_count += 1
        #print("Current option says go:", self.cur_option.getAction(state))
        return self.cur_option.getAction(state)

    def update(self, state, action, nextState, reward, update_vals=True):
        if not update_vals:
            return
        
        # update the options that are consistent with taking the current action in the given state.
        for option in self.options:
            if state in option.initiationSet and option.getAction(state) == action:
                legalOptions = self.getLegalOptions(nextState)
                qvals = [self.getQValue(nextState, legal_option) for legal_option in legalOptions]
                max_qval = 0 if len(legalOptions) == 0 else max(qvals)

                uval = ((1 - option.shouldTerminate(nextState)) * self.getQValue(nextState, option)) + (option.shouldTerminate(nextState) * max_qval)
        
                target = reward + (self.discount * uval)
        
                cur_qval = self.getQValue(state, option)
                new_qval = cur_qval + (self.alpha * (target - cur_qval))
                self.setQValue(state, option, new_qval)
    
    def getPolicy(self, state):
        option_for_state = self.computeOptionFromQValues(state)
        if option_for_state is None:
            return None
        return option_for_state.getAction(state)
        #return self.computeActionFromQValues(state)

    def getValue(self, state):
        val = self.computeValueFromQValues(state)
        return val

class SMDPIntraCombinedQLearningAgent(ReinforcementAgent):
    def __init__(self, options, **args):
        ReinforcementAgent.__init__(self, **args)
        self.options = options
        self.qvals = defaultdict(lambda: 0) # key: (state, option), value: qvals

    def startEpisode(self, initialState):
        super().startEpisode()
        
        self.cur_option = None # list of actions
        self.cur_option_action_count = -1
        self.start_option_state = None
        self.cur_total_reward = 0

    def getQValue(self, state, option):
        opt_index = self.options.index(option)
        return self.qvals[(state, opt_index)]
    
    def setQValue(self, state, option, qval):
        opt_index = self.options.index(option)
        self.qvals[(state, opt_index)] = qval
    
    def computeValueFromQValues(self, state):
        legalOptions = self.getLegalOptions(state)
        if len(legalOptions) == 0:
            return 0
        
        qvals = [self.getQValue(state, legal_option) for legal_option in legalOptions]
        return max(qvals)
    
    def computeOptionFromQValues(self, state):
        legalOptions = self.getLegalOptions(state)
        if len(legalOptions) == 0:
            return None
        
        qvals = [(self.getQValue(state, legal_option), legal_option) for legal_option in legalOptions]
       
        max_qval = max([qv for qv, opt in qvals]) 
        all_max = [option for qval, option in qvals if (math.isnan(max_qval) and math.isnan(qval)) or qval == max_qval]
        return random.choice(all_max) # random max
    
    def getAction(self, state):
        if self.cur_option is None or state not in self.cur_option.initiationSet or self.cur_option.shouldTerminate(state):
            # get the best option e-greedily.        
            legalOptions = self.getLegalOptions(state)
            if len(legalOptions) == 0:
                print("no legal opts in state", state)
                return None
    
            option = None
            if util.flipCoin(self.epsilon):
            	option = random.choice(legalOptions)
            else:
            	option = self.computeOptionFromQValues(state)
        
            self.cur_option = option
            self.cur_option_action_count = 0
            self.start_option_state = state
            self.cur_total_reward = 0
        
            #print("At state", state, ", should terminate? ", self.cur_option.shouldTerminate(state))
        
        self.cur_option_action_count += 1
        #print("Current option says go:", self.cur_option.getAction(state))
        
        action = self.cur_option.getAction(state)
        assert action is not None
        return action

    def update(self, state, action, nextState, reward, update_vals=True):
        if not update_vals:
            return
        
        if self.cur_option is None:
            return # first move
        
        # update the options that are consistent with taking the current action in the given state.
        for option in self.options:
            if state in option.initiationSet and option.getAction(state) == action:
                legalOptions = self.getLegalOptions(nextState)
                qvals = [self.getQValue(nextState, legal_option) for legal_option in legalOptions]
                max_qval = 0 if len(legalOptions) == 0 else max(qvals)

                uval = ((1 - option.shouldTerminate(nextState)) * self.getQValue(nextState, option)) + (option.shouldTerminate(nextState) * max_qval)
        
                target = reward + (self.discount * uval)
        
                cur_qval = self.getQValue(state, option)
                new_qval = cur_qval + (self.alpha * (target - cur_qval))
                self.setQValue(state, option, new_qval)
        
        self.cur_total_reward += (self.discount**(self.cur_option_action_count-1)) * reward
        if self.cur_option.shouldTerminate(nextState):
            # finished this option, so update the state we started at when this option was chosen      
            legalOptions = self.getLegalOptions(nextState)
            qvals = [self.getQValue(nextState, legal_option) for legal_option in legalOptions]
            max_qval = 0 if len(legalOptions) == 0 else max(qvals)
        
            timestep_delta = self.cur_option_action_count-1
            target = self.cur_total_reward + ((self.discount**timestep_delta) * max_qval)
        
            cur_qval = self.getQValue(self.start_option_state, self.cur_option)
            new_qval = cur_qval + (self.alpha * (target - cur_qval))
            self.setQValue(self.start_option_state, self.cur_option, new_qval)
    
    def getPolicy(self, state):
        option_for_state = self.computeOptionFromQValues(state)
        if option_for_state is None:
            return None
        return option_for_state.getAction(state)
        #return self.computeActionFromQValues(state)

    def getValue(self, state):
        val = self.computeValueFromQValues(state)
        return val

class SMDPIntraCombinedSarsaAgent(ReinforcementAgent):
    def __init__(self, options, **args):
        ReinforcementAgent.__init__(self, **args)
        self.options = options
        self.qvals = defaultdict(lambda: 0) # key: (state, option), value: qvals

    def startEpisode(self, initialState):
        super().startEpisode()
        
        self.cur_option = None # list of actions
        self.cur_option_action_count = -1
        self.start_option_state = None
        self.cur_total_reward = 0
        self.next_option = None
        self.next_action = None
        self.cur_timestep = 0
    
    def getQValue(self, state, option):
        opt_index = self.options.index(option)
        return self.qvals[(state, opt_index)]
    
    def setQValue(self, state, option, qval):
        opt_index = self.options.index(option)
        self.qvals[(state, opt_index)] = qval
    
    def computeValueFromQValues(self, state):
        legalOptions = self.getLegalOptions(state)
        if len(legalOptions) == 0:
            return 0
        
        qvals = [self.getQValue(state, legal_option) for legal_option in legalOptions]
        return max(qvals)
    
    def computeOptionFromQValues(self, state):
        legalOptions = self.getLegalOptions(state)
        if len(legalOptions) == 0:
            return None
        
        qvals = [(self.getQValue(state, legal_option), legal_option) for legal_option in legalOptions]
       
        max_qval = max([qv for qv, opt in qvals]) 
        all_max = [option for qval, option in qvals if (math.isnan(max_qval) and math.isnan(qval)) or qval == max_qval]
        return random.choice(all_max) # random max
    
    def getAction(self, state):
        new_option = False
        if self.cur_option is None:
            # initial move.
            assert self.cur_timestep == 0
            self.cur_option = self.getBestOption(state)
            new_option = True
        elif self.cur_option.shouldTerminate(state) or state not in self.cur_option.initiationSet:
            # set our current option to the one already chosen
            assert self.next_option is not None
            self.cur_option = self.next_option
            new_option = True
        
        if new_option:
            self.cur_option_action_count = 0
            self.start_option_state = state
            self.cur_total_reward = 0
        
        self.cur_option_action_count += 1
        
        if self.next_action is None:
            assert self.cur_timestep == 0
            self.next_action = self.cur_option.getAction(state)
        
        assert self.next_action is not None
        return self.next_action # already chosen in update()
        
    def getBestOption(self, state):
        legalOptions = self.getLegalOptions(state)
        if len(legalOptions) == 0:
        	return None
        
        #option = None
        #if util.flipCoin(self.epsilon):
        #	option = random.choice(legalOptions)
        #else:
        
        option = self.computeOptionFromQValues(state)
        assert option is not None
        return option
    
    def update(self, state, action, nextState, reward, update_vals=True):
        if self.cur_option is None: #or nextState == 'TERMINAL_STATE':
            return # first move        
        
        self.cur_timestep += 1
        #print(self.cur_timestep)
        
        # the next option will only be taken if the current option is about to terminate.
        # if cur option is not terminating, next option will not be taken since when we end the option,
        #   we will be in a different (unknown) state. but, we still need a next option for the intra-
        #   option sarsa update...
        
        self.next_option = self.getBestOption(nextState)
        if self.cur_option.shouldTerminate(nextState):
            # first, choose the new option that we will follow now.
            #print(nextState)
            
            #if nextState != 'TERMINAL_STATE':
            #    print("next opt primitive?", self.next_option.primitive)
            #    print("next option:", self.next_option.initiationSet, " and next state:", nextState, "target:", self.next_option.target)
            #else:
            #    print("terminal")
            self.next_action = 'exit' if nextState == 'TERMINAL_STATE' else self.next_option.getAction(nextState)
        else:
            # still executing current option, so choose action w.r.to it
            self.next_action = self.cur_option.getAction(nextState)
        
        if not update_vals:
            return
        
        # intra-opt q-vals
        # update the options that are consistent with taking the current action in the given state.
        for option in self.options:
            #print(state, action, option.initiationSet)
            if state in option.initiationSet and option.getAction(state) == action:
                #legalOptions = self.getLegalOptions(nextState)
                #qvals = [self.getQValue(nextState, legal_option) for legal_option in legalOptions]
                #max_qval = 0 if len(legalOptions) == 0 else max(qvals)
                
                target = reward
                if nextState != 'TERMINAL_STATE':
                    next_qval = 0
                    if int(option.shouldTerminate(nextState)) > 0:
                        print(option.shouldTerminate(nextState))
                        next_qval = self.getQValue(nextState, self.next_option)
                    uval = ((1 - int(option.shouldTerminate(nextState))) * self.getQValue(nextState, option)) + (int(option.shouldTerminate(nextState)) * next_qval)
                    target += (self.discount * uval)
                
                cur_qval = self.getQValue(state, option)
                new_qval = cur_qval + (self.alpha * (target - cur_qval))
                self.setQValue(state, option, new_qval)
                #print("Updating", state, option.initiationSet, new_qval)
        
        self.cur_total_reward += (self.discount**(self.cur_option_action_count-1)) * reward
        if self.cur_option.shouldTerminate(nextState):
            # finished this option, so update the state we started at when this option was chosen
            # (smdp q-learning)
            
            target = self.cur_total_reward
            if nextState != 'TERMINAL_STATE':
                next_qval = self.getQValue(nextState, self.next_option)            
                timestep_delta = self.cur_option_action_count-1
                target += ((self.discount**timestep_delta) * next_qval)
            
            cur_qval = self.getQValue(self.start_option_state, self.cur_option)
            new_qval = cur_qval + (self.alpha * (target - cur_qval))
            self.setQValue(self.start_option_state, self.cur_option, new_qval)
    
    def getPolicy(self, state):
        option_for_state = self.computeOptionFromQValues(state)
        if option_for_state is None:
            return None
        return option_for_state.getAction(state)
        #return self.computeActionFromQValues(state)
    
    def getValue(self, state):
        val = self.computeValueFromQValues(state)
        return val