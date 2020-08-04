'''
Agents from Sutton-Barto, Intro to RL
'''


class Agent():
    def act(self, state):
        pass
    def update(self, state, action, reward, next_state, next_action):
        pass
    def reset_e(self):
        pass
    def plan(self, n=0):
        pass
    def reset_q(self):
        pass



class Sarsa(Agent):
    
    def __init__(self, env, params = None):
        
        if params is None:
            params = {}
        
        self.env = env
        self.epsilon = params.get('epsilon', 0.1) #Exploration probability
        self.alpha = params.get('alpha', 0.01) # Learning rate
        self.gamma = params.get('gamma', 1) # Discount rate
        
        self.q_matrix = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        
        
    def update(self,state,action,reward,next_state,next_action):
        
        delta = (reward + self.gamma*self.q_matrix[next_state,next_action] - self.q_matrix[state,action])
        self.q_matrix[state,action] = self.q_matrix[state,action] + self.alpha*delta
        
    
    def act(self,state):
        
        if random.uniform(0,1)<self.epsilon:
            return env.action_space.sample()

        else:
            return np.argmax(self.q_matrix[state])     
        
class QLearn(Agent):
    
    def __init__(self, env, params = None):
        
        if params is None:
            params = {}
        
        self.env = env
        self.epsilon = params.get('epsilon', 0.1) #Exploration probability
        self.alpha = params.get('alpha', 0.01) # Learning rate
        self.gamma = params.get('gamma', 1) # Discount rate
        
        self.q_matrix = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        
        
    def update(self,state,action,reward,next_state,next_action):
        
        delta = (reward + self.gamma*np.max(self.q_matrix[next_state]) - self.q_matrix[state,action])
        self.q_matrix[state,action] = self.q_matrix[state,action] + self.alpha*delta
        
    
    def act(self,state):
        
        if random.uniform(0,1)<self.epsilon:
            return env.action_space.sample()

        else:
            return np.argmax(self.q_matrix[state]) 
        
class LambdaQLearn(Agent):
    
    def __init__(self, env, params=None):
        
        if params is None:
            params = {}
        
        self.env = env
        self.epsilon = params.get('epsilon', 0.1) #Exploration probability
        self.alpha = params.get('alpha', 0.01) # Learning rate
        self.gamma = params.get('gamma', 1) # Discount rate
        self.lambda_ = params.get('lambda',0.5) # Eligibility decay
        
        self.q_matrix = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.e_matrix = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        
        
    def update(self,state,action,reward,next_state,next_action):
        
        if self.q_matrix[next_state, next_action] == np.max(self.q_matrix[next_state]):
            max_a = next_action
        else:
            max_a = np.argmax(self.q_matrix[next_state])
        
        delta = (reward + self.gamma*self.q_matrix[next_state, max_a] - self.q_matrix[state, action])
        self.e_matrix[state, action] += 1
        
        self.q_matrix += self.alpha*delta*self.e_matrix
        
        if max_a == next_action:
            self.e_matrix *= self.gamma*self.lambda_
        else:
            self.reset_e()
        
    
    def act(self,state):
        
        if random.uniform(0,1)<self.epsilon:
            return env.action_space.sample()

        else:
            return np.argmax(self.q_matrix[state]) 
        
    def reset_e(self):
        self.e_matrix = np.zeros(self.e_matrix.shape)
        
class DynaQLearn(Agent):
    
    def __init__(self, env, params = None):
            
        if params is None:
            params = {}
            
        self.env = env
        self.epsilon = params.get('epsilon', 0.1) #Exploration probability
        self.alpha = params.get('alpha', 0.01) # Learning rate
        self.gamma = params.get('gamma', 1) # Discount rate
        
        self.q_matrix = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        
        self.model = np.zeros([self.env.observation_space.n, self.env.action_space.n, 2]) # extra 2 dimensions (R, S')
        self.model_bool = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        
        
    def update(self,state,action,reward,next_state,next_action):
        
        delta = (reward + self.gamma*np.max(self.q_matrix[next_state]) - self.q_matrix[state,action])
        self.q_matrix[state,action] += self.alpha*delta
        
        self.model_bool[state, action] = 1
        self.model[state, action, :] = reward, next_state 
        
    
    def act(self, state):
        
        if random.uniform(0,1)<self.epsilon:
            return env.action_space.sample()

        else:
            return np.argmax(self.q_matrix[state]) 
        
    def plan(self, n=1):
             
        if (self.model_bool.flatten().sum() > n):
            for _ in range(n):
                nonzero = np.transpose(np.nonzero(self.model_bool))
                pstate, paction = nonzero[np.random.choice(nonzero.shape[0])]
                preward, pnext_state = self.model[pstate, paction]

                pnext_state = int(pnext_state)

                pdelta = (preward + self.gamma*np.max(self.q_matrix[pnext_state]) - self.q_matrix[pstate,paction])
                self.q_matrix[pstate, paction] +=  self.alpha*pdelta
        
        
        