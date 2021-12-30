"""
Modified from wwagent.py written by Greg Scott

Modified to only do random motions so that this can be the base
for building various kinds of agent that work with the wwsim.py 
wumpus world simulation -----  dml Fordham 2019

# FACING KEY:
#    0 = up
#    1 = right
#    2 = down
#    3 = left

# Actions
# 'move' 'grab' 'shoot' 'left' right'

"""

'''
Name: Deepak Kumar Gunturu, MS Data Science, Class of 2022, CISC 6525: Artificial Inteligence 
Date: 12/18/2021

Objective: Using truth table enumeration model checking method to safely get an agent across to gold without falling into a pit and being swallowed by a Wumpus
For this, each position is computed along with its neighbors and the knowledge base, and models generated are used to satisfy the alpha which is the query. Then,
based on the probability of the alpha and KB, a safe move is generated for the agent. 
'''

from random import randint

# This is the class that represents an agent

class WWAgent:

    def __init__(self):
        self.max=4 # number of cells in one side of square world
        self.stopTheAgent=False # set to true to stop th agent at end of episode
        self.position = (0, 3) # top is (0,0)
        self.directions=['up','right','down','left']
        self.facing = 'right'
        self.arrow = 1
        self.percepts = (None, None, None, None, None)
        self.prevMove = None
        self.probKB = 0.01
        self.cnt = 0
        self.probKBAndAlpha = 0
        self.map = [[ self.percepts for i in range(self.max) ] for j in range(self.max)]
        print("New agent created")

    # Add the latest percepts to list of percepts received so far
    # This function is called by the wumpus simulation and will
    # update the sensory data. The sensor data is placed into a
    # map structured KB for later use
    
    def update(self, percept):
        self.percepts=percept
        #[stench, breeze, glitter, bump, scream]
        if self.position[0] in range(self.max) and self.position[1] in range(self.max):
            self.map[ self.position[0]][self.position[1]]=self.percepts
        # puts the percept at the spot in the map where sensed

    # Calculating the neighbors of a given position
    def calculatePositions(self):
    
        if self.position[1] == 3:
            if self.position[0] == 3:
                return [(2,3),(3,2)]

            if self.position[0] == 0:
                return [(2,0),(3,1)]

            else:
                return [(3,self.position[0]-1),(2,self.position[0]),(3,self.position[0]+1)]

        if self.position[1] == 0:

            if self.position[0] == 0:
                return [(1,0),(0,1)]

            if self.position[0] == 3:
                return [(0,2),(1,3)]

            else:
                return [(0,self.position[0]-1),(1,self.position[0],(0,self.position[0]+1))]

        if self.position[0] == 0 and self.position[1] != 3 and self.position[1] != 0:
            return [(self.position[1]-1,self.position[0]),(self.position[1],self.position[0]+1),(self.position[0]+1,self.position[1])]

        if self.position[0] == 3 and self.position[1] != 3 and self.position[1] != 0:
            return [(self.position[1]-1,self.position[0]),(self.position[1],self.position[0]-1),(self.position[1]+1,self.position[0])]

        else:
            return [(self.position[1]-1,self.position[0]),(self.position[1],self.position[0]-1),(self.position[1]+1,self.position[0]),(self.position[1],self.position[0]+1)]

    # Since there is no percept for location, the agent has to predict
    # what location it is in based on the direction it was facing
    # when it moved

    def calculateNextPosition(self,action):
        if self.facing=='up':
            self.position = (self.position[0],max(0,self.position[1]-1))
        elif self.facing =='down':
            self.position = (self.position[0],min(self.max-1,self.position[1]+1))
        elif self.facing =='right':
            self.position = (min(self.max-1,self.position[0]+1),self.position[1])
        elif self.facing =='left':
            self.position = (max(0,self.position[0]-1),self.position[1])
        return self.position

    # and the same is true for the direction the agent is facing, it also
    # needs to be calculated based on whether the agent turned left/right
    # and what direction it was facing when it did
    
    def calculateNextDirection(self,action):
        if self.facing=='up':
            if action=='left':
                self.facing = 'left'
            else:
                self.facing = 'right'
        elif self.facing=='down':
            if action=='left':
                self.facing = 'right'
            else:
                self.facing = 'left'
        elif self.facing=='right':
            if action=='left':
                self.facing = 'up'
            else:
                self.facing = 'down'
        elif self.facing=='left':
            if action=='left':
                self.facing = 'down'
            else:
                self.facing = 'up'

    # Entailment process for truth tables
    def entailment(self,prop,model):
        '''Check whether prop is true in model'''
        # assumes prop and model use the list/logic notation
        if isinstance(prop,str):
            return (prop,True) in model
        elif len(prop)==1:
            return self.entailment(prop[0],model)
        elif prop[0]=='not':
            return not self.entailment(prop[1],model)
        elif prop[1]=='and':
            return self.entailment(prop[0],model) and self.entailment(prop[2],model)
        elif prop[1]=='or':
            return self.entailment(prop[0],model) or self.entailment(prop[2],model)
        elif prop[1]=='implies':
            return (not self.entailment(prop[0],model)) or self.entailment(prop[2],model)
        return False

    # Model checking after enumerating truth table
    def modelcheck(self,symbols,model,KB,alpha):
        if len(symbols)==0:
            if self.entailment(KB,model):                
                self.probKB += 1
                if self.entailment(alpha,model):
                    self.probKBAndAlpha += 1
                    return self.entailment

            else:
                self.probKB += 0.01
                self.probKBAndAlpha += 0.003
                return False
        else:
            p = symbols[0]
            rest = list(symbols[1:len(symbols)])
            return self.modelcheck(rest,model+[(p,True)],KB,alpha) and self.modelcheck(rest,model+[(p,False)],KB,alpha)

    # this is the function that will pick the next action of
    # the agent. This is the main function that needs to be
    # modified when you design your new intelligent agent
    # right now it is just a random choice agent
    
    def action(self):

        KBforPits =  [ ['not', 'p30'] ], 'and', ['b30','implies',['p31','or','p20']], 'and', 'b31', 'implies', ['p30','or',['p21','or','p32']]
        KBforWumpus = [ ['not', 'w30'] ], 'and', ['s30','implies',['w31','or','w20']], 'and', 's31', 'implies', ['w30','or',['w21','or','w32']]
        bestProb = 0
        besProb = randint(0,1)    
        # test for controlled exit at end of successful gui episode
        if self.stopTheAgent:
            print("Agent has won this episode.")
            return 'exit' # will cause the episide to end
        
        #reflect action -- get the gold!
        if 'glitter' in self.percepts:
            print("Agent will grab the gold!")
            self.stopTheAgent=True
            return 'grab'

        if self.percepts[0] == None and self.percepts[1] == None and self.percepts[2] == None and self.percepts[2] == None and self.percepts[4] == None:
            print('Safe to move to next square')
            action = 'move'
            self.calculateNextDirection(action)
            
        def model_enum(symbols,model,initial):
            if len(symbols)==0:
                models.append(model)
            else:
                p = symbols[0]
                rest = list(symbols[1:len(symbols)])
                model_enum(rest,model+[(initial+p[1]+p[2],True)],initial)
                model_enum(rest,model+[(initial+p[1]+p[2],False)],initial)
                return

        if 'breeze' in self.percepts:

            newPossible = self.calculatePositions()
            symbols = []
            alpha = []
            models = []

            for i in range(0,len(newPossible)):
                symbols.append('b'+str(newPossible[i][0])+str(newPossible[i][1]))
                alpha.append('p'+str(newPossible[i][0])+str(newPossible[i][1]))
                
                if i != len(newPossible)-1:
                    alpha.append('or')
            
            print('\nSymbols for breeze for each neighbor:',symbols)
            print('Alpha for each neighbor:',alpha)
            model_enum(symbols,[],'p')
            print("\nEnumerated models for pits:\n")
            for i in range(0,len(models)):
                print(str(i+1)+'. '+str(models[i])) 
            print()   

            for i in models:
                self.modelcheck(symbols,i,KBforPits,alpha)
                print('Probability of action: ',self.probKBAndAlpha/self.probKB)

                if bestProb < self.probKBAndAlpha/self.probKB:
                    bestProb = self.probKBAndAlpha/self.probKB

        if 'stench' in self.percepts:

            newPossible = self.calculatePositions()
            symbols = []
            alpha = []
            models = []
            
            for i in range(0,len(newPossible)):
                symbols.append('s'+str(newPossible[i][0])+str(newPossible[i][1]))

                alpha.append('w'+str(newPossible[i][0])+str(newPossible[i][1]))
                
                if i != len(newPossible)-1:
                    alpha.append('or')
            
            print('\nSymbols for stench for each neighbor',symbols)
            print('Alpha for each neighbor:',alpha)
            model_enum(symbols,[],'w')
            print("\nEnumerated models for pits:\n")            
            for i in range(0,len(models)):
                print(str(i+1)+'. '+str(models[i]))
            print()

            for i in models:
                self.modelcheck(symbols,i,KBforWumpus,alpha)
                print('Probability of : ',self.probKBAndAlpha/self.probKB)

                if bestProb < self.probKBAndAlpha/self.probKB:
                    bestProb = self.probKBAndAlpha/self.probKB

        actionSelection = besProb
        if actionSelection > 0.9: 
            action = 'move'
            # predict the effect of this
            self.calculateNextPosition(action)
        else: 
            actionSelection=bestProb
            if actionSelection < 0:
                action = 'left'
            else:
                action= 'right'
            # predict the effect of this action
            self.calculateNextDirection(action)
        
        self.probKBAndAlpha = 0
        self.probKB = 0.01
            
        return action
