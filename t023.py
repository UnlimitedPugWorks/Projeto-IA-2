#Miguel Mota 90964 Pedro Moreira 90768 Grupo23
import random
import math


# LearningAgent to implement
# no knowledege about the environment can be used
# the code should work even with another environment
class LearningAgent:

    # init
    # nS maximum number of states
    # nA maximum number of action per state
    # qTable Ã© o tabela dos Q-values
    # alpha Learning factor
    # gamma Discount factor
    # epsilon Exploration factor
    def __init__(self, nS, nA):
        # define this function
        self.nS = nS
        self.nA = nA
        self.qTable = []
        self.frequency = []
        for i in range(0, nS):
            self.qTable.append([])
            self.frequency.append([])
            for j in range(0, nA):
                self.qTable[i].append(-math.inf)
                self.frequency[i].append(0)
        self.gamma = 0.9
        self.epsilon = 1
        self.alpha = 0.2
        self.N = 100

    # Select one action, used when learning
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontolearn(self, st, aa):
        # define this function
        # print("select one action to learn better"
        a = 0
        if random.random() > self.epsilon:
            a = self.exploit(st, aa)
        else:
            a = self.explore(st, aa)
        if self.epsilon > 0.05:
            self.epsilon = self.epsilon * 0.998
        # print("o conjunto e accoes a aprender Ã© " + str(aa))
        # print("o indice da accao que vai ser aprendida Ã© : " + str(a))
        # define this function
        return a

    def explore(self, st, aa):
        frequencyList = [self.frequency[st][x] for x in range(0, len(aa))]
        minaction = min(frequencyList)
        if (minaction > self.N):
            return self.exploit(st, aa)
        minindexList = []
        for x in range(0, len(frequencyList)):
            if frequencyList[x] == minaction:
                minindexList.append(x)
        a = random.choice(minindexList)
        return a

    def exploit(self, st, aa):
        a = 0
        actionQList = [self.qTable[st][x] for x in range(0, len(aa))]
        maxindexList = []
        maxaction = max(actionQList)
        for x in range(0, len(actionQList)):
            if actionQList[x] == maxaction:
                maxindexList.append(x)
        a = random.choice(maxindexList)
        return a

    # Select one action, used when evaluating
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontoexecute(self, st, aa):
        # define this function
        a = 0
        actionQList = [self.qTable[st][x] for x in range(0, len(aa))]
        maxaction = max(actionQList)
        maxindexList = []
        for x in range(0, len(actionQList)):
            if actionQList[x] == maxaction:
                maxindexList.append(x)
        a = random.choice(maxindexList)
        # print("o conjunto e accoes a executar Ã© " + str(aa))
        # print("a accao que vai ser executado Ã© : " + str(a))
        # print("select one action to see if I learned")
        return a

    def learn(self, ost, nst, a, r):
        # alpha = (1 / (1 + self.frequency[ost][a]))
        maxV = max(self.qTable[nst])
        if (math.isinf(maxV)):
            maxV = 0
        # error = r + self.gamma * maxV - self.qTable[ost][a]
        self.qTable[ost][a] = (r + self.gamma * maxV) if math.isinf(self.qTable[ost][a]) else self.qTable[ost][a] + self.alpha * (r + self.gamma * maxV -self.qTable[ost][a])
        # self.qTable[ost][a] = (1 - self.alpha) * self.qTable[ost][a] + (self.alpha) * error
        # print("o Q-value do estado " + str(ost) + " e da accao " + str(a) + " Ã©: " + str(self.qTable[ost][a]))
        self.frequency[ost][a] += 1
