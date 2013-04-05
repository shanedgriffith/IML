#!/usr/bin/python

""""A Frogger implementation of ``Policy Shaping: Integrating Human Feedback with Reinforcement Learning''
 -Implements Bayesian Q-Learning from Dearden, Friedman, and Russell. ``Bayesian Q-Learning.'' AAAI. 1998. 
 -Implements Advise from Griffith, Subramanian, Scholz, Isbell, and Thomaz. ``Policy Shaping: Integrating Human Feedback with Reinforcement Learning.'' ICML. 2013. (under review)
 -Implements Action Biasing and Control Sharing from Knox and Stone. ``Combining manual feedback with subsequent MDP reward signals for reinforcement learning.'' AAMAS. 2010.
"""

import os
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.utils import TaskSpecVRLGLUE3


import random,math
from numpy.random.mtrand import dirichlet
from scipy import stats
      #This statistics package is used to call the Cumulative Distribution Function for Bayesian Q-learning.
      #from scipy: http://connectmv.com/tutorials/python-matlab-tutorial/dealing-with-distributions/
      #to install scipy: http://www.thisisthegreenroom.com/2011/installing-python-numpy-scipy-matplotlib-and-ipython-on-lion/#numpy

#This is the implementation of Bayesian Q-learning, together with the interface to Frogger. 
class BayesianQLearning():

  
  def __init__(self, gameType, save=False):
  #called for initializing learning parameters before episodic learning starts
        
        #variables for episodic learning 
        self.gamma = 0.8  #the discount factor
        self.lastState = None;
        self.lastAction = None;
        self.episodesSoFar = 0
        self.curEpisode = 0;
        self.VPIProbState = None;
        self.VPIProbData = [];
        
        #possible priors for the normal gamma distribution
        #the initial values are set to represent the uncertainty in the normal distribution over Q-values of q-learning
        #mean, precision, shape, scale
        #see `Murphy - normal gamma model' for how to tweak alpha and beta.
        prior1 = [0, 0.01, 1000, 0.00001];
        prior2 = [0, 0.1, 1.1, 0.01]; 
        prior3 = [0, 0.00001, 10000, 0.000001];
        self.priorhyperparameters=prior1;
        self.PrintParameterInitialization(); 
#        self.debugPrior();
        
        self.stateindex = {}; #stores the index for each state.
        self.knownstates = []; #stores each state that's found as the agent explores
        self.N = 500; #number of samples used to estimate the probability distribution over the optimal actions
        
        #variables for saving the data
        self.numfiles = 0;
        self.saveData = [];
        self.FILENAME = ''

  def setParams(self, flikelihood, fconsistency, iterations):
  #command line parameters
        self.flikelihood = flikelihood;
        self.fconsistency = fconsistency;
        self.iterations = iterations;

  def agent_init(self, taskSpecString, objectsInfo=None):
  #Frogger specific code
      TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
      if not TaskSpec.valid:
          print 'Task Spec could not be parsed: ' + taskSpecString

  def agent_start(self, observation):
  #called at the beginning of each episode
        self.ep_reward = 0;
        self.lastState = observation.doubleArray;
        
        #start acting
        self.lastAction = self.getAction(self.lastState);
        return self.lastAction

  def agent_step(self, reward, transition):
  #called after each action is performed.
        self.ep_reward += reward;

        nextState = transition.doubleArray;
        
        self.update(self.lastState, self.lastAction, nextState, reward);
        
        self.lastState = nextState;
        if self.lastState.find('t') == -1 : #only call this if it's not the terminal state.
            self.lastAction = self.getAction(nextState);
        
        return self.lastAction

  def getFileName(self):
  #get a file name for saving learning data
        filestr = ('exp_' + repr(self.flikelihood) + '_test_' + repr(self.fconsistency) + '_iteration_');
        
        count = -1;
        gotfile = True
        while gotfile :
            count += 1;
            gotfile = os.path.isfile('learningcurves/' + filestr + str(count) + '.txt');
        
        return 'learningcurves/' + filestr + str(count) + '.txt';

  def agent_end(self, reward):
  #Called after a terminal state is reached. 
        self.episodesSoFar += 1
        
        str = '{numepisodes},{score}\n'.format(numepisodes=(self.episodesSoFar), score=self.ep_reward)
        self.saveData.append(str);
        
        #Only save the data if episodic learning finished all the episodes
        if self.episodesSoFar == self.iterations :
            fname = self.getFileName();
            print 'Saving data to: ' + fname;
            f = open(fname, 'w');
            for s in self.saveData :
                f.write(s);
            f.close();
        
        return
        
  def agent_cleanup(self):
        return

  def PrintParameterInitialization(self):
  #Output the parameter values.
    print 'Parameters'
    print '  discount factor (gamma): ' + repr(self.gamma);
    print '  prior: ' + repr(self.priorhyperparameters);

  def debugPrior(self):
  #Outputs a range of Q-values sampled from the prior.
  #Used to tune the prior.
      for i in range(0, 10000):
          tow = random.gammavariate(self.priorhyperparameters[2], self.priorhyperparameters[3]);
          x = random.normalvariate(self.priorhyperparameters[0], math.sqrt(1/(self.priorhyperparameters[1]*tow)));

          print repr(x) + ','
      
      quit();

  def debugVPI(self, state):
  #Outputs the VPI values for all the state/action pairs
  #These are computed using the myopic-VPI exploration strategy.
      
      for i in range(0, 100) :
          vpilist = self.VPI(state)
          print 'iter ' + repr(i) + '; vpi: ' + repr(vpilist)
      
      quit();

  def DebugState(self, state):
      sidx = self.stateindex[state];
      alist = self.CreateOrderedListOfActions(state);
      print 'state:\n' + state + '\n' + repr(alist);
      print 'VPI: ' + repr(self.VPI(state));
      print 'statedata: ' + repr(self.knownstates[sidx]);

  def getLegalActions(self, state) :
  #How many actions available in Frogger. 
      return range(5);

  def getAction(self, str):
  #Called for action selection
      if not self.stateindex.has_key(str) :
          actions = self.getLegalActions(str);
          action = random.choice(actions);
      else :
          action = self.ActionSelection(str);

      return action;

  def update(self, str, action, nstr, reward):
  #Called upon the transition to a new state, nstr, after performing action in state str, and receiving reward.
  
    actions = self.getLegalActions(str);
    
    #If the state is new, save it
    self.AddStateToList(str, actions);
    
    #update the parameters to the NG distribution
    self.QValueUpdating(str, action, nstr, reward);

  def AddStateToList(self, state, actions):
  #Saves previously unknown states
      if not self.stateindex.has_key(state) :
          sidx = len(self.knownstates);
          self.stateindex[state] = sidx;
          stateinfo = [];

          for action in actions :
              stateinfo.append([action, self.priorhyperparameters]); #a list of hyperperparameters for each state
          
          self.knownstates.append(stateinfo);

  def SetHyperparameters(self, state, action, hyperparameters):
  #Setter method for the hyperparameters of the NG distribution for a specific state/action pair.
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      for i in range(0,len(statedata)) :
          if statedata[i][0] is action :
              statedata[i][1] = hyperparameters;

  def GetHyperparameters(self, state, action):
  #Getter method for the hyperparameters of the NG distribution for a specific state/action pair.
      if self.stateindex.has_key(state) :
          sidx = self.stateindex[state];
          statedata = self.knownstates[sidx];
          for i in range(0,len(statedata)) :
              if statedata[i][0] is action :
                  return statedata[i][1];
      else :
          return self.priorhyperparameters

  def IdentifyActionWithMaxExpectedValue(self, state):
  #Use the hyperparameters to identify the action with the maximum expected value, i.e., max hyperparameter[0]
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];

      maxaction = '';
      maxExpectedval = -100000000000;

      for i in range(0,len(statedata)) :
          action = statedata[i][0];
          posteriorhyperparameters = statedata[i][1];

          if maxExpectedval < posteriorhyperparameters[0] :
              maxExpectedval = posteriorhyperparameters[0];
              maxaction = action;
      
      return maxaction;

  def GetMomentsFromTheNormalGammaDistribution(self, r, hyperparameters):
  #Get the first and second moments from the hyperparameters.
      ER = hyperparameters[0];
      ER2 = (1.0 + hyperparameters[1])/hyperparameters[1] * hyperparameters[3]/(hyperparameters[2]-1) + hyperparameters[0]**2;
      
      M1 = r + self.gamma*ER;
      M2 = r**2 + 2*self.gamma*r*ER + self.gamma**2 * ER2;
      
      return [M1,M2];
  
  def GetThePosterior(self, priorhyperparameters, moments):
  #Use the prior and the moments to compute the posterior for the NG distribution. The posterior is also a NG distribution
      mu0 = priorhyperparameters[0];
      lambda0 = priorhyperparameters[1];
      alpha0 = priorhyperparameters[2];
      beta0 = priorhyperparameters[3];
      M1 = moments[0];
      M2 = moments[1];
      n = 1.0;
      
      mu1 = (1.0*lambda0*mu0 + n*M1)/(lambda0+n);
      lambda1 = lambda0+n;
      alpha1 = alpha0 + 0.5*n;
      beta1 = beta0 + 0.5*n*(M2-M1**2) + (n*lambda0*(M1-mu0)**2)/(2*(lambda0+n));
      
      return [mu1, lambda1, alpha1, beta1];

  def QValueUpdating(self, state, action, nextState, reward):
  #Update the NG distributions for the Q values of the last state
      #for the observed nextState, identify the action at with the max expected reward.
      nextstatehyperparameters = self.priorhyperparameters;
      if self.stateindex.has_key(nextState) :
          at = self.IdentifyActionWithMaxExpectedValue(nextState);
          nextstatehyperparameters = self.GetHyperparameters(nextState, at);
      
      #compute the moments M1 and M2 using r, gamma, and these N samples.
      moments = self.GetMomentsFromTheNormalGammaDistribution(reward, nextstatehyperparameters);
      
      #recompute the posterior of the state/action pair
      testparameters = self.GetHyperparameters(state, action);
      posteriorhyperparameters = self.GetThePosterior(testparameters, moments);
      self.SetHyperparameters(state, action, posteriorhyperparameters);
      
  def SampleFromTheNormalGammaDistribution(self, hyperparameters):
  #samples one estimated Q value from the hyperparameters of one NG distribution
      tow = random.gammavariate(hyperparameters[2], hyperparameters[3]);
      x = random.normalvariate(hyperparameters[0], math.sqrt(1/(hyperparameters[1]*tow)));
      return x;

  def EstimateTheProbabilityThatActionsAreOptimal(self, state):
  #Sample self.N Q-values as a way to estimate the probability an action is optimal.
      
      nSamples = self.N;
      
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      
      countlist = [];
      for i in range(0,len(statedata)) :
          countlist.append(0);
      
      #count the number of times each action has the max Q-value
      for i in range(0, nSamples) :
          maxq = -100000000;
          maxj = 0;
          for j in range(0, len(statedata)) :
              action = statedata[j][0];
              hyperparameters = statedata[j][1];
              estimatedq = self.SampleFromTheNormalGammaDistribution(hyperparameters);

              if estimatedq > maxq :
                  maxq = estimatedq;
                  maxj = j;
          countlist[maxj] = countlist[maxj] + 1;
      
      #normalize the distribution
      for i in range(0, len(countlist)) :
          countlist[i] = (1.0*countlist[i])/nSamples;
          
      return countlist;
  
  def EstimateTheProbabilityWithVPI(self, state):
  #sample the E[u] + VPI to estimate the probability an action is optimal.
      
      nSamples = self.N;
      smallamount = 0;      #This value can ensure that an action is taken with a small probability. 
      
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      
      countlist = [];
      for i in range(0,len(statedata)) :
          countlist.append(smallamount);
      
      #Count the number of times an action is optimal according to myopic VPI
      for i in range(0, nSamples) :
          maxq = -100000000;
          maxj = 0;
          vpilist = self.VPI(state);
          
          for j in range(0, len(statedata)) :
              action = statedata[j][0];
              curvpi = 0;
              for k in range(0, len(vpilist)) :
                  if vpilist[k][1] == action :
                      curvpi = vpilist[k][0];
                      break;
              if curvpi > maxq :
                  maxq = curvpi;
                  maxj = j;

          countlist[maxj] = countlist[maxj] + 1;
      
      #normalize the distribution
      for i in range(0, len(countlist)) :
          countlist[i] = (1.0*countlist[i])/(nSamples+len(countlist)*smallamount);

      return countlist;
  
  def CreateOrderedListOfActions(self, state):
  #Sort the actions by their expected value. This is used to compute the VPI for each action.
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      
      #Get the expected values for all the actions.
      alist = [];
      for i in range(0, len(statedata)) :
          action = statedata[i][0];
          expectedval = statedata[i][1][0];
          alist.append([expectedval,action]);
      
      #Selection sort.
      for i in range(0, len(alist)) :
          maxidx = i;
          maxval = alist[i][0];
          for j in range(i+1, len(alist)) :
              if alist[j][0] > maxval :
                  maxval = alist[j][0];
                  maxidx = j;
          temp = alist[i];
          alist[i] = alist[maxidx];
          alist[maxidx] = temp;
      
      return alist;

  def getCvalue(self, hyperparameters):
  #The c value used to compute myopic-VPI. See Dearden et al. 1998.
      #a gamma distribution with one parameter corresponds to gamma(a,1)
      mu = hyperparameters[0];
      lambdu = hyperparameters[1];
      alpha = hyperparameters[2];
      beta = hyperparameters[3];
      
      num1 = (alpha * random.gammavariate(alpha+0.5, 1) * math.sqrt(beta));
      denom1 = ((alpha-0.5)*random.gammavariate(alpha,1)*random.gammavariate(0.5,1)*alpha*math.sqrt(2*lambdu));
      prod1 = (1.0 + mu**2/(2*alpha));
      pow1 = -1.0 * alpha+0.5;
      
      c = (num1/denom1) * prod1**pow1;
      return c;
  
  def ProbQLessThanX(self, hyperparameters, x):
  #Estimates the probability that a particular Q-value is less than X.
      mu = hyperparameters[0];
      lambdu = hyperparameters[1];
      alpha = hyperparameters[2];
      beta = hyperparameters[3];
      
      param1 = (x-mu)*((lambdu*alpha/beta)**0.5);
      param2 = 2*alpha;
      
      #Call the cumulative t-distribution function.
      return stats.t.cdf(param1, param2);

  def VPI(self, state):
  #Implements myopic-VPI as described in Dearden et al. 1998.
      alist = self.CreateOrderedListOfActions(state);
      
      VPIlist = [];
      
      if len(alist) < 2 :
          return alist;
      
      if self.VPIProbState != self.lastState :
          self.VPIProbState = self.lastState;
          self.VPIProbData = [];
          
          for i in range(0, len(alist)) :
              hyperparameters = self.GetHyperparameters(state, alist[i][1]);
              vpi = 0;
              if i == 0 :
                  #a = a1
                  vpi = (alist[1][0] - alist[i][0]) * self.ProbQLessThanX(hyperparameters, alist[1][0]);
              else :
                  #a != a1
                  vpi = (alist[i][0] - alist[0][0]) * (1.0 - self.ProbQLessThanX(hyperparameters, alist[0][0]));
              
              myopicvpi = hyperparameters[0] + vpi;
              self.VPIProbData.append(myopicvpi);
      
      #After the VPI value is computed, add c to it. 
      for i in range(0, len(alist)) :
          hyperparameters = self.GetHyperparameters(state, alist[i][1]);
          c = self.getCvalue(hyperparameters);
          VPIlist.append([self.VPIProbData[i]+c, alist[i][1]]);
      
      return VPIlist;

  def ActionSelection(self, state):
  #Select the action that maximizes myopic-VPI
      myopicvpi = self.VPI(state);
      maxi = 0
      max = myopicvpi[0][0];
      for i in range(0, len(myopicvpi)):
          if myopicvpi[i][0] > max :
              max = myopicvpi[i][0];
              maxi = i;
      
      action = myopicvpi[maxi][1];
      return action;

#General superclass for learning from human feedback. This generates feedback using an oracle, but doesn't process received feedback.
class BayesianRLHumanFeedback(BayesianQLearning):
  def __init__(self, gameType, save=False):
      BayesianQLearning.__init__(self, gameType, save=False);
      
      self.ostateindex = {}; #holds the correct action for each state, as specific by an oracle
      
      self.LoadOracleQVals();
  
  def AddStateToList(self, state, actions):
  #The state is redefined to include counters for received feedback.
      if not self.stateindex.has_key(state) :
          sidx = len(self.knownstates);
          self.stateindex[state] = sidx;
          stateinfo = [];

          for action in actions :
              stateinfo.append([action, self.priorhyperparameters, [0,0], 0]); #[action, hyperparameters, feedback counts, feedback rcvd] for each state
          
          self.knownstates.append(stateinfo);
  
  def update(self, str, action, nextState, reward):
  #The update function is redefined to include the oracle, which generates the feedback. 
      BayesianQLearning.update(self, str, action, nextState, reward);
      self.ProcessOracleFeedback(str, action);
      
  def GenerateOracleFeedback(self, state, action):
  #generates feedback according to the desired test value.
  #CONSISTENCY, here, represents the feedback consistency
      
      #self.fconsistency should be run between 0 and 20 (a total of 21 different runs)
      if self.fconsistency >=0 or self.fconsistency <= 20 :
          CONSISTENCY = 1.0 - self.fconsistency * 0.05;
      else :
          print 'the test should be 1 through 6.'
          quit()
          
      if self.ostateindex[state] != action :
          if random.random() < CONSISTENCY :
              return False;
          else :
              return True;
      else :
          if random.random() < CONSISTENCY :
              return True;
          else :
              return False;
      
      #error called when the oracle doesn't know the state.
      print 'Something is wrong with the oracle. The Oracle Qsa pair wasn\'t found.'
      print 'The state is: \n' + str + '\nThe action is: ' + repr(action)
      quit()
  
  def UpdateOraclePolicy(self, state, action, bAGREE):
  #Update the oracle policy using feedback.
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      
      for i in range(0, len(statedata)) :
          if statedata[i][0] == action:
              if bAGREE:
                  statedata[i][2][0] = statedata[i][2][0] + 1;
              else :
                  statedata[i][2][0] = statedata[i][2][0] - 1;
          else : #for the "only one optimal action" case (and any other hallucination)
               if bAGREE:
                   statedata[i][2][1] = statedata[i][2][1] + 1;
               else :
                   statedata[i][2][1] = statedata[i][2][1] - 1;

      for i in range(0, len(statedata)) :
          if statedata[i][0] == action :
              statedata[i][3] = statedata[i][3] + 1;
              return;

  def ProcessOracleFeedback(self, state, action):
  #Calculates whether the oracle should provide feedback in the current state.
  #LIKELIHOOD represents the likelihood of receiving feedback in this function.
    
    #thing should be between 0 and 20
    if self.flikelihood >= 0 or self.flikelihood <= 20 : #21 tests per 
        LIKELIHOOD = 1.0 - self.flikelihood*0.05;
    else :
        print 'the test should be 0 through 20.'
        quit()
    
    #give feedback with a probability
    if random.random() < LIKELIHOOD :
        bAGREE = self.GenerateOracleFeedback(state, action);
        self.UpdateOraclePolicy(state, action, bAGREE);

  def LoadOracleQVals(self) :
  #function to load the optimal state/action pairs from the csv file.
    print 'Loading oracle q-values.' 
    
    f = open('oracle/oracle.txt', 'r');
    
    count = 0
    state = ''
    for line in f :
        if count % 4 == 3 : 
            entries = line.rsplit(';'); 
            state += entries[0]; 
            action = int(entries[1].rsplit()[0]);
            self.ostateindex[state] = action;
            state = ''; 
        else : 
            state += line + '';
        
        count = count + 1;


#This is the implementation of Action Biasing from Knox and Stone.
class BayesianRLActionBiasing(BayesianRLHumanFeedback):
  def __init__(self, gameType, save=False):
  #Action Biasing requires extra parameters.
      self.absMAXreward = 100;
      self.Beta = 1;                #the human influence parameter
      self.bdecay = 0.999;          #the decay rate of Beta
      self.lastep = 0;
      BayesianRLHumanFeedback.__init__(self, gameType, save=False);

  def PrintParameterInitialization(self):
  #Includes the new parameters in the Parameter Initialization debugging code.
    BayesianRLHumanFeedback.PrintParameterInitialization(self);
    print '  human feedback --> reward: ' + repr(self.absMAXreward);
    print '  beta: ' + repr(self.Beta);
    print '  beta decay: ' + repr(self.bdecay);

  def update(self, state, action, nextState, reward):
  #Decay beta for each new episode.
      BayesianRLHumanFeedback.update(self, state, action, nextState, reward);
      if self.episodesSoFar != self.lastep :
          self.Beta = self.Beta*self.bdecay;
          self.lastep = self.episodesSoFar;

  def GetActionBiasFromHumanFeedback(self, state):
  #Based on the accumulated feedback in each state, determine the reward value to use to bias action selection.
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      
      abias = [];
      #calculate the bias for each action using the reward mapping, human influence, and accumulated feedback
      for i in range(0, len(statedata)) :
          afeedback = statedata[i][2][0];
          if afeedback > 0 :
              abias.append(self.Beta*self.absMAXreward);
          elif afeedback < 0 :
              abias.append(self.Beta*-1.0*self.absMAXreward);
          else :
              abias.append(0);
      
      return abias;

  def ChooseActionsUsingVPI(self, state):
  #Estimate the optimal actions per state using myopic VPI with action biasing.
  
    sidx = self.stateindex[state];
    statedata = self.knownstates[sidx];
    
    #the values used to bias the Q-value
    abias = self.GetActionBiasFromHumanFeedback(state);
    
    #bias the actions for the myopic VPI computation
    for i in range(0, len(abias)) :
        statedata[i][1][0] = statedata[i][1][0] + abias[i];
    
    #compute the myopic VPI
    myopicvpi = self.VPI(state);
    maxi = 0
    max = myopicvpi[0][0];
    for i in range(0, len(myopicvpi)):
        if myopicvpi[i][0] > max :
            max = myopicvpi[i][0];
            maxi = i;
    
    action = myopicvpi[maxi][1];
    
    #unbias the actions
    for i in range(0, len(abias)) :
        statedata[i][1][0] = statedata[i][1][0] - abias[i];
    
    return action;

  def ActionSelection(self, state):
      return self.ChooseActionsUsingVPI(state);

#This is the implementation of Control Sharing as described in Knox and Stone. 2010.
class BayesianRLControlSharing(BayesianRLHumanFeedback):
  def __init__(self, gameType, save=False):
  #Control Sharing requires extra parameters.
      self.absMAXreward = 10;
      self.Beta = 1;
      self.bdecay = 0.99;
      self.lastep = 0;
      BayesianRLHumanFeedback.__init__(self, gameType, save=False);

  def PrintParameterInitialization(self):
  #Includes the new parameters in the Parameter Initialization debugging code.
    BayesianRLHumanFeedback.PrintParameterInitialization(self);
    print '  human feedback --> reward: ' + repr(self.absMAXreward);
    print '  beta: ' + repr(self.Beta);
    print '  beta decay: ' + repr(self.bdecay);

  def update(self, state, action, nextState, reward):
  #Decay beta for each new episode.
      BayesianRLHumanFeedback.update(self, state, action, nextState, reward);
      if self.episodesSoFar != self.lastep :
          self.Beta = self.Beta*self.bdecay;
          self.lastep = self.episodesSoFar;
          
  def GetBestActionAccordingToHumanFeedback(self, state):
  #Determine the human feedback policy from the accumulated feedback
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      
      maxa = -1000000000;
      maxaction = statedata[0][0];
      
      #Find the action with the most positive feedback.
      for i in range(0, len(statedata)) :
          afeedback = statedata[i][2][0];
          
          if afeedback > maxa :
              maxa = afeedback;
              maxaction = statedata[i][0];
      
      return maxaction;
  
  def ActionSelection(self, state):
    sidx = self.stateindex[state];
    statedata = self.knownstates[sidx];
    
    hitcount = 0;
    for i in range(0, len(statedata)) :
        if statedata[i][3] > 0 :
            hitcount = 1;
            break;
    
    #The transition mechanism of Control Sharing. Use the value of the human influence parameter to decide when to transition to the RL policy.
    if hitcount > 0 :
        if random.random() < self.Beta :
            return self.GetBestActionAccordingToHumanFeedback(state);

    return BayesianRLHumanFeedback.ActionSelection(self, state);
  
#This is the implementation of Advise with feedback propagation. 
class BayesianRLAdvisePropagate(BayesianRLHumanFeedback):
  def AddStateToList(self, state, actions):
  #The state is redefined to include the feedback backup information
      if not self.stateindex.has_key(state) :
          sidx = len(self.knownstates);
          self.stateindex[state] = sidx;
          stateinfo = [];

          for action in actions :
              #[action, hyperparameters, feedback counts, feedback rcvd, conf. next state, q conf. this state] for each state
              stateinfo.append([action, self.priorhyperparameters, [0,0], 0, 1.0/len(actions), 1.0/len(actions)]); 
          
          self.knownstates.append(stateinfo);

  def GetThisQDistr(self, state):
  #Get the BQL policy for the current state.
      Q_distr = [];
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      for i in range(0, len(statedata)) :
          Q_distr.append(statedata[i][5]);
      return Q_distr;
  
  def PropagateAdvise(self, state, action, nextState):
  #Identifies the agent's confidence in the human feedback policy in the next state. This is the propagation value for the current state.
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      
      if self.stateindex.has_key(nextState) :
          #Get the feedback policy in the next state.
          O_distr = self.GetOraclePolicy(nextState);
          
          maxpa = 0.0;
          usedQ = 0;
          #Find the max value.
          for i in range(0, len(O_distr)) :
              if O_distr[i] > maxpa :
                  maxpa = O_distr[i];
                  usedQ = 0;
                  
          #Save it as the backup value.
          for i in range(0, len(statedata)) :
              if statedata[i][0] == action :
                  statedata[i][4] = maxpa;
                  break;

  def update(self, str, action, nextState, reward):
  #Propagate feedback for each update. Propagation is analogous to Q Value backup.
      BayesianRLHumanFeedback.update(self, str, action, nextState, reward);
      self.PropagateAdvise(str, action, nextState);
      self.UpdateQLearningPolicy(str); #update the Q_distribution when the NG distribution changes.
  
  def UpdateQLearningPolicy(self, state):
  #Maintain the Bayesian Q-Learning policy and save to the state.
      Q_distr = BayesianRLHumanFeedback.EstimateTheProbabilityWithVPI(self, state);
      
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      for i in range(0, len(statedata)) :
          statedata[i][5] = Q_distr[i];

  def GetOraclePolicy(self, state):
  #Calculates the Oracle Policy alone.
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      
      #get the accumulated feedback for each action
      countsdir = [];
      for i in range(0,len(statedata)):
          countsdir.append(statedata[i][2]);
      
      if len(countsdir) == 0:
          return [];
      
      #use the feedback consistency to compute the oracle policy
      if self.fconsistency >=0 or self.fconsistency <= 20 :
          CONSISTENCY = 1.0 - self.fconsistency * 0.05;
      else :
          print 'the test should be 0 through 20.'
          quit()
      
      C = CONSISTENCY;
      if C == 1.0 :
          C = 0.99999999999999;
      elif C == 0.0 :
          C = 0.00000000000001;
      
      O_distr = [];
      sum = 0.0;

      #compute the oracle policy.
      for i in range(0, len(countsdir)) :
          N = countsdir[i];
          
          try :
              val = C**N[0] * (1.0-C)**N[1] * statedata[i][4];
          
          except :
              if N[0] > N[1] :
                  val = statedata[i][4];
              else :
                  val = 0;
          
          O_distr.append(val)
          sum = sum + val;
      
      #avoid divide by zero errors.
      try :
          #normalize the distribution
          for i in range(0,len(countsdir)) :
              O_distr[i] = O_distr[i]/sum;
      except :
          return [];

      return O_distr;

  def GetCombinedPolicy(self, state):
  #Get the policy for the state. 
      sidx = self.stateindex[state];
      statedata = self.knownstates[sidx];
      
      #get the accumulated feedback for each action
      countsdir = [];
      for i in range(0,len(statedata)):
          countsdir.append(statedata[i][2]);
      
      if len(countsdir) == 0:
          return [];
      
      #use the feedback consistency to compute the oracle policy
      if self.fconsistency >= 0 or self.fconsistency <= 20 :
          CONSISTENCY = 1.0 - self.fconsistency * 0.05;
      else :
          print 'the test should be 0 through 20.'
          quit()
      
      C = CONSISTENCY;
      if C == 1.0 :
          C = 0.99999999999999;
      elif C == 0.0 :
          C = 0.00000000000001;
      
      O_distr = [];
      Q_distr = self.GetThisQDistr(state);
      
      sum = 0.0;
      max = 0.0;
      #calculate a distribution over the actions
      for i in range(0, len(countsdir)) :
          N = countsdir[i];
          
          try :
              val = C**N[0] * (1.0-C)**N[1] * statedata[i][4];
          
          except :
              if N[0] > N[1] :
                  val = statedata[i][4];
              else :
                  val = 0;
          
          if val > max :
              max = val;
          
          O_distr.append(val)

      #combine the BQL policy and the Oracle Policy
      OM_distr = [];
      for i in range(0, len(O_distr)) :
          if O_distr[i] == max :
              OM_distr.append(Q_distr[i]);
              sum = sum + Q_distr[i];
          else :
              OM_distr.append(0.0);
      
      #avoid divide by zero errors
      try :
          for i in range(0,len(countsdir)) :
              OM_distr[i] = OM_distr[i]/sum;
      except :
          return Q_distr;
      
      return OM_distr;

  def ChooseAction(self, prob_distr):
  #Sample an action according to the probability distribution.
      chosena = random.random();
      sum = 0;
      count = 0
      for actioni in range(0, len(prob_distr)) :
          sum = sum + prob_distr[actioni];
          if chosena < sum :
              break;
          
          count = count + 1
      
      #an edge case: when the probability distribution adds up to slightly less than 1.0, but the prob distr chose something near 1.0
      if count >= len(prob_distr) :
          count = len(prob_distr) - 1;
      
      return count;

  def ActionSelection(self, state):
  #Choose an action by sampling from the policy.
    sidx = self.stateindex[state];
    statedata = self.knownstates[sidx];
    
    distr = self.GetCombinedPolicy(state);

    #sample an action from the policy
    aidx = self.ChooseAction(distr);
    return statedata[aidx][0];

