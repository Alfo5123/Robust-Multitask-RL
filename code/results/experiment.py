import sys, os 


# Run DQN
sys.path.append('../dqn')
from trainingDQN import trainDQN
#dqn_rewards , dqn_durations = trainingDQN.trainDQN()

##### # To fix conflicts 
#Run Soft Q-learning
sys.path.append('../sql')  
import trainingSQL  
sql_rewards , sql_durations = trainingSQL.trainSQL()

