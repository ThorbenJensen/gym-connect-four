
# gym-connect-four  
Open AI Gym for ConnectFour game  
  
### Setup  
  
Install the environment:  
  
``` bash  
git clone https://github.com/IASIAI/gym-connect-four.git  
cd gym-connect-four  
pip install -e .  
```  
  
Test two random players :  
  
``` bash  
python example/sample.opponent.py  
```  
  
### Usage  
  
Inside the environment there are a couple of sample players provided:  
* **RandomPlayer**: as name suggests it only does random moves, although valid ones  
* **SavedPlayer**: loads a saved model and uses it to play  
  
Inside the repo there are a couple of examples:  
* **sample_nn**: Neural Network implementation identical to the one from CartPole playing against a random opponent  
* **sample_opponent**: Simple random vs random  
* **sample_two_players**: Using two defined players, without relying on the opponent feature in the environment  
  
Considerations for the environment:  
* the ```reset``` function is the one that assigns the opponent and must be specified each time, unless both players are handled by user (as in sample_two_players)  
  
### Assignment  
  
For the competition the following set of deliverables must be provided:  
* python file containing a Player class used for training the model (NNPlayer + DQNSolver if relative to sample_nn.py)  
* python class containing a SavedPlayer implementation (only if any customization was needed)  
* a .h5 (HDF5) file containing the model structure, weights and optimizer (will be the provided or default SavedPlayer class) (https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)  
  
### Evaluation of the assignment  
  
All supplied models will pitted against each other in a round robbin tournament with the winner being the one that has won the most amount of matches.  
  
### Have fun and happy training!