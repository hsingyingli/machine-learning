import argparse 
from agent import *


def main(args):
    agent = Agent(args)
    # agent.env = gym.wrappers.Monitor(agent.env, directory="mon", force=True)
    print('Game Start')
    agent.solve()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env'        , type = str   , default = "CartPole-v0")
    parser.add_argument('--lr'         , type = float   , default = 1e-2)
    parser.add_argument('--batch_size' , type = int   , default = 16)
    parser.add_argument('--hidden_size', type = int   , default = 128)
    parser.add_argument('--percentile' , type = int   , default = 70)
    
    args = parser.parse_args()
    print(args)
    main(args)