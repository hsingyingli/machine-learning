import argparse 


from framework import *


def main(args):
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    environment = Framework(args)
    environment.show_model()
    print("||   Start to load data ......")
    environment.get_data()
    print("||   %4d Training data has been Found"%(len(environment.train_loader.dataset)))
    print("||   %4d Testing  data has been Found"%(len(environment.test_feature)))
    print("||   Start Training ......")
    environment.train()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path'           , type = str   , default = "./../data/")
    parser.add_argument('--img_w'          , type = int   , default = 108)
    parser.add_argument('--img_h'          , type = int   , default = 108)
    parser.add_argument('--device'         , type = str   , default = "cuda")
    parser.add_argument('--epoch'          , type = int   , default = 30)
    parser.add_argument('--batch_size'     , type = int   , default = 32)
    parser.add_argument('--lr'             , type = float , default = 4e-4)
    parser.add_argument('--blocks'         , type = int   , nargs = '+', default = [4, 5, 6, 7, 8])
    parser.add_argument('--growth_rate'    , type = int   , default = 16)
    parser.add_argument('--in_channels'    , type = int   , default = 3)
    parser.add_argument('--n_classes'      , type = int   , default = 32)
    parser.add_argument('--exp_id'         , type = int   , default = 3)
    parser.add_argument('--original'       , type = bool   , default = True)
    parser.add_argument('--seed'           , type = int   , default = 6666)
    
    args = parser.parse_args()
    print(args)
    main(args)