import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob
import argparse 
from PIL import Image


class DataGenerator():
    def __init__(self, args):

        self.path  = args.path         
        self.img_w = args.img_w        
        self.img_h = args.img_h 

        self.label_code, self.label_name = self.get_label_code()
        self.num_channel = len(self.label_code)
        self.code2id = {color: i for i, color in enumerate(self.label_code)}
        self.id2code = {i: color for i, color in enumerate(self.label_code)}
        
        self.test_mean  = None
        self.test_std   = None

    def open_image(self, img):
        x = np.asarray(Image.open(img).resize((self.img_h, self.img_w), Image.NEAREST))
        x = x.astype(np.float32).transpose(2,0,1)     # (h, w, ch)  to (ch, h, w)
        x = np.ascontiguousarray(x)[::-1]             # (R, G, B)   to (B, G, R)
        return x                                                                  

    def get_data(self): 
        images  = sorted(glob.glob(self.path +"train/"+ "images/"+ '*.png'))
        labels  = sorted(glob.glob(self.path +"train/"+ "masks/" + "*.png"))
        
        train_imgs   = np.array([self.open_image(i) for i in images])
        train_labels = np.array([self.open_image(i) for i in labels])

        
        images  = sorted(glob.glob(self.path +"test/"+ "images/"+ '*.png'))
        labels  = sorted(glob.glob(self.path +"test/"+ "masks/" + "*.png"))
        
        test_imgs   = np.array([self.open_image(i) for i in images])
        test_labels = np.array([self.open_image(i) for i in labels])
        
        
        '''
        z-score
        '''
        train_imgs /= 255.
        train_imgs =( train_imgs - train_imgs.mean()) / train_imgs.std()

        test_imgs /= 255.
        self.test_mean  = test_imgs.mean()
        self.test_std   = test_imgs.std()
        test_imgs =( test_imgs - test_imgs.mean()) / test_imgs.std()

        

        return  train_imgs, self.conv_all_label(train_labels), test_imgs, self.conv_all_label(test_labels)



        
    def get_label_code(self):
        path = "./../data/class_dict.csv"
        df = pd.read_csv(path)
        # print(df)
        df = df[["name", "b", "g", "r"]]
        # print(df)
        label_code = [ tuple(i) for i in df.iloc[:,1:].values.tolist()]
        label_name = df.name.values.tolist()
        
        return label_code, label_name


    def conv_one_label(self, img):   
            
        res = np.zeros((self.num_channel, img.shape[1], img.shape[2]), 'float32')
    
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                color=tuple(img[:,j,k].astype("int64").tolist())
                if color in self.code2id:
                    res[self.code2id[color], j, k] = 1
                else:
                    res[0, j, k] = 1
        return res  


    def conv_all_label(self, img):
        res = []
        for i in img:
            res.append(self.conv_one_label(i))
        return np.asarray(res)

    def array2img(self, arr):
        
        arr = arr[::-1].transpose(1,2,0).astype("uint8")
        arr = Image.fromarray(arr.astype("uint8"))
        return arr


    def decode_labels(self, masks):     # (batch, 32, h, w) to (batch, 3, h, w)
        result = []
        for batch in range(masks.shape[0]):
            decode_map = masks[batch]
            decode_map = np.argmax(decode_map, 0).astype(np.float32)
            color_img = np.zeros((self.img_h, self.img_w, 3), dtype=np.float32)# * 192
            for k in range(self.num_channel):
                    color_img[decode_map == k] = np.asarray(self.label_code)[k]
            result.append(color_img.transpose(2, 0, 1))
        
        return np.asarray(result)

    def draw(self, origin, y_true, y_pred):
        
        origin, y_true, tmp = origin.numpy(), y_true.numpy(), y_pred.numpy()
        tmp = np.argmax(tmp, axis = 1)

        y_pred = np.zeros((y_true.shape[0], y_true.shape[1], y_true.shape[2], y_true.shape[3]))
        for batch in range(y_true.shape[0]):
            for row in range(y_true.shape[2]):
                for col in range(y_true.shape[3]):
                    index = tmp[batch, row, col]
                    y_pred[batch, index,row, col] = 1
        
        y_true, y_pred = self.decode_labels(y_true), self.decode_labels(y_pred)
        plt.cla()
        plt.clf()
        
        plt.subplot(331)
        plt.axis('off')
        plt.imshow(self.array2img((origin[0] * self.test_std + self.test_mean) * 255.))
        plt.subplot(332)
        plt.axis('off')
        plt.imshow(self.array2img(y_true[0]))
        plt.subplot(333)
        plt.axis('off')
        plt.imshow(self.array2img(y_pred[0]))

        plt.subplot(334)
        plt.axis('off')
        plt.imshow(self.array2img((origin[1] * self.test_std + self.test_mean) * 255.))
        plt.subplot(335)
        plt.axis('off')
        plt.imshow(self.array2img(y_true[1]))
        plt.subplot(336)
        plt.axis('off')
        plt.imshow(self.array2img(y_pred[1]))

        plt.subplot(337)
        plt.axis('off')
        plt.imshow(self.array2img((origin[2] * self.test_std + self.test_mean) * 255.))
        plt.subplot(338)
        plt.axis('off')
        plt.imshow(self.array2img(y_true[2]))
        plt.subplot(339)
        plt.axis('off')
        plt.imshow(self.array2img(y_pred[2]))
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path'           , type = str   , default = "./../data/")
    parser.add_argument('--img_w'          , type = int   , default = 64)
    parser.add_argument('--img_h'          , type = int   , default = 64)
    
    args = parser.parse_args()
    print(args)
    test = DataGenerator(args)
    test.get_data()
    