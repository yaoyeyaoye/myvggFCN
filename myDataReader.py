# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc





class BatchDatset:

    ele_to_image = []
    images = []
    grids = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, inpath,inzone,image_options):
        """
        Intialize a generic file reader with batching for list of files

        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}

        :param image_options: A dictionary of options for modifying the output image
        sample record: {'image_size': IMAGE_SIZE}
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)

        self.image_options = image_options
        self._read_images(inpath,inzone)


    def get_records(self):
        return self.grids, self.images

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.grids.shape[0]: # 没有更多batch，打乱顺序重新开始
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.grids.shape[0])
            np.random.shuffle(perm)
            self.grids = self.grids[perm]
            self.images = self.images[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return  self.grids[start:end],self.images[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.grids.shape[0], size=[batch_size]).tolist()
        return self.grids[indexes], self.images[indexes]



    def min_max_elements(self,elements):
        '''
        将element的x，y，z坐标值放缩到与图片相符的大小
        '''
        # change the scale of numbers
        ele_scaler = preprocessing.MinMaxScaler()
        ele_scaler.feature_range = (0, self.image_options['image_size'] - 1)
        # print(ele_scaler.get_params())
        print(ele_scaler.fit(elements))
        ele_trans = ele_scaler.transform(elements)

        # round the numbers
        ele_around = np.around(ele_trans)
        return ele_around

    def map_elements_to_image(self,ele_around):
        '''将element的数据映射到image中,ele_to_image表示从element到图片的映射关系：
            ele_to_image[i][0] 表示element的编号[0,22189]，总共22190个
            ele_to_image[i][1] 表示image的x坐标[0,127]，总共128个
            ele_to_image[i][2] 表示image的y坐标[0,127]，总共128个
        '''
        for i in range(len(ele_around)):
            # print(ele_around[i])
            self.ele_to_image.append([i, ele_around[i][0], ele_around[i][1]])
        # print(ele_to_image)
        return

    def genarate_a_image(self,elements,m, h, b, a):
        '''利用ele_to_image中的映射关系，将element的属性值赋给到图片的pixel上
            image: 基于p，rho，热通量等生成的图片
            grid: 网格信息，共四层，依次分别为m,h,b,a
        '''
        image = np.zeros(shape=[self.image_options['image_size'], self.image_options['image_size']])
        grid = np.zeros(shape=[self.image_options['image_size'], self.image_options['image_size'],4])

        for i in range(len(self.ele_to_image)):
            e = int(self.ele_to_image[i][0])
            x = int(self.ele_to_image[i][1])
            y = int(self.ele_to_image[i][2])
            g = grid[x][y][0]
            if g == 0.:
                image[x][y] = elements[3][e]
                grid[x][y][0] += 1
            else:
                image[x][y] = (image[x][y] * g + elements[3][e]) / (g + 1)
                grid[x][y][0] += 1
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j][0] > 1:
                    grid[i][j][0] = m
                    grid[i][j][1] = h
                    grid[i][j][2] = b
                    grid[i][j][3] = a

        return image, grid

    def image_and_grib(self,filepath,m,h,b,a):
        '''
        生成类似自然图片的矩阵，包括5个channel，channel_1是P,channel_2-4是mhba
        '''
        ## 读入表格 elements = [[x,y,z,P]*m]
        elements = pd.read_csv(filepath, low_memory=False,
                               header=None, sep=',', encoding="ISO-8859-1")
        ## 得到element和image的映射关系表
        ele_around = self.min_max_elements(elements)
        if len(self.ele_to_image) == 0:
            self.map_elements_to_image(ele_around)

        ## 生成图片
        [image, grid] = self.genarate_a_image(elements,m,h,b,a)

        return np.array(image),np.array(grid)


    def _read_images(self,inpath,inzone):
        for root, dirs, files in os.walk(inpath):
            for f in files:

                par = f.split('_')
                [m, h, b, a] = par[:4]
                zone = par[6]

                if zone == inzone:
                    filepath = os.path.join(root, f)
                    print(filepath)
                    print(m, h, b, a, zone)

                    m = float(m.replace('m', ''))
                    h = float(h.replace('h', ''))
                    b = float(b.replace('b', ''))
                    a = float(a.replace('a', ''))

                    image,grid = self.image_and_grib(filepath, m, h, b, a)

                    self.images.append(image)
                    self.grids.append(grid)

        self.images = np.array(self.images)
        self.grids = np.array(self.grids)


        return






# if __name__=="__main__":
#     # ## 修改当前工作路径
#     # os.getcwd()
#     # os.chdir(path)
#     generate_image_files(ELEMENT_PATH,IMAGE_PATH)

