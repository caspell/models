from PIL import Image, ImageFont, ImageDraw

import numpy as np
import matplotlib.pyplot as plt
import sys , os
from utils import mnist_cnn_train

SAVE_PATH = '/home/mhkim/data/fonts'

class FontUtil :
    error_font_file = "fonts_error.txt"
    def __init__(self, text="0123456789"):
        self._text = text

        #print ( fonts)
    def load(self, filePath):
        if os.path.exists(self.error_font_file):
            os.remove(self.error_font_file)

        with open(filePath, "r") as f :
            self._fonts = [ line.replace('\n', '') for line in f.readlines() ]

    def getFullImages (self) :

        if not self._fonts :
            print ('muse load font list file !')

        list = []
        for font in self._fonts:
            try :
                list.append(self.getToImage(font, self._text))
            except :
                print ("error")
                with open(self.error_font_file, "a") as error_list:
                    error_list.write(font + '\n')
        return list

    def getToImage (self, fontPath, strs) :

        returnImage = []
        returnChars = []

        font = ImageFont.truetype(fontPath, 35)

        fontFileName = os.path.basename(fontPath)

        print ( 'font create : {}'.format(fontFileName) )

        for _x in range(10):
            for _y in range(10) :
                for _char in strs :

                    dir = '{}/{}'.format(SAVE_PATH, _char)
                    if not os.path.exists(dir) :
                        os.makedirs(dir)

                    im = Image.new("RGB", (56, 56))

                    draw = ImageDraw.Draw(im)

                    draw.text((5 + _x, 5 + _y), _char, font=font)

                    # im = im.convert('RGB')

                    im.save(os.path.join(dir, '{}_{}_{}.jpg'.format(fontFileName, _x, _y)))

                    im = np.array(im.getdata()).reshape(im.size[0], im.size[1], 3)

                    returnImage.append(im)
                    returnChars.append(_char)

        return (returnImage , returnChars)

    '''
    for font in fonts :
        print(font)
        try :
            fonta = ImageFont.truetype(font, 30)
            #print(font.size)
        except :
            print ('error ' , font)
    '''

def test() :
    fontUtil = FontUtil()

    test = '/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf'

    # test = img.imread(test)

    # plt.imshow(test)
    # plt.show()

    result = fontUtil.getToImage(test, 'abvc가ㅏ쀍')

    # print(result)

    for r in result[0]:
        print(np.shape(r))
        # print ( r )

        plt.imshow(np.squeeze(r, axis=2))
        plt.show()

def main() :

    fontUtil = FontUtil('0123456789')

    fontUtil.load('fonts_number.txt')

    fontUtil.getFullImages()

    fontUtil = FontUtil('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    fontUtil.load('fonts_alpha.txt')

    fontUtil.getFullImages()

    fontUtil = FontUtil('가나다라마바사아자차카타파하')

    fontUtil.load('fonts_korean.txt')

    fontUtil.getFullImages()


    #------------------------------------------
    # fontUtil.load('fonts.txt')
    # results = fontUtil.getFullImages()
    #print(result)

    # mnistCnn = mnist_cnn_train.MnistCnn()
    #
    # print ( np.shape(results) )
    #
    # total_count = 0
    # notmatch_count = 0
    #
    # for result in results :
    #     print ('step %d' % (total_count / 10))
    #
    #     labels = result[1]
    #     datas = result[0]
    #
    #     for i in range(len(result[0])):
    #         resultValue = mnistCnn.execute([datas[i]])
    #         total_count += 1
    #         if int(labels[i]) != int(resultValue) :
    #             notmatch_count += 1
    #
    # print('%.2f' % (total_count / notmatch_count), '%')


    #--------------
    # for r in result[0] :
    #     #print ( np.shape([r]) )
    #     resultValue = mnistCnn.execute([r])
    #     print ( resultValue )

    # for c in result:
    #     plt.imshow(c[0])
    #     plt.show()

def files () :
    fontFileList = 'fonts.txt'

    filters = (
        ('number',  '/home/mhkim/data/fontlist/number.txt')
        , ('alpha' , '/home/mhkim/data/fontlist/alpha.txt')
        , ('korean' , '/home/mhkim/data/fontlist/korean.txt')
    )

    target = {}
    target2 = {}

    get_fn = (lambda mk : 'fonts_{}.txt'.format(mk))

    for k, v in filters :
        if os.path.exists(get_fn(k)) :
            os.remove(get_fn(k))

        with open(v, 'r') as f :
            target[k] = f.readlines()

    for m in target :
        target2 [m] = {}
        for val in target[m]:
            fn = val.strip('\n')
            fn = os.path.basename(fn)
            fn = os.path.splitext(fn)[0]
            target2[m][fn] = True

    with open(fontFileList, 'r') as f :
        list = f.readlines()

    for fn in list :
        bname = os.path.basename(fn).strip('\n')
        for mk in target2 :
            with open(get_fn(mk), 'a') as f :
                if bname in target2[mk] :
                    f.write(fn)


if __name__ == '__main__' :
    main()
    # test()
    # files()