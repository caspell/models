from PIL import Image, ImageFont, ImageDraw

import numpy as np
import matplotlib.pyplot as plt
import sys , os
from utils import mnist_cnn_train

class FontUtil :
    error_font_file = "fonts_error.txt"
    def __init__(self, text="0123456789"):
        self._text = text

        #print ( fonts)
    def load(self, filePath):
        if os.path.exists(self.error_font_file):
            os.remove(self.error_font_file)

        with open("fonts.txt", "r") as f :
            self._fonts = [ line.replace('\n', '') for line in f.readlines() ]

    def getFullImages (self) :
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
        for _char in strs :

            im = Image.new("RGB", (28, 28))

            draw = ImageDraw.Draw(im)

            font = ImageFont.truetype(fontPath, 30)

            draw.text((0, -5), _char, font=font)

            im = im.convert('L')

            im = np.array(im.getdata()).reshape(im.size[0], im.size[1], 1)

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

    result = fontUtil.getToImage(test, 'abvc')

    # print(result)

    for r in result[0]:
        print(np.shape(r))
        # print ( r )

        plt.imshow(np.squeeze(r, axis=2))
        plt.show()

def main() :

    fontUtil = FontUtil()

    test = '/home/mhkim/tools/android-studio/plugins/android/lib/layoutlib/data/fonts/DroidSans.ttf'

    # test = img.imread(test)

    # plt.imshow(test)
    # plt.show()

    result = fontUtil.getToImage(test, '0123')

    # print(result)

    for r in result[0] :
        print ( np.shape(r) )
        # print ( r )

        plt.imshow(np.squeeze(r, axis=2))
        plt.show()


    #--------------
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

if __name__ == '__main__' :
    main()