from PIL import Image
import struct

#图片：
def read_image(filename):
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()

    magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')

    for i in range(images):
    #for i in range(2000):
        image = Image.new('L', (columns, rows))

        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')
                #print ('save' + str(i) + 'image')
                
                image1 = image.transpose(Image.FLIP_LEFT_RIGHT)
                image2 = image1.transpose(Image.FLIP_LEFT_RIGHT)

                image2.save('train/' + str(i) + '.png')

# 标签：
def read_label(filename, saveFilename):
  f = open(filename, 'rb')
  index = 0
  buf = f.read()

  f.close()

  magic, labels = struct.unpack_from('>II' , buf , index)
  index += struct.calcsize('>II')
  
  labelArr = [0] * labels
  #labelArr = [0] * 2000


  for x in range(labels):
  #for x in range(2000):
    labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
    index += struct.calcsize('>B')

    save = open(saveFilename, 'w')

    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')

    save.close()
    #print ('save labels success')


if __name__ == '__main__':
  imagePath = './train-images-idx3-ubyte'
  labelPath = './train-labels-idx1-ubyte'
  labelSavTransPath = 'train/label.txt'

  #读取数据集：
  read_image(imagePath)
  #读取标签，并解析为txt文档：
  read_label(labelPath, labelSavTransPath)
