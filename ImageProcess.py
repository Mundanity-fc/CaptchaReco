from PIL import Image
from io import BytesIO


class ImageProcess(object):
    """
    图片处理类
    将Requests中获取到的二进制图片转换为Pillow可以读取的JPEG图片格式
    并对图像进行二值化处理
    可使用的成员函数
    byte2jpeg(byte)——将二进制格式图片byte转化为Pillow可读取格式
    image_binarize(image)——将Pillow格式图片image进行二值化
    process(source)——将request返回的二进制图片格式进行二值化
    """""

    def byte2jpeg(self, byte_content):
        """
        :param byte_content: Byte类型的图片数据
        :return: Pillow可以读取的jpeg图片格式
        """""
        return Image.open(BytesIO(byte_content))

    def image_binarize(self, image):
        """
        将验证码图像进行二值化处理
        :param image: 原始图像
        :return: 二值化处理后的图像
        """""

        # 将图片转化为灰度图像
        image = image.convert("L")
        # 按像素点将图像转换为矩阵
        image_pixel_matrix = image.load()
        # 获取图像的宽度与高度，并进行遍历
        # 图片的宽度与高度默认都为66和22
        width, height = image.size
        for h in range(height):
            for w in range(width):
                # 消除首尾行列的黑框
                if h == 0 or h == 21 or w == 0 or w == 61:
                    image_pixel_matrix[w, h] = 255
                else:
                    # 通过反复尝试，阈值为110-140时对于目前的验证码效果最好
                    # 105±5的范围内为最优情况，过小会丢失图像，过大会保留干扰图像
                    if image_pixel_matrix[w, h] < 103:
                        image_pixel_matrix[w, h] = 0
                    else:
                        image_pixel_matrix[w, h] = 255
        return image

    def process(self, source):
        """
        :param source: Requests返回的二进制格式图片
        :return: 二值化处理后的JPEG格式图片
        """""
        image = self.byte2jpeg(source)
        result = self.image_binarize(image)
        return result
