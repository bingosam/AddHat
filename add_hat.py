# -*- encoding=utf8 -*-
__author__ = "Zhang kunbin"

import os

import cv2
import dlib
import numpy as np

from geometry import *


def rotate(_img, _degrees, scale=1):
    """
    旋转图片
    :param _img: 图片
    :param _degrees:  角度
    :param scale: 缩放比例
    :return: 结果图
    """
    height, width = _img.shape[:2]
    (cX, cY) = (width // 2, height // 2)
    rotation = cv2.getRotationMatrix2D((cX, cY), _degrees, scale)

    cos = np.abs(rotation[0, 0])
    sin = np.abs(rotation[0, 1])

    new_height = int(width * sin + height * cos)
    new_width = int(height * sin + width * cos)

    # adjust the rotation matrix to take into account translation
    rotation[0, 2] += (new_width / 2) - cX
    rotation[1, 2] += (new_height / 2) - cY
    return cv2.warpAffine(_img, rotation, (new_width, new_height))


def get_foreground_roi(foreground_img, background_img, background_rect):
    """
    获取前景图需要的区域
    :param foreground_img: 前景图
    :param background_img: 背景图
    :param background_rect: 背景区域的left top和right bottom 2点坐标
    :return:
    """
    foreground_h, foreground_w = foreground_img.shape[:2]
    background_h, background_w = background_img.shape[:2]

    # 左上角坐标
    left_top = (
        0 if background_rect[0][0] >= 0 else (-background_rect[0][0]),
        0 if background_rect[0][1] >= 0 else (-background_rect[0][1])
    )

    # 右下角
    right_bottom = (
        foreground_w if background_rect[1][0] <= background_w else (
                foreground_w - (background_rect[1][0] - background_w)),
        foreground_h if background_rect[1][1] <= background_h else (
                foreground_h - (background_rect[1][1] - background_h))
    )

    # 获取分割后的帽子ROI
    return foreground_img[
           left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]
           ]


def get_hat_left_bottom(raw_size, rotated_size, degrees, point_hat_center_bottom):
    """
    获取帽子左下点坐标
    :param raw_size: 帽子大小
    :param rotated_size: 旋转后的大小
    :param degrees: 旋转角度
    :param point_hat_center_bottom: 原帽子中下点坐标
    :return: 帽子左下点坐标
    """
    relative_y = int(math.sin(math.radians(abs(degrees))) * (raw_size[0] / 2))
    if degrees < 0:
        relative_x = int(
            rotated_size[0]
            - abs(math.cos(math.radians(-degrees)) * (raw_size[0] / 2))
            - abs(math.sin(math.radians(-degrees)) * raw_size[1])
        )

    else:
        relative_x = int(
            abs(math.cos(math.radians(degrees)) * (raw_size[0] / 2))
            + abs(math.sin(math.radians(degrees)) * raw_size[1])
        )
    return point_hat_center_bottom[0] - relative_x, point_hat_center_bottom[1] + relative_y


class Hatter:
    def __init__(self, hat_img_path, mini_head_degrees=0, debug=False):
        """
        构造舒适化
        :param hat_img_path: 帽子图片路径
        :param predictor_path: 特征模型
        :param mini_head_degrees: 头部最小倾斜度
        :param debug: 是否调试模式
        """
        # dlib人脸关键点检测器
        self.predictor_path = os.path.join(os.path.dirname(__file__),
                                           "data/weights/shape_predictor_5_face_landmarks.dat")
        self.predictor = dlib.shape_predictor(self.predictor_path)

        # dlib正脸检测器
        self.detector = dlib.get_frontal_face_detector()

        self.hat_img, self.raw_hat_height, self.raw_hat_width = None, None, None
        # 帽子伸缩尺寸
        self.hat_scale = 1
        self.hat_center_bottom_provider = calc_equilateral_triangle_point
        self.set_hat_img(hat_img_path)
        self.debug = debug
        self.mini_head_degrees = mini_head_degrees

    def set_predictor(self, predictor_path):
        """
        设置检测器
        :param predictor_path: 特征模型
        """
        self.predictor_path = predictor_path
        self.predictor = dlib.shape_predictor(self.predictor_path)

    def set_hat_img(self, hat_img_path):
        """
        设置帽子图片
        :param hat_img_path: 帽子图片路径
        """
        # 读取帽子图，必须读取RGBA通道，A通道用来mask
        self.hat_img = cv2.imread(hat_img_path, -1)
        self.raw_hat_height, self.raw_hat_width = self.hat_img.shape[:2]
        self.hat_scale = 2 ** int(self.raw_hat_width / self.raw_hat_height)

    def enable_debug(self, enabled):
        self.debug = enabled

    def _get_hat_center_bottom(self, point_right_eye_right,
                               point_left_eye_left,
                               degrees):
        """
        获取帽子中下点坐标
        :param point_right_eye_right: 右眼眼角坐标
        :param point_left_eye_left: 左眼眼角坐标
        :param degrees: 倾斜角度
        :return: 帽子中下点坐标
        """
        if self.raw_hat_height > self.raw_hat_width:
            # 高帽，取等边三角形的顶点作为帽子中下点坐标
            point_hat_bottom = calc_equilateral_triangle_point(point_right_eye_right,
                                                               point_left_eye_left, degrees)
        else:
            # 高帽，取等腰直角三角形的B点作为帽子中下点坐标
            eyes_center = calc_center_point(point_left_eye_left, point_right_eye_right)
            point_hat_bottom = calc_isosceles_right_triangle_point_b(point_right_eye_right, eyes_center)
        return int(point_hat_bottom[0]), int(point_hat_bottom[1])

    def add_hat(self, background):
        """
        添加帽子
        :param background: 背景图
        :param hat_img: 帽子图片
        :return:
        """
        # 正脸检测
        dets = self.detector(background, 1)

        background_height, background_width = background.shape[:2]
        # 如果检测到人脸
        for d in dets:
            x, y, w, h = d.left(), d.top(), d.right() - d.left(), d.bottom() - d.top()

            if self.debug:
                cv2.rectangle(background, (x, y), (x + w, y + h), (255, 0, 0), 2, 8, 0)

            # 关键点检测，5个关键点
            shape = self.predictor(background, d)

            if self.debug:
                for p in shape.parts():
                    cv2.circle(background, (p.x, p.y), 3, color=(0, 255, 0))

            # 选取左右眼眼角的点
            point_left_eye_left = shape.part(0)
            point_left_eye_left = (point_left_eye_left.x, point_left_eye_left.y)
            point_right_eye_right = shape.part(2)
            point_right_eye_right = (point_right_eye_right.x, point_right_eye_right.y)

            # 两眼距离
            distance_eyes = calc_distance(point_left_eye_left, point_right_eye_right)

            # 头部倾斜角度
            degrees = calc_degrees(point_right_eye_right, point_left_eye_left)
            if abs(degrees) < self.mini_head_degrees:
                degrees = 0

            # 调整帽子大小
            new_w_no_rotate = distance_eyes * self.hat_scale
            new_h_no_rotate = self.raw_hat_height / self.raw_hat_width * new_w_no_rotate

            # 旋转帽子后，帽子图片的宽度会再次发生变化
            img_resized_hat = rotate(self.hat_img, degrees, new_w_no_rotate / self.raw_hat_width)
            if self.debug:
                cv2.imwrite('img_resized_hat.jpg', img_resized_hat)
            resized_hat_h, resized_hat_w = img_resized_hat.shape[:2]

            # 计算帽子底部中心点
            point_hat_center_bottom = self._get_hat_center_bottom(point_right_eye_right,
                                                                  point_left_eye_left,
                                                                  degrees)
            point_hat_left_bottom_on_background = \
                get_hat_left_bottom((new_w_no_rotate, new_h_no_rotate), (resized_hat_w, resized_hat_h), degrees,
                                    point_hat_center_bottom)

            # 计算帽子在背景图的位置
            rect_hat_on_background = (
                # left_top
                (point_hat_left_bottom_on_background[0], point_hat_left_bottom_on_background[1] - resized_hat_h),
                # right_bottom
                (point_hat_left_bottom_on_background[0] + resized_hat_w, point_hat_left_bottom_on_background[1])
            )

            # 获取分割后的帽子ROI
            roi_clipped_hat = get_foreground_roi(img_resized_hat, background, rect_hat_on_background)

            if self.debug:
                cv2.circle(background, point_hat_left_bottom_on_background, 3, color=(255, 0, 0))
                cv2.circle(background, rect_hat_on_background[1], 3, color=(255, 0, 0))
                # 三角形
                points = np.array([point_right_eye_right, point_left_eye_left, point_hat_center_bottom])
                cv2.polylines(background, [points], True, color=(0, 0, 255), thickness=3)
                cv2.line(background, point_right_eye_right, (point_left_eye_left[0], point_right_eye_right[1]),
                         color=(0, 0, 255), thickness=3)

            r, g, b, a = cv2.split(roi_clipped_hat)
            img_rbg_resized_hat = cv2.merge((r, g, b))

            # 用alpha通道作为mask
            mask = a
            mask_inv = cv2.bitwise_not(mask)

            # 原图ROI的帽子区域
            rect_clipped_hat_on_background = [
                # left top
                (rect_hat_on_background[0][0] if rect_hat_on_background[0][0] > 0 else 0,
                 rect_hat_on_background[0][1] if rect_hat_on_background[0][1] > 0 else 0
                 ),
                # right bottom
                (rect_hat_on_background[1][0] if rect_hat_on_background[1][0] < background_width else background_width,
                 rect_hat_on_background[1][1] if rect_hat_on_background[1][1] < background_height else background_height
                 )
            ]

            # 原图ROI中提取放帽子的区域
            roi_background = background[rect_clipped_hat_on_background[0][1]:rect_clipped_hat_on_background[1][1],
                             rect_clipped_hat_on_background[0][0]:rect_clipped_hat_on_background[1][0]]

            roi_background = roi_background.astype(float)
            mask_inv = cv2.merge((mask_inv, mask_inv, mask_inv))
            alpha = mask_inv.astype(float) / 255

            bg = cv2.multiply(alpha, roi_background)
            bg = bg.astype('uint8')

            # 提取帽子区域
            hat = cv2.bitwise_and(img_rbg_resized_hat, img_rbg_resized_hat, mask=mask)

            # 两个ROI区域相加
            img_hat_added = cv2.add(bg, hat)

            # 把添加好帽子的区域放回原图
            background[rect_clipped_hat_on_background[0][1]:rect_clipped_hat_on_background[1][1],
            rect_clipped_hat_on_background[0][0]:rect_clipped_hat_on_background[1][0]] = img_hat_added

        return background, len(dets)
