# -*- encoding=utf8 -*-
__author__ = "Zhang kunbin"

import math


def calc_degrees(point1, point2):
    """
    计算角度
    :param point1:
    :param point2:
    :return:
    """
    if point1[1] == point2[1]:
        return 0

    if point1[0] == point2[0]:
        return 90

    if point1[0] < point2[0]:
        left = point1
        right = point2
    else:
        left = point2
        right = point1

    result = math.degrees(math.atan(math.fabs(right[1] - left[1]) / math.fabs(right[0] - left[0])))
    return result if left[1] > right[1] else result * -1


def calc_a(degrees_a, hypotenuse):
    """
    计算直角三角形的A边，相对坐标长度
         ∠b
          *
        A *  *   C
          *    *
       ∠c ******** ∠a
            B
    :param degrees_a: A角
    :param hypotenuse: 斜边
    :return: A边长度
    """
    return math.sin(math.radians(degrees_a)) * hypotenuse


def calc_b(degrees_a, hypotenuse):
    """
    计算直角三角形的B边
         ∠b
          *
        A *  *   C
          *    *
       ∠c ******** ∠a
            B
    :param degrees_a: A角
    :param hypotenuse: 斜边
    :return: B边长度
    """
    return math.cos(math.radians(degrees_a)) * hypotenuse


def calc_center_point(point_a, point_b):
    """
    已知两点坐标，计算中间点坐标
    :param point_a:  A点坐标
    :param point_b: B点坐标
    :return:  中心点坐标
    """
    return (point_a[0] + point_b[0]) // 2, \
           (point_a[1] + point_b[1]) // 2


def calc_distance(point_a, point_b):
    """
    计算两点之间的长度
    :param point_a: A坐标
    :param point_b: B坐标
    :return:  两点之间的长度
    """
    return ((point_a[1] - point_b[1]) ** 2 + (point_a[0] - point_b[0]) ** 2) ** 0.5


def calc_equilateral_triangle_point(point_a, point_b, degrees_a=None, top=True):
    """
    计算等边三角形第三个点的坐标
    :param point_a: A点坐标
    :param point_b: B点坐标
    :param degrees_a: A与B的夹角
    :param top: 第三个点是在上面还是下面，默认上面
    :return: C点坐标
    """
    if not degrees_a:
        degrees_a = calc_degrees(point_a, point_b)
    length = calc_distance(point_a, point_b)

    radians = math.radians(degrees_a + 60)
    x_base = length * math.cos(radians)
    y_base = length * math.sin(radians)
    if top:
        return point_a[0] + x_base, point_a[1] - y_base
    else:
        return point_b[0] - x_base, point_b[1] + y_base


def calc_isosceles_right_triangle_point_b(point_a, point_c, top=True):
    """
    计算等腰直角三角形的B点
                               ∠a
                                *
         right_angle_length     *  *
                                *     *
                                ********* ∠b
                             ∠c
    :param point_a: A点坐标
    :param point_c: C点坐标
    :param top: B点坐标是否在上面
    :return: B点坐标
    """
    right_angle_length = calc_distance(point_a, point_c)
    degrees = calc_degrees(point_a, point_c)
    hypotenuse = (right_angle_length ** 2 * 2) ** 0.5
    # A与C点的相对于水平线的夹角
    degrees_a_to_c = 45 + degrees if top else 45 - degrees
    relative_y = calc_a(degrees_a_to_c, hypotenuse)
    relative_x = calc_b(degrees_a_to_c, hypotenuse)
    if top:
        return point_a[0] + relative_x, point_a[1] - relative_y
    else:
        return point_a[0] + relative_x, point_a[1] + relative_y
