"""
CJ 대한통운 미래기술 챌린지
경로 최적화 + 적재 최적화 통합 솔루션
"""

import json
import polars as pl
import numpy as np
import math
from pyvrp import Model
from pyvrp.stop import MaxRuntime
from pyproj import Transformer
# --- py3dbp constants.py ---
class RotationType:
    RT_WHD = 0
    RT_HWD = 1
    RT_HDW = 2
    RT_DHW = 3
    RT_DWH = 4
    RT_WDH = 5

    ALL = [RT_WHD, RT_HWD, RT_HDW, RT_DHW, RT_DWH, RT_WDH]
    # un upright or un updown
    Notupdown = [RT_WHD,RT_HWD]
 
class Axis:
    WIDTH = 0
    HEIGHT = 1
    DEPTH = 2

    ALL = [WIDTH, HEIGHT, DEPTH]

# --- py3dbp auxiliary_methods.py ---
from decimal import Decimal

def rectIntersect(item1, item2, x, y):
    d1 = item1.getDimension()
    d2 = item2.getDimension()

    cx1 = item1.position[x] + d1[x]/2
    cy1 = item1.position[y] + d1[y]/2
    cx2 = item2.position[x] + d2[x]/2
    cy2 = item2.position[y] + d2[y]/2

    ix = max(cx1, cx2) - min(cx1, cx2)
    iy = max(cy1, cy2) - min(cy1, cy2)

    return ix < (d1[x]+d2[x])/2 and iy < (d1[y]+d2[y])/2

def intersect(item1, item2):
    return (
        rectIntersect(item1, item2, Axis.WIDTH, Axis.HEIGHT) and
        rectIntersect(item1, item2, Axis.HEIGHT, Axis.DEPTH) and
        rectIntersect(item1, item2, Axis.WIDTH, Axis.DEPTH)
    )

def getLimitNumberOfDecimals(number_of_decimals):
    return Decimal('1.{}'.format('0' * number_of_decimals))

def set2Decimal(value, number_of_decimals=0):
    number_of_decimals = getLimitNumberOfDecimals(number_of_decimals)
    return Decimal(value).quantize(number_of_decimals)

# --- py3dbp main.py ---
import numpy as np
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from collections import Counter
import copy
DEFAULT_NUMBER_OF_DECIMALS = 0
START_POSITION = [0, 0, 0]

class Item:
    def __init__(self, partno, name, typeof, WHD, weight, level, loadbear, updown, color):
        self.partno = partno
        self.name = name
        self.typeof = typeof
        self.width = WHD[0]
        self.height = WHD[1]
        self.depth = WHD[2]
        self.weight = weight
        self.level = level
        self.loadbear = loadbear
        self.updown = updown if typeof == 'cube' else False
        self.color = color
        self.rotation_type = 0
        self.position = START_POSITION
        self.number_of_decimals = DEFAULT_NUMBER_OF_DECIMALS
    def formatNumbers(self, number_of_decimals):
        self.width = set2Decimal(self.width, number_of_decimals)
        self.height = set2Decimal(self.height, number_of_decimals)
        self.depth = set2Decimal(self.depth, number_of_decimals)
        self.weight = set2Decimal(self.weight, number_of_decimals)
        self.number_of_decimals = number_of_decimals
    def string(self):
        return "%s(%sx%sx%s, weight: %s) pos(%s) rt(%s) vol(%s)" % (
            self.partno, self.width, self.height, self.depth, self.weight,
            self.position, self.rotation_type, self.getVolume())
    def getVolume(self):
        return set2Decimal(self.width * self.height * self.depth, self.number_of_decimals)
    def getMaxArea(self):
        a = sorted([self.width, self.height, self.depth], reverse=True) if self.updown == True else [self.width, self.height, self.depth]
        return set2Decimal(a[0] * a[1], self.number_of_decimals)
    def getDimension(self):
        if self.rotation_type == RotationType.RT_WHD:
            dimension = [self.width, self.height, self.depth]
        elif self.rotation_type == RotationType.RT_HWD:
            dimension = [self.height, self.width, self.depth]
        elif self.rotation_type == RotationType.RT_HDW:
            dimension = [self.height, self.depth, self.width]
        elif self.rotation_type == RotationType.RT_DHW:
            dimension = [self.depth, self.height, self.width]
        elif self.rotation_type == RotationType.RT_DWH:
            dimension = [self.depth, self.width, self.height]
        elif self.rotation_type == RotationType.RT_WDH:
            dimension = [self.width, self.depth, self.height]
        else:
            dimension = []
        return dimension

class Bin:
    def __init__(self, partno, WHD, max_weight, corner=0, put_type=1):
        self.partno = partno
        self.width = WHD[0]
        self.height = WHD[1]
        self.depth = WHD[2]
        self.max_weight = max_weight
        self.corner = corner
        self.items = []
        self.fit_items = np.array([[0, WHD[0], 0, WHD[1], 0, 0]])
        self.unfitted_items = []
        self.number_of_decimals = DEFAULT_NUMBER_OF_DECIMALS
        self.fix_point = False
        self.check_stable = False
        self.support_surface_ratio = 0
        self.put_type = put_type
        self.gravity = []
    def formatNumbers(self, number_of_decimals):
        self.width = set2Decimal(self.width, number_of_decimals)
        self.height = set2Decimal(self.height, number_of_decimals)
        self.depth = set2Decimal(self.depth, number_of_decimals)
        self.max_weight = set2Decimal(self.max_weight, number_of_decimals)
        self.number_of_decimals = number_of_decimals
    def string(self):
        return "%s(%sx%sx%s, max_weight:%s) vol(%s)" % (
            self.partno, self.width, self.height, self.depth, self.max_weight,
            self.getVolume())
    def getVolume(self):
        return set2Decimal(self.width * self.height * self.depth, self.number_of_decimals)
    def getTotalWeight(self):
        total_weight = 0
        for item in self.items:
            total_weight += item.weight
        return set2Decimal(total_weight, self.number_of_decimals)
    def putItem(self, item, pivot, axis=None):
        fit = False
        valid_item_position = item.position
        item.position = pivot
        rotate = RotationType.ALL if item.updown == True else RotationType.Notupdown
        for i in range(0, len(rotate)):
            item.rotation_type = i
            dimension = item.getDimension()
            if (
                self.width < pivot[0] + dimension[0] or
                self.height < pivot[1] + dimension[1] or
                self.depth < pivot[2] + dimension[2]
            ):
                continue
            fit = True
            for current_item_in_bin in self.items:
                if intersect(current_item_in_bin, item):
                    fit = False
                    break
            if fit:
                if self.getTotalWeight() + item.weight > self.max_weight:
                    fit = False
                    return fit
                if self.fix_point == True:
                    [w, h, d] = dimension
                    [x, y, z] = [float(pivot[0]), float(pivot[1]), float(pivot[2])]
                    for i in range(3):
                        y = self.checkHeight([x, x+float(w), y, y+float(h), z, z+float(d)])
                        x = self.checkWidth([x, x+float(w), y, y+float(h), z, z+float(d)])
                        z = self.checkDepth([x, x+float(w), y, y+float(h), z, z+float(d)])
                    if self.check_stable == True:
                        item_area_lower = int(dimension[0] * dimension[1])
                        support_area_upper = 0
                        for i in self.fit_items:
                            if z == i[5]:
                                area = len(set([j for j in range(int(x), int(x+int(w)))]) & set([j for j in range(int(i[0]), int(i[1]))])) * \
                                       len(set([j for j in range(int(y), int(y+int(h)))]) & set([j for j in range(int(i[2]), int(i[3]))]))
                                support_area_upper += area
                        if support_area_upper / item_area_lower < self.support_surface_ratio:
                            four_vertices = [[x, y], [x+float(w), y], [x, y+float(h)], [x+float(w), y+float(h)]]
                            c = [False, False, False, False]
                            for i in self.fit_items:
                                if z == i[5]:
                                    for jdx, j in enumerate(four_vertices):
                                        if (i[0] <= j[0] <= i[1]) and (i[2] <= j[1] <= i[3]):
                                            c[jdx] = True
                            if False in c:
                                item.position = valid_item_position
                                fit = False
                                return fit
                    self.fit_items = np.append(self.fit_items, np.array([[x, x+float(w), y, y+float(h), z, z+float(d)]]), axis=0)
                    item.position = [set2Decimal(x), set2Decimal(y), set2Decimal(z)]
                if fit:
                    self.items.append(copy.deepcopy(item))
            else:
                item.position = valid_item_position
            return fit
        else:
            item.position = valid_item_position
        return fit
    def checkDepth(self, unfix_point):
        z_ = [[0, 0], [float(self.depth), float(self.depth)]]
        for j in self.fit_items:
            x_bottom = set([i for i in range(int(j[0]), int(j[1]))])
            x_top = set([i for i in range(int(unfix_point[0]), int(unfix_point[1]))])
            y_bottom = set([i for i in range(int(j[2]), int(j[3]))])
            y_top = set([i for i in range(int(unfix_point[2]), int(unfix_point[3]))])
            if len(x_bottom & x_top) != 0 and len(y_bottom & y_top) != 0:
                z_.append([float(j[4]), float(j[5])])
        top_depth = unfix_point[5] - unfix_point[4]
        z_ = sorted(z_, key=lambda z_: z_[1])
        for j in range(len(z_)-1):
            if z_[j+1][0] - z_[j][1] >= top_depth:
                return z_[j][1]
        return unfix_point[4]
    def checkWidth(self, unfix_point):
        x_ = [[0, 0], [float(self.width), float(self.width)]]
        for j in self.fit_items:
            z_bottom = set([i for i in range(int(j[4]), int(j[5]))])
            z_top = set([i for i in range(int(unfix_point[4]), int(unfix_point[5]))])
            y_bottom = set([i for i in range(int(j[2]), int(j[3]))])
            y_top = set([i for i in range(int(unfix_point[2]), int(unfix_point[3]))])
            if len(z_bottom & z_top) != 0 and len(y_bottom & y_top) != 0:
                x_.append([float(j[0]), float(j[1])])
        top_width = unfix_point[1] - unfix_point[0]
        x_ = sorted(x_, key=lambda x_: x_[1])
        for j in range(len(x_)-1):
            if x_[j+1][0] - x_[j][1] >= top_width:
                return x_[j][1]
        return unfix_point[0]
    def checkHeight(self, unfix_point):
        y_ = [[0, 0], [float(self.height), float(self.height)]]
        for j in self.fit_items:
            x_bottom = set([i for i in range(int(j[0]), int(j[1]))])
            x_top = set([i for i in range(int(unfix_point[0]), int(unfix_point[1]))])
            z_bottom = set([i for i in range(int(j[4]), int(j[5]))])
            z_top = set([i for i in range(int(unfix_point[4]), int(unfix_point[5]))])
            if len(x_bottom & x_top) != 0 and len(z_bottom & z_top) != 0:
                y_.append([float(j[2]), float(j[3])])
        top_height = unfix_point[3] - unfix_point[2]
        y_ = sorted(y_, key=lambda y_: y_[1])
        for j in range(len(y_)-1):
            if y_[j+1][0] - y_[j][1] >= top_height:
                return y_[j][1]
        return unfix_point[2]
    def addCorner(self):
        if self.corner != 0:
            corner = set2Decimal(self.corner)
            corner_list = []
            for i in range(8):
                a = Item(
                    partno='corner{}'.format(i),
                    name='corner', 
                    typeof='cube',
                    WHD=(corner, corner, corner), 
                    weight=0, 
                    level=0, 
                    loadbear=0, 
                    updown=True, 
                    color='#000000')
                corner_list.append(a)
            return corner_list
    def putCorner(self, info, item):
        fit = False
        x = set2Decimal(self.width - self.corner)
        y = set2Decimal(self.height - self.corner)
        z = set2Decimal(self.depth - self.corner)
        pos = [[0, 0, 0], [0, 0, z], [0, y, z], [0, y, 0], [x, y, 0], [x, 0, 0], [x, 0, z], [x, y, z]]
        item.position = pos[info]
        self.items.append(item)
        corner = [float(item.position[0]), float(item.position[0])+float(self.corner), float(item.position[1]), float(item.position[1])+float(self.corner), float(item.position[2]), float(item.position[2])+float(self.corner)]
        self.fit_items = np.append(self.fit_items, np.array([corner]), axis=0)
        return
    def clearBin(self):
        self.items = []
        self.fit_items = np.array([[0, self.width, 0, self.height, 0, 0]])
        return

class Packer:
    def __init__(self):
        self.bins = []
        self.items = []
        self.unfit_items = []
        self.total_items = 0
        self.binding = []
    def addBin(self, bin):
        return self.bins.append(bin)
    def addItem(self, item):
        self.total_items = len(self.items) + 1
        return self.items.append(item)
    def pack2Bin(self, bin, item, fix_point, check_stable, support_surface_ratio):
        fitted = False
        bin.fix_point = fix_point
        bin.check_stable = check_stable
        bin.support_surface_ratio = support_surface_ratio
        if bin.corner != 0 and not bin.items:
            corner_lst = bin.addCorner()
            for i in range(len(corner_lst)):
                bin.putCorner(i, corner_lst[i])
        elif not bin.items:
            response = bin.putItem(item, item.position)
            if not response:
                bin.unfitted_items.append(item)
            return
        for axis in range(0, 3):
            items_in_bin = bin.items
            for ib in items_in_bin:
                pivot = [0, 0, 0]
                w, h, d = ib.getDimension()
                if axis == Axis.WIDTH:
                    pivot = [ib.position[0] + w, ib.position[1], ib.position[2]]
                elif axis == Axis.HEIGHT:
                    pivot = [ib.position[0], ib.position[1] + h, ib.position[2]]
                elif axis == Axis.DEPTH:
                    pivot = [ib.position[0], ib.position[1], ib.position[2] + d]
                if bin.putItem(item, pivot, axis):
                    fitted = True
                    break
            if fitted:
                break
        if not fitted:
            bin.unfitted_items.append(item)
    def sortBinding(self, bin):
        b, front, back = [], [], []
        for i in range(len(self.binding)):
            b.append([])
            for item in self.items:
                if item.name in self.binding[i]:
                    b[i].append(item)
                elif item.name not in self.binding:
                    if len(b[0]) == 0 and item not in front:
                        front.append(item)
                    elif item not in back and item not in front:
                        back.append(item)
        min_c = min([len(i) for i in b])
        sort_bind = []
        for i in range(min_c):
            for j in range(len(b)):
                sort_bind.append(b[j][i])
        for i in b:
            for j in i:
                if j not in sort_bind:
                    self.unfit_items.append(j)
        self.items = front + sort_bind + back
        return
    def putOrder(self):
        r = []
        for i in self.bins:
            if i.put_type == 2:
                i.items.sort(key=lambda item: item.position[0], reverse=False)
                i.items.sort(key=lambda item: item.position[1], reverse=False)
                i.items.sort(key=lambda item: item.position[2], reverse=False)
            elif i.put_type == 1:
                i.items.sort(key=lambda item: item.position[1], reverse=False)
                i.items.sort(key=lambda item: item.position[2], reverse=False)
                i.items.sort(key=lambda item: item.position[0], reverse=False)
            else:
                pass
        return
    def gravityCenter(self, bin):
        w = int(bin.width)
        h = int(bin.height)
        d = int(bin.depth)
        area1 = [set(range(0, w//2+1)), set(range(0, h//2+1)), 0]
        area2 = [set(range(w//2+1, w+1)), set(range(0, h//2+1)), 0]
        area3 = [set(range(0, w//2+1)), set(range(h//2+1, h+1)), 0]
        area4 = [set(range(w//2+1, w+1)), set(range(h//2+1, h+1)), 0]
        area = [area1, area2, area3, area4]
        for i in bin.items:
            x_st = int(i.position[0])
            y_st = int(i.position[1])
            if i.rotation_type == 0:
                x_ed = int(i.position[0] + i.width)
                y_ed = int(i.position[1] + i.height)
            elif i.rotation_type == 1:
                x_ed = int(i.position[0] + i.height)
                y_ed = int(i.position[1] + i.width)
            elif i.rotation_type == 2:
                x_ed = int(i.position[0] + i.height)
                y_ed = int(i.position[1] + i.depth)
            elif i.rotation_type == 3:
                x_ed = int(i.position[0] + i.depth)
                y_ed = int(i.position[1] + i.height)
            elif i.rotation_type == 4:
                x_ed = int(i.position[0] + i.depth)
                y_ed = int(i.position[1] + i.width)
            elif i.rotation_type == 5:
                x_ed = int(i.position[0] + i.width)
                y_ed = int(i.position[1] + i.depth)
            x_set = set(range(x_st, int(x_ed)+1))
            y_set = set(range(y_st, y_ed+1))
            for j in range(len(area)):
                if x_set.issubset(area[j][0]) and y_set.issubset(area[j][1]):
                    area[j][2] += int(i.weight)
                    break
                elif x_set.issubset(area[j][0]) == True and y_set.issubset(area[j][1]) == False and len(y_set & area[j][1]) != 0:
                    y = len(y_set & area[j][1]) / (y_ed - y_st) * int(i.weight)
                    area[j][2] += y
                    if j >= 2:
                        area[j-2][2] += (int(i.weight) - x)
                    else:
                        area[j+2][2] += (int(i.weight) - y)
                    break
                elif x_set.issubset(area[j][0]) == False and y_set.issubset(area[j][1]) == True and len(x_set & area[j][0]) != 0:
                    x = len(x_set & area[j][0]) / (x_ed - x_st) * int(i.weight)
                    area[j][2] += x
                    if j >= 2:
                        area[j-2][2] += (int(i.weight) - x)
                    else:
                        area[j+2][2] += (int(i.weight) - x)
                    break
                elif x_set.issubset(area[j][0])== False and y_set.issubset(area[j][1]) == False and len(y_set & area[j][1]) != 0  and len(x_set & area[j][0]) != 0:
                    all = (y_ed - y_st) * (x_ed - x_st)
                    y = len(y_set & area[0][1])
                    y_2 = y_ed - y_st - y
                    x = len(x_set & area[0][0])
                    x_2 = x_ed - x_st - x
                    area[0][2] += x * y / all * int(i.weight)
                    area[1][2] += x_2 * y / all * int(i.weight)
                    area[2][2] += x * y_2 / all * int(i.weight)
                    area[3][2] += x_2 * y_2 / all * int(i.weight)
                    break
        r = [area[0][2], area[1][2], area[2][2], area[3][2]]
        result = []
        for i in r:
            result.append(round(i / sum(r) * 100, 2))
        return result
    def pack(self, bigger_first=False, distribute_items=True, fix_point=True, check_stable=True, support_surface_ratio=0.75, binding=[], number_of_decimals=DEFAULT_NUMBER_OF_DECIMALS):
        for bin in self.bins:
            bin.formatNumbers(number_of_decimals)
        for item in self.items:
            item.formatNumbers(number_of_decimals)
        self.binding = binding
        self.bins.sort(key=lambda bin: bin.getVolume(), reverse=bigger_first)
        self.items.sort(key=lambda item: item.getVolume(), reverse=bigger_first)
        self.items.sort(key=lambda item: item.loadbear, reverse=True)
        self.items.sort(key=lambda item: item.level, reverse=False)
        if binding != []:
            self.sortBinding(bin)
        for idx, bin in enumerate(self.bins):
            for item in self.items:
                self.pack2Bin(bin, item, fix_point, check_stable, support_surface_ratio)
            if binding != []:
                self.items.sort(key=lambda item: item.getVolume(), reverse=bigger_first)
                self.items.sort(key=lambda item: item.loadbear, reverse=True)
                self.items.sort(key=lambda item: item.level, reverse=False)
                bin.items = []
                bin.unfitted_items = self.unfit_items
                bin.fit_items = np.array([[0, bin.width, 0, bin.height, 0, 0]])
                for item in self.items:
                    self.pack2Bin(bin, item, fix_point, check_stable, support_surface_ratio)
            self.bins[idx].gravity = self.gravityCenter(bin)
            if distribute_items:
                for bitem in bin.items:
                    no = bitem.partno
                    for item in self.items:
                        if item.partno == no:
                            self.items.remove(item)
                            break
        self.putOrder()
        if self.items != []:
            self.unfit_items = copy.deepcopy(self.items)
            self.items = []

class Painter:
    def __init__(self, bins):
        self.items = bins.items
        self.width = bins.width
        self.height = bins.height
        self.depth = bins.depth
    def _plotCube(self, ax, x, y, z, dx, dy, dz, color='red', mode=2, linewidth=1, text="", fontsize=15, alpha=0.5):
        xx = [x, x, x+dx, x+dx, x]
        yy = [y, y+dy, y+dy, y, y]
        kwargs = {'alpha': 1, 'color': color, 'linewidth': linewidth}
        if mode == 1:
            ax.plot3D(xx, yy, [z]*5, **kwargs)
            ax.plot3D(xx, yy, [z+dz]*5, **kwargs)
            ax.plot3D([x, x], [y, y], [z, z+dz], **kwargs)
            ax.plot3D([x, x], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y+dy, y+dy], [z, z+dz], **kwargs)
            ax.plot3D([x+dx, x+dx], [y, y], [z, z+dz], **kwargs)
        else:
            p = Rectangle((x, y), dx, dy, fc=color, ec='black', alpha=alpha)
            p2 = Rectangle((x, y), dx, dy, fc=color, ec='black', alpha=alpha)
            p3 = Rectangle((y, z), dy, dz, fc=color, ec='black', alpha=alpha)
            p4 = Rectangle((y, z), dy, dz, fc=color, ec='black', alpha=alpha)
            p5 = Rectangle((x, z), dx, dz, fc=color, ec='black', alpha=alpha)
            p6 = Rectangle((x, z), dx, dz, fc=color, ec='black', alpha=alpha)
            ax.add_patch(p)
            ax.add_patch(p2)
            ax.add_patch(p3)
            ax.add_patch(p4)
            ax.add_patch(p5)
            ax.add_patch(p6)
            if text != "":
                ax.text((x+dx/2), (y+dy/2), (z+dz/2), str(text), color='black', fontsize=fontsize, ha='center', va='center')
            art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
            art3d.pathpatch_2d_to_3d(p2, z=z+dz, zdir="z")
            art3d.pathpatch_2d_to_3d(p3, z=x, zdir="x")
            art3d.pathpatch_2d_to_3d(p4, z=x+dx, zdir="x")
            art3d.pathpatch_2d_to_3d(p5, z=y, zdir="y")
            art3d.pathpatch_2d_to_3d(p6, z=y+dy, zdir="y")
    def _plotCylinder(self, ax, x, y, z, dx, dy, dz, color='red', mode=2, text="", fontsize=10, alpha=0.2):
        p = Circle((x+dx/2, y+dy/2), radius=dx/2, color=color, alpha=0.5)
        p2 = Circle((x+dx/2, y+dy/2), radius=dx/2, color=color, alpha=0.5)
        ax.add_patch(p)
        ax.add_patch(p2)
        art3d.pathpatch_2d_to_3d(p, z=z, zdir="z")
        art3d.pathpatch_2d_to_3d(p2, z=z+dz, zdir="z")
        center_z = np.linspace(0, dz, 10)
        theta = np.linspace(0, 2*np.pi, 10)
        theta_grid, z_grid = np.meshgrid(theta, center_z)
        x_grid = dx / 2 * np.cos(theta_grid) + x + dx / 2
        y_grid = dy / 2 * np.sin(theta_grid) + y + dy / 2
        z_grid = z_grid + z
        ax.plot_surface(x_grid, y_grid, z_grid, shade=False, fc=color, alpha=alpha, color=color)
        if text != "":
            ax.text((x+dx/2), (y+dy/2), (z+dz/2), str(text), color='black', fontsize=fontsize, ha='center', va='center')
    def plotBoxAndItems(self, title="", alpha=0.2, write_num=False, fontsize=10):
        fig = plt.figure()
        axGlob = plt.axes(projection='3d')
        self._plotCube(axGlob, 0, 0, 0, float(self.width), float(self.height), float(self.depth), color='black', mode=1, linewidth=2, text="")
        counter = 0
        for item in self.items:
            rt = item.rotation_type
            x, y, z = item.position
            [w, h, d] = item.getDimension()
            color = item.color
            text = item.partno if write_num else ""
            if item.typeof == 'cube':
                self._plotCube(axGlob, float(x), float(y), float(z), float(w), float(h), float(d), color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)
            elif item.typeof == 'cylinder':
                self._plotCylinder(axGlob, float(x), float(y), float(z), float(w), float(h), float(d), color=color, mode=2, text=text, fontsize=fontsize, alpha=alpha)
            counter = counter + 1
        plt.title(title)
        self.setAxesEqual(axGlob)
        return plt
    def setAxesEqual(self, ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

import warnings
warnings.filterwarnings('ignore')


class CJOptimizer:
    """CJ 대한통운 경로 및 적재 최적화 클래스"""
    
    def __init__(self):
        # 트럭 규격 (width x height x depth)
        self.truck_dimensions = (160, 270, 170)
        self.truck_capacity = 160 * 280 * 180
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        
    def load_data(self, data_file, distance_file):
        """데이터 파일 로드"""
        print("📂 데이터 로딩 중...")

        with open(data_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        depot = json_data["depot"]
        destinations = json_data["destinations"]
        orders = json_data["orders"]

        # 1️⃣ Depot Row
        depot_row = {
            "Vehicle_ID": 0,
            "Route_Order": 0,
            "Destination": depot["destination"],
            "Order_Number": "DEPOT",
            "Box_ID": None,
            "Stacking_Order": None,
            "Lower_Left_X": 0,
            "Lower_Left_Y": 0,
            "Lower_Left_Z": 0,
            "Longitude": depot["location"]["longitude"],
            "Latitude": depot["location"]["latitude"],
            "Box_Width": 0,
            "Box_Length": 0,
            "Box_Height": 0,
            "Volume": 0
        }
        rows = [depot_row]

        # 2️⃣ destination_id → 좌표 매핑
        dest_coords = {
            d["destination_id"]: (d["location"]["longitude"], d["location"]["latitude"])
            for d in destinations
        }

        # 3️⃣ Orders Rows
        for order in orders:
            dest_id = order["destination"]
            longitude, latitude = dest_coords.get(dest_id, (None, None))
            width = order["dimension"]["width"]
            length = order["dimension"]["length"]
            height = order["dimension"]["height"]

            row = {
                "Vehicle_ID": 0,
                "Route_Order": 0,
                "Destination": dest_id,
                "Order_Number": order["order_number"],
                "Box_ID": order["box_id"],
                "Stacking_Order": 0,
                "Lower_Left_X": 0,
                "Lower_Left_Y": 0,
                "Lower_Left_Z": 0,
                "Longitude": longitude,
                "Latitude": latitude,
                "Box_Width": width,
                "Box_Length": length,
                "Box_Height": height,
                "Volume": width * length * height
            }
            rows.append(row)

        # 4️⃣ DataFrame 생성
        self.df = pl.DataFrame(rows)

        # 5️⃣ 거리 매트릭스
        self.matrix = pl.read_csv(distance_file, separator='\t')

        print(f"✅ 주문 데이터: {len(self.df) - 1}건")
        print(f"✅ 거리 매트릭스: {len(self.matrix)}건")

    def route_optimization(self):
        """경로 최적화 수행"""
        print("\n🚛 경로 최적화 시작...")
        
        # 좌표 변환 (WGS84 -> Web Mercator)
        coords = [
            self.transformer.transform(lon, lat) 
            for lon, lat in zip(self.df['Longitude'], self.df['Latitude'])
        ]
        
        # 필요 차량 수 계산
        total_volume = self.df.select('Volume').sum()[0, 0]
        self.load_factor = 0.7
        num_vehicles = math.ceil(total_volume / (self.truck_capacity * self.load_factor))

        
        print(f"📊 총 부피: {total_volume:,}")
        print(f"🚛 필요 차량 수: {num_vehicles}")
        
        # PyVRP 모델 생성
        m = Model()
        m.add_vehicle_type(
            num_available=num_vehicles + 3, 
            capacity=int(self.truck_capacity * 0.62), 
            fixed_cost=150000,
            unit_distance_cost=500
        )
        
        # 인덱스 매핑 생성
        self.index_to_order_number = []
        self.index_to_location_name = []
        
        # 창고(Depot) 추가
        depot = m.add_depot(x=coords[0][0], y=coords[0][1], name="Depot")
        self.index_to_order_number.append(self.df['Order_Number'][0])
        self.index_to_location_name.append("Depot")
        
        # 배송지 추가
        for idx, row in enumerate(self.df.iter_rows(named=True)):
            if idx != 0:  # 첫 번째는 창고이므로 제외
                m.add_client(
                    x=coords[idx][0],
                    y=coords[idx][1],
                    delivery=row['Volume'],
                    name=row['Destination']
                )
                self.index_to_order_number.append(row['Order_Number'])
                self.index_to_location_name.append(row['Destination'])
        
        distance_dict = {}
        for row in self.matrix.iter_rows(named=True):
            key = (row['ORIGIN'], row['DESTINATION'])
            distance_dict[key] = row['DISTANCE_METER'] / 1000

        # 거리 매트릭스 추가
        for frm in m.locations:
            for to in m.locations:
                origin = frm.name
                destination = to.name
                if origin != destination:
                    distance = distance_dict.get((origin, destination))
                    m.add_edge(frm, to, distance=distance)
                else:
                    m.add_edge(frm, to, distance=0)
        
        # 경로 최적화 실행
        print("🔄 경로 최적화 실행 중... (최대 900초)")
        res = m.solve(stop=MaxRuntime(900), display=False)

        
        # 결과 처리
        routes = [list(route) for route in res.best.routes()]
        self.process_routes(routes)
        
        print(f"✅ 경로 최적화 완료 - {len(routes)}개 경로 생성")
        
    def process_routes(self, routes):
        """경로 결과 처리"""
        new_df = self.df.clear()
        
        for vehicle_id, route in enumerate(routes):
            print(f"▶ Vehicle {vehicle_id} route: {route}")
            if not route:
                continue
            
            route_str = [str(x) for x in route]
            order_map = {s: idx for idx, s in enumerate(route_str)}

            print(f"▶ order_map: {order_map}")
            
            # Depot 제외
            filtered_df = (
                self.df
                .filter(
                    pl.col('Order_Number').is_in(route_str)
                )
                .with_columns(
                    pl.col('Order_Number').map_elements(
                        lambda x: order_map.get(x, float('inf')),
                        return_dtype=pl.Int64
                    ).alias('route_order')
                )
                .sort('route_order')
                .drop('route_order')
            )
            
            # 창고 행 생성
            depot_row = pl.DataFrame({
                col: [None] if col != 'Destination' else ['Depot'] 
                for col in filtered_df.columns
            }).cast(dict(zip(filtered_df.columns, filtered_df.dtypes)))
            
            # 창고 -> 배송지들 -> 창고 순서로 결합
            result = pl.concat([depot_row, filtered_df, depot_row])
            
            # Route_Order와 Vehicle_ID 추가
            result = result.with_columns([
                (pl.int_range(pl.len()) + 1).alias('Route_Order'),
                pl.lit(vehicle_id).alias('Vehicle_ID').cast(pl.Int64)
            ])
            
            new_df = pl.concat([new_df, result])
        
        self.route_df = new_df
        
    def load_optimization(self):
        """적재 최적화 수행"""
        print("\n📦 적재 최적화 시작...")
        
        all_results = []
        
        # 각 차량별로 적재 최적화 수행
        vehicle_ids = self.route_df.select('Vehicle_ID').unique().to_series().to_list()
        
        for vehicle_id in vehicle_ids:
            if vehicle_id is None:
                continue
                
            print(f"🚛 Vehicle {vehicle_id} 적재 최적화...")
            
            # 해당 차량의 배송지만 필터링 (창고 제외)
            vehicle_items = self.route_df.filter(
                (pl.col('Vehicle_ID') == vehicle_id) & 
                (pl.col('Destination') != 'Depot')
            )
            
            if len(vehicle_items) == 0:
                continue
            
            # Stacking_Order는 route_order 역순으로 설정
            vehicle_items = vehicle_items.with_columns(
                pl.int_range(vehicle_items.height, 0, -1).alias('Stacking_Order')
            )
            
            # 3D 빈 패킹 수행
            packer = Packer()
            
            # 트럭 빈 생성 (width, height, depth, max_weight, max_items)
            # 트럭 빈 생성
            truck = Bin(
                partno='Truck',
                WHD=(160, 170, 270),
                max_weight=999999,
                put_type=1
            )
            packer.addBin(truck)

            # 아이템 추가
            for row in vehicle_items.iter_rows(named=True):
                item = Item(
                    partno=row["Box_ID"],
                    name=row["Box_ID"],
                    typeof='cube',
                    WHD=(
                        int(row["Box_Width"]),
                        int(row["Box_Height"]),
                        int(row["Box_Length"])
                    ),
                    weight=1,
                    level=row["Stacking_Order"],
                    updown=True,
                    loadbear=999999,
                    color='#FFCC00'
                )
                packer.addItem(item)

            # 패킹 실행
            packer.pack(
                fix_point=True,
                check_stable=False,
                bigger_first=False
            )
            
            # 결과 수집
            for item in packer.bins[0].items:
                all_results.append({
                    "Vehicle_ID": vehicle_id,
                    "Box_ID": item.name,
                    "Lower_Left_X": item.position[0],
                    "Lower_Left_Y": item.position[2],  # Y와 Z 좌표 교환
                    "Lower_Left_Z": item.position[1],
                    "Box_Width": item.width,
                    "Box_Length": item.depth,
                    "Box_Height": item.height
                })
        
        # 적재 결과 DataFrame 생성
        self.load_df = pl.DataFrame(all_results).with_columns([
            pl.col("Lower_Left_X").cast(pl.Float64),
            pl.col("Lower_Left_Y").cast(pl.Float64),
            pl.col("Lower_Left_Z").cast(pl.Float64),
            pl.col("Box_Width").cast(pl.Float64),
            pl.col("Box_Length").cast(pl.Float64),
            pl.col("Box_Height").cast(pl.Float64),
        ])
        
        print(f"✅ 적재 최적화 완료 - {len(self.load_df)}개 박스 배치")
        
    def save_results(self, output_file='Result.xlsx'):
        """결과를 Excel 파일로 저장"""
        print(f"\n💾 결과 저장 중: {output_file}")
        
        # 경로 최적화 결과
        route_pandas = self.route_df
        # 적재 최적화 결과
        load_pandas = self.load_df

        joined = route_pandas.join(load_pandas, on='Box_ID', how='left', suffix='_qwer')

        cols_to_replace = ['Lower_Left_X', 'Lower_Left_Y', 'Lower_Left_Z']

        route_pandas = joined.with_columns([
            pl.coalesce([pl.col(f"{col}_qwer"), pl.col(col)]).alias(col)
            for col in cols_to_replace
        ]).select([col for col in joined.columns if not col.endswith('_qwer')])

        route_pandas = route_pandas.with_columns([
            pl.when(pl.col("Destination") != "Depot")
            .then(
                pl.col("Route_Order")
                .rank("dense", descending=True)
                .over("Vehicle_ID")
                .cast(pl.Int64) - 1
            )
            .otherwise(None)
            .alias("Stacking_Order")
        ])

        route_pandas = route_pandas.with_columns(
            pl.col("Lower_Left_Z").rank(method="ordinal").over("Vehicle_ID").alias('Stacking_Order')
            )
        
        route_pandas = route_pandas.with_columns(
            pl.col('Order_Number').cast(pl.Int64)
        )

        if "Volume" in route_pandas.columns:
            route_pandas = route_pandas.drop("Volume")

        route_pandas.write_excel(output_file)
        
        print(f"✅ 결과 저장 완료: {output_file}")
        
    def run_optimization(self, data_file='data.json', distance_file='distance-data.txt', output_file='Result.xlsx'):
        """전체 최적화 프로세스 실행"""
        print("🚀 CJ 대한통운 경로 및 적재 최적화 시작")
        print("=" * 50)
        
        try:
            # 1. 데이터 로드
            self.load_data(data_file, distance_file)
            
            # 2. 경로 최적화
            self.route_optimization()
            
            # 3. 적재 최적화
            self.load_optimization()
            
            # 4. 결과 저장
            self.save_results(output_file)
            
            print("\n" + "=" * 50)
            print("🎉 최적화 완료!")
            print(f"📊 총 {len(self.route_df.select('Vehicle_ID').unique()) - 1}대 차량")
            print(f"📦 총 {len(self.load_df)}개 박스 최적 배치")
            print(f"📁 결과 파일: {output_file}")
            print("=== self.route_df ===")
            print(self.route_df)

            print("=== self.load_df ===")
            print(self.load_df)
            
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            raise


def main():
    optimizer = CJOptimizer()
    optimizer.run_optimization(
        data_file=r"data.json",
        distance_file=r"distance-data.txt",
        output_file=r"Result.xlsx"
    )


if __name__ == "__main__":
    main()
