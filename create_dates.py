import numpy as np
import cv2
import random
import glob


default_root = "./"

path_blue_1 = [
            default_root +"/datas/小地图素材/background/蓝二半血_2.png",
            default_root +"/datas/小地图素材/background/蓝二残血_0.png",
            default_root +"/datas/小地图素材/background/蓝二残血_2.png",
            default_root +"/datas/小地图素材/background/蓝二满血_0.png",
            default_root +"/datas/小地图素材/background/蓝二满血_1.png",
            default_root +"/datas/小地图素材/background/蓝一半血_1.png",
            default_root +"/datas/小地图素材/background/蓝一残血_1.png",
            default_root +"/datas/小地图素材/background/蓝一满血_0.png",
            default_root +"/datas/小地图素材/background/蓝一满血_1.png",
            default_root +"/datas/小地图素材/background/no_tower.png",
        ]

path_blue_23 = [
            default_root +"/datas/小地图素材/background/蓝二半血_2.png",
            default_root +"/datas/小地图素材/background/蓝二残血_0.png",
            default_root +"/datas/小地图素材/background/蓝二残血_2.png",
            default_root +"/datas/小地图素材/background/蓝二满血_0.png",
            default_root +"/datas/小地图素材/background/蓝二满血_1.png",
            default_root +"/datas/小地图素材/background/no_tower.png",
        ]

path_red_1 = [
            default_root +"/datas/小地图素材/background/红二半血_3.png",
            default_root +"/datas/小地图素材/background/红二残血_4.png",
            default_root +"/datas/小地图素材/background/红二满血_0.png",
            default_root +"/datas/小地图素材/background/红二满血_1.png",
            default_root +"/datas/小地图素材/background/红二满血_2.png",
            default_root +"/datas/小地图素材/background/红二残血_1.png",
            default_root +"/datas/小地图素材/background/红一满血_0.png",
            default_root +"/datas/小地图素材/background/红一满血_1.png",
            default_root +"/datas/小地图素材/background/no_tower.png",
        ]

path_red_23 = [
            default_root +"/datas/小地图素材/background/红二半血_3.png",
            default_root +"/datas/小地图素材/background/红二残血_4.png",
            default_root +"/datas/小地图素材/background/红二满血_0.png",
            default_root +"/datas/小地图素材/background/红二满血_1.png",
            default_root +"/datas/小地图素材/background/红二满血_2.png",
            default_root +"/datas/小地图素材/background/红二残血_1.png",
            default_root +"/datas/小地图素材/background/no_tower.png",
        ]


path_monster = [
            default_root +"/datas/小地图素材/background/no_buff_monster.png",
            default_root +"/datas/小地图素材/background/monster.png",
        ]

path_buff = [
            default_root +"/datas/小地图素材/background/no_buff_monster.png",
            default_root +"/datas/小地图素材/background/buff.png",
        ]

path_soldier = [
        default_root +"/datas/小地图素材/background/a_soldier.png",
        default_root +"/datas/小地图素材/background/b_soldier.png",
        ]

path_dragon = [default_root +"/datas/小地图素材/background/先锋.png",
            default_root +"/datas/小地图素材/background/先锋_0.png"]


our_tower_xy = {
    "up_1":
        {"cxcywh":[19, 107, 20, 30],
        "paths": path_blue_1,
        },
    "up_2": {"cxcywh":[20, 195, 16, 26],
        "paths": path_blue_23,
        },
    "up_3":{"cxcywh":[23, 259, 16, 26],
        "paths": path_blue_23,
        }, 
    # =========================================
    "mid_1": {"cxcywh":[142, 202, 20, 30],
        "paths":path_blue_1,
        }, 
    "mid_2": {"cxcywh":[117, 243, 16, 26],
        "paths": path_blue_23,
        },
    "mid_3": {"cxcywh":[71, 282, 16, 26],
        "paths": path_blue_23,
        }, 
    # =========================================
    "down_1": {"cxcywh":[261, 326, 20, 30],
        "paths": path_blue_1,
        },
    "down_2": {"cxcywh":[171, 328, 16, 26],
        "paths": path_blue_23,
        },
    "down_3": {"cxcywh":[92, 329, 16, 26],
        "paths": path_blue_23,
        },
    # =========================================
}

enemy_tower_xy = {
    "up_1": {"cxcywh":[113, 17, 20, 30],
        "paths":path_red_1
        },
    "up_2":{"cxcywh":[188, 15, 16, 26],
        "paths":path_red_23
        }, 
    "up_3": {"cxcywh":[266, 15, 16, 26],
        "paths":path_red_23
        }, 
    # =========================================
    "mid_1": {"cxcywh":[217, 142, 20, 30],
        "paths":path_red_1
        },
    "mid_2": {"cxcywh":[242, 101, 16, 26],
        "paths":path_red_23
        },
    "mid_3": {"cxcywh":[289, 64, 16, 26],
        "paths":path_red_23
        },
    # =========================================
    "down_1": {"cxcywh":[338, 254, 20, 30],
        "paths":path_red_1
        },
    "down_2": {"cxcywh":[341, 149, 16, 26],
        "paths":path_red_23
        },
    "down_3": {"cxcywh":[337, 85, 16, 26],
        "paths":path_red_23
        },
    # =========================================
}


monsters = {
    "others":{
        "cxcywh": [[63, 100, 12, 12], [56, 157, 12, 12], [77, 203, 12, 12],[168, 231, 12, 12], [236, 293, 12, 12], [341, 332, 12, 12],
                    [290, 238, 12, 12], [303, 186, 12, 12], [282, 140, 12, 12],[191, 112, 12, 12], [123, 49, 12, 12],
                    [195, 186, 12, 12], [68, 67, 12, 12], [293, 275, 12, 12]],
        "paths":path_monster
    },
    "buff":{
        "cxcywh": [[101, 163, 22, 22], [192, 270, 22, 22], [257, 179, 22, 22], [167, 72, 22, 22]],
        "paths":path_buff
    }
}

soldiers = {
    "up":{
        "wh": [10, 10],
        "dragon_wh": [[22, 22], [30, 30]],
        "dragon_path": path_dragon,
        "road": [[30, 240], [30, 22], [253, 22]],
        "paths": path_soldier
    },
    "mid":{
        "wh": [10, 10],
        "dragon_wh": [[22, 22], [30, 30]],
        "dragon_path": path_dragon,
        "road": [[83, 269], [180, 170], [276, 76]],
        "paths": path_soldier
    },

    "down":{
        "wh": [10, 10],
        "dragon_wh": [[22, 22], [30, 30]],
        "dragon_path": path_dragon,
        "road": [[100, 324], [319, 316], [331, 100]],
        "paths": path_soldier
    },
}

label2num ={
    "bg": 0, 
    "our_tower": 1,
    "enemy_tower": 2,
    "our_hero": 3,
    "enemy_hero":4,
    "my_hero": 5,
}


my_heros = glob.glob(default_root +"/datas/小地图素材/用户英雄/*.png")
our_heros = glob.glob(default_root +"/datas/小地图素材/我方英雄/*.png")
enemy_heros = glob.glob(default_root +"/datas/小地图素材/敌方英雄/*.png")


def vertical_grad(src, color_start, color_end):
    h = src.shape[0]
    # 创建一幅与原图片一样大小的透明图片
    grad_img = np.ndarray(src.shape, dtype=np.uint8)

    g_b = float(color_end[0] - color_start[0]) / h
    g_g = float(color_end[1] - color_start[1]) / h
    g_r = float(color_end[2] - color_start[2]) / h
    for i in range(h):
        for j in range(src.shape[1]):
            grad_img[i,j,0] = color_start[0] + i * g_b
            grad_img[i,j,1] = color_start[1] + i * g_g
            grad_img[i,j,2] = color_start[2] + i * g_r

    return grad_img


def bboxes_iou(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    iou = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
    return iou


def bboxes_diou(boxes1, boxes2, diou=False):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    iou = bboxes_iou(boxes1, boxes2)
    if not diou:
        return iou

    left = np.maximum(boxes1[..., 0], boxes2[..., 0])
    up = np.maximum(boxes1[..., 1], boxes2[..., 1])
    right = np.maximum(boxes1[..., 2], boxes2[..., 2])
    down = np.maximum(boxes1[..., 3], boxes2[..., 3])

    c = (right - left) * (right - left) + (up - down) * (up - down)
    ax = (boxes1[..., 0] + boxes1[..., 2]) / 2
    ay = (boxes1[..., 1] + boxes1[..., 3]) / 2
    bx = (boxes2[..., 0] + boxes2[..., 2]) / 2
    by = (boxes2[..., 1] + boxes2[..., 3]) / 2

    u = (ax - bx) * (ax - bx) + (ay - by) * (ay - by)
    diou_term = u / c
    return iou - diou_term

def nms(bboxes, labels, iou_threshold, sigma=0.3, method='nms', classes_in_img = ["tower", "hero"]):

    best_bboxes = []
    best_labels = []

    cls_mask_hero = [i for i in range(len(labels)) if "hero" in labels[i]]
    cls_bboxes = bboxes[cls_mask_hero]
    cls_labels = [labels[i] for i in cls_mask_hero]

    best_bboxes_tower = []
    best_labels_tower = []
    cls_mask_tower = [i for i in range(len(labels)) if "tower" in labels[i]]
    cls_bboxes_tower = bboxes[cls_mask_tower]
    cls_labels_tower = [labels[i] for i in cls_mask_tower]

    for clz in classes_in_img:
        if clz == "tower":
            for idx, tower in enumerate(cls_bboxes_tower):
                iou = bboxes_diou(tower, cls_bboxes)
                iou_mask = iou > 0.1

                if True not in iou_mask:
                    best_bboxes_tower.append(tower)
                    best_labels_tower.append(cls_labels_tower[idx])
                
        elif clz == "hero":
            len_box = len(cls_bboxes)
            not_accept = []
            while len_box > 0:
                max_ind = len_box - 1
                best_bbox = cls_bboxes[max_ind]

                if max_ind not in not_accept:
                    best_bboxes.append(best_bbox)
                    best_labels.append(cls_labels[max_ind])

                cls_bboxes_tmp = cls_bboxes[: max_ind]

                iou = bboxes_diou(best_bbox[np.newaxis], cls_bboxes_tmp)

                weight = np.ones((len(iou),), dtype=np.float32)
                assert method in ['nms', 'soft-nms']
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes_tmp = cls_bboxes_tmp * weight[:, np.newaxis]
                score_mask = np.sum(cls_bboxes_tmp, axis=-1) > 0.
                not_accept_idx = np.where(score_mask == False)

                if not_accept_idx[0].size > 0:
                    not_accept += list(not_accept_idx[0])

                # cls_bboxes = cls_bboxes[score_mask]
                cls_bboxes = cls_bboxes[:-1]
                len_box = len(cls_bboxes)
        else:
            pass

    best_bboxes = best_bboxes_tower + best_bboxes
    best_labels = best_labels_tower + best_labels    
    
    return best_bboxes, best_labels

def create_maps(output_name, idx=None, iou_threshold=0.25):

    json_dict = {}
    boxes = []
    labels = []

    # for _ in range(num_map):
    if True:

        bg_size = 355 * 3
        bg_material = default_root +"/datas/小地图素材/background/base1.jpg"
        image = cv2.imread(bg_material)

        # =========================================================================
        # 野怪/buff
        # =========================================================================
        for k, vs in monsters.items():
            cxcywhs = vs["cxcywh"]
            paths = vs["paths"]
            for cxcywh in cxcywhs:
                img_path = random.choice(paths)

                y = cxcywh[1] + random.choice([-1, -2, 0, 1, 2])
                x = cxcywh[0] + random.choice([-1, -2, 0, 1, 2])
                w = cxcywh[2] // 2
                h = cxcywh[3] // 2

                if "no_buff_monster" not in img_path:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (cxcywh[-2], cxcywh[-1]))

                    image[y - h: y + h, x - w: x + w] = img
                else:
                    if k == "buff":
                        if random.random() > 0.75:
                            num = random.randint(1, 30)
                            image = cv2.resize(image, (bg_size, bg_size))
                            cv2.putText(image, str(num), (x * 3 - 3, y * 3), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(255, 255, 255), thickness=2)
                            image = cv2.resize(image, (bg_size // 3, bg_size // 3))

        # =========================================================================
        # 兵线
        # =========================================================================
        for k, vs in soldiers.items():
            wh = vs["wh"]
            d_wh = vs["dragon_wh"]
            d_path = vs["dragon_path"]
            road = vs["road"]
            paths = vs["paths"]
            num_s = int(random.random() * 8) + 6
            for i in range(num_s):
                if i < num_s // 2:
                    line = [road[0], road[1]]
                    if k == "up":
                        road_line = line[0][1] - line[1][1]
                        randm = int(random.random() * road_line)
                        new_y = line[0][1] - randm
                        new_x = line[0][0]
                    elif k == "mid":
                        road_line = line[0][1] - line[1][1]
                        randm = int(random.random() * road_line)
                        new_y = line[0][1] - randm
                        new_x = line[0][0] + randm
                    else:
                        road_line = line[1][0] - line[0][0]
                        randm = int(random.random() * road_line)
                        new_y = line[0][1]
                        new_x = line[0][0] + randm
                    
                    img_path = paths[0]
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (wh[-2], wh[-1]))

                    y = new_y + random.choice([-1, -2, 0, 1, 2])
                    x = new_x + random.choice([-1, -2, 0, 1, 2])
                    w = wh[0] // 2
                    h = wh[1] // 2
                    image[y - h: y + h, x - w: x + w] = img

                    if random.random() > 0.95:
                        d_idx = random.choice([0, 1])
                        dimg_path = d_path[d_idx]
                        dimg = cv2.imread(dimg_path)
                        new_wh = d_wh[d_idx]
                        dimg = cv2.resize(dimg, (new_wh[-2], new_wh[-1]))
                    
                        dw = new_wh[0] // 2
                        dh = new_wh[1] // 2
                        image[y - dh: y + dh, x - dw: x + dw] = dimg

                else:
                    line = [road[1], road[2]]
                    if k == "down":
                        road_line = line[0][1] - line[1][1]
                        randm = int(random.random() * road_line)
                        new_y = line[0][1] - randm
                        new_x = line[0][0]
                    elif k == "mid":
                        road_line = line[0][1] - line[1][1]
                        randm = int(random.random() * road_line)
                        new_y = line[0][1] - randm
                        new_x = line[0][0] + randm
                    else:
                        road_line = line[1][0] - line[0][0]
                        randm = int(random.random() * road_line)
                        new_y = line[0][1]
                        new_x = line[0][0] + randm
                    
                    img_path = paths[1]
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (wh[-2], wh[-1]))

                    y = new_y + random.choice([-1, -2, 0, 1, 2])
                    x = new_x + random.choice([-1, -2, 0, 1, 2])
                    w = wh[0] // 2
                    h = wh[1] // 2
                    image[y - h: y + h, x - w: x + w] = img

                    if random.random() > 0.95:
                        d_idx = random.choice([0, 1])
                        dimg_path = d_path[d_idx]
                        dimg = cv2.imread(dimg_path)
                        new_wh = d_wh[d_idx]
                        dimg = cv2.resize(dimg, (new_wh[-2], new_wh[-1]))
                    
                        dw = new_wh[0] // 2
                        dh = new_wh[1] // 2
                        image[y - dh: y + dh, x - dw: x + dw] = dimg

        # =========================================================================
        # 塔
        # =========================================================================
        for k, vs in our_tower_xy.items():
            cxcywh = vs["cxcywh"]
            paths = vs["paths"]
            img_path = random.choice(paths)

            w = cxcywh[2] // 2
            h = cxcywh[3] // 2
            box = [cxcywh[0] - w, cxcywh[1] - h, cxcywh[0] + w, cxcywh[1] + h]

            if "no_tower" not in img_path:
                boxes.append(box)
                labels.append("our_tower")        

            if "full" not in img_path:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (cxcywh[-2], cxcywh[-1]))
                image[box[1]: box[3], box[0]: box[2]] = img


        for k, vs in enemy_tower_xy.items():
            cxcywh = vs["cxcywh"]
            paths = vs["paths"]
            img_path = random.choice(paths)

            w = cxcywh[2] // 2
            h = cxcywh[3] // 2
            box = [cxcywh[0] - w, cxcywh[1] - h, cxcywh[0] + w, cxcywh[1] + h]

            if "no_tower" not in img_path:
                boxes.append(box)
                labels.append("enemy_tower")        

            if "full" not in img_path:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (cxcywh[-2], cxcywh[-1]))
                image[box[1]: box[3], box[0]: box[2]] = img

        ###########################################################################
        # 扩大三倍
        ###########################################################################

        image = cv2.resize(image, (bg_size, bg_size))

        # =========================================================================
        # 倒计时
        # =========================================================================
        if random.random() > 0.75:
            num = random.randint(1, 90)
            cv2.putText(image, str(num), (100 * 3, 110 * 3), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=(214, 112, 218), thickness=8)
        if random.random() > 0.75:
            num = random.randint(1, 90)
            cv2.putText(image, str(num), (220 * 3, 260 * 3), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=3, color=(140, 230, 240), thickness=8)

        # =========================================================================
        # 英雄
        # =========================================================================
        def put_heros(idx, num_hero_list, heros, image, boxes, labels, labels_type, logo_size=300):
            num_hero = random.choice(num_hero_list)
            for i in range(num_hero):
                hero = random.choice(heros)
                if i == 0:
                    if idx is not None:
                        hero = heros[idx % len(heros)]
                hero = cv2.imread(hero, cv2.IMREAD_UNCHANGED)

                xy = (
                    int(logo_size // 2 + random.random() * (image.shape[0] - logo_size)), 
                    int(logo_size // 2 + random.random() * (image.shape[1] - logo_size))
                    )
                box = [xy[0], xy[1], xy[0] + hero.shape[1], xy[1] + hero.shape[0]]

                boxes.append([val // 3 for val in box])
                labels.append(labels_type)

                crop = image[box[1]: box[3], box[0]: box[2]]
                (
                    image[box[1]: box[3], box[0]: box[2]]
                ) = np.where(hero[..., -1:] > 0, hero[..., :3], crop)

            return image, boxes, labels

        def heros(idx, image, boxes, labels, type):
            if type == 0:
                image, boxes, labels = put_heros(idx + 1 if idx is not None else None, 
                                            [1, 2, 3, 4], 
                                            enemy_heros, 
                                            image, 
                                            boxes, 
                                            labels, 
                                            "enemy_hero")
            elif type == 1:
                image, boxes, labels = put_heros(idx + 4 if idx is not None else None, 
                                        [1, 2, 3], 
                                        our_heros, 
                                        image, 
                                        boxes, 
                                        labels, 
                                        "our_hero")
            else:
                image, boxes, labels = put_heros(idx + 7 if idx is not None else None, 
                                            [1, 2, 3], 
                                            my_heros, 
                                            image, 
                                            boxes, 
                                            labels, 
                                            "my_hero")

            return image, boxes, labels

        enemy_first = random.random()
        if enemy_first < 0.25:
            image, boxes, labels = heros(idx, image, boxes, labels, 0)
            image, boxes, labels = heros(idx, image, boxes, labels, 1)
            image, boxes, labels = heros(idx, image, boxes, labels, 2)
        elif enemy_first < 0.5:
            image, boxes, labels = heros(idx, image, boxes, labels, 0)
            image, boxes, labels = heros(idx, image, boxes, labels, 2)
            image, boxes, labels = heros(idx, image, boxes, labels, 1)
        elif enemy_first < 0.75:
            image, boxes, labels = heros(idx, image, boxes, labels, 1)
            image, boxes, labels = heros(idx, image, boxes, labels, 2)
            image, boxes, labels = heros(idx, image, boxes, labels, 0)
        else:
            image, boxes, labels = heros(idx, image, boxes, labels, 1)
            image, boxes, labels = heros(idx, image, boxes, labels, 0)
            image, boxes, labels = heros(idx, image, boxes, labels, 2)

        # resize 回原图
        image = cv2.resize(image, (bg_size // 3, bg_size // 3))

        ###########################################################################
        # 扩大三倍结束
        ###########################################################################

        # =========================================================================
        # 技能/通知等 线条 圆形 框
        # =========================================================================
        colors = [(255, 255, 0), (255, 255, 255), (10, 215, 255), (0, 255, 255), (0, 0, 255), (255, 0, 0)]
        if random.random() > 0.7:
            # line
            num_line =  random.choice([1, 2, 3])
            color_line = random.choice(colors)
            image_copy = image.copy()
            for i in range(num_line):
                start_xy = int(random.random() * (image.shape[0]//2)), int(random.random()* image.shape[1])
                end_xy = int(image.shape[0]//2 + random.random() * (image.shape[0]//2)), int(random.random()* image.shape[1])
                cv2.line(image_copy, start_xy, end_xy, color_line, 2)

            alpha = 0.4 + random.random() * 0.3
            cv2.addWeighted(image_copy, alpha, image, 1 - alpha, 0, image)

        if random.random() > 0.7:
            # rect
            wh = (100, 50)
            lt = int(image.shape[0] // 5 + random.random() * (image.shape[0] * 3 // 5)), int(image.shape[1] // 5 + random.random() * (image.shape[1] * 3 // 5))
            rb = (lt[0] + wh[0], lt[1] + wh[1])

            image_copy = image.copy()
            cv2.rectangle(image_copy, lt, rb, (255, 255, 255), 2)

            alpha = 0.7 + random.random() * 0.3
            cv2.addWeighted(image_copy, alpha, image, 1 - alpha, 0, image)

        if random.random() > 0.7:
            # circle
            num_circle = random.choice([3, 4, 5, 6])
            color_circle = random.choice(colors)
            cxcy = int(image.shape[0] // 5 + random.random() * (image.shape[0] * 3 // 5)), int(image.shape[1] // 5 + random.random() * (image.shape[1] * 3 // 5))
            image_copy = image.copy()
            for i in range(num_circle):
                radius = max(int(random.random() * (image.shape[0] // 5)), image.shape[0] // 30)
                cv2.circle(image_copy, cxcy, radius, color_circle, 2)

            alpha = 0.4 + random.random() * 0.3
            cv2.addWeighted(image_copy, alpha, image, 1 - alpha, 0, image)

        # =========================================================================
        # 渐变背景色
        # =========================================================================
        colors = [(255, 255, 0), (255, 255, 255), (10, 215, 255), 
                (0, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0), 
                (30, 105, 210), (128, 128, 128), (230, 216, 173),
                (0, 128, 128), (128, 0, 128), (203, 192, 255),
                (147, 20, 255), (238, 130, 238), (130, 0, 75),
                ]

        if random.random() > 0.75:
            # line
            color_line = random.choice(colors)
            # color_line = (30, 105, 210)
            # bg = np.ones_like(image) * color_line
            # bg = np.uint8(bg)
            bg = vertical_grad(image, color_line, (0, 0, 0))
            alpha = 0.6 + random.random() * 0.3
            cv2.addWeighted(image, alpha, bg, 1 - alpha, 0, image)

        # =========================================================================
        # NMS / save image
        # =========================================================================
        boxes = np.array(boxes) / [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
        boxes_nms, labels_nms = nms(boxes, labels, iou_threshold)  # todo: iou thresh 0.3 default

        json_dict[output_name] = {
            "boxes": boxes_nms,
            "labels": labels_nms,
        }

        # =========================================================================
        # 缩小再放大，模糊画面
        # =========================================================================
        if random.random() > 0.9:
            sizes = [(256, 256), (224, 224)]
            size_s = random.choice(sizes)
            image = cv2.resize(image, size_s)

        # resize 到最终大小
        image = cv2.resize(image, (320, 320))

    return image, json_dict


if __name__ == "__main__":

    ''' json
    "path to image": {
            "boxes": [
                [
                    boxes_xyxy 0~1
                ]
            ],
            "labels": [
                "bg", "our_tower", "enemy_tower", "our_hero", "enemy_hero"
            ]
        },
    '''

    num_datas = 1
    for _ in range(num_datas):
        output_name = f"{_:08}.jpg"
        image, label = create_maps(output_name, _)

        boxes_nms = label[output_name]["boxes"]
        labels_nms = label[output_name]["labels"]

        # draw images
        for i, b in enumerate(boxes_nms):
            cv2.rectangle(image, 
                          (int(b[0] * image.shape[0]), int(b[1] * image.shape[1])), 
                          (int(b[2] * image.shape[0]), int(b[3]* image.shape[1])), 
                          (255, 255, 255), 
                          2)
            label = label2num[labels_nms[i]]
            cv2.putText(image, 
                        str(label), 
                        (int(b[0] * image.shape[0]), int(b[1] * image.shape[1]) + 20), 
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                        fontScale=1, 
                        color=(255, 255, 255), 
                        thickness=2)

        cv2.imwrite(output_name, image)




