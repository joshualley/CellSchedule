from cell_schedule.gene_data_and_train_net import cac_cls_machine_num, alloc_machine
from cell_schedule.ade_fsm import ADE
import tensorflow as tf
import numpy as np
import pandas as pd

def load_data(fn, flatten=False):
    data = pd.read_csv(fn, header=None)
    data[pd.isna(data)] = -1
    newdata = []
    nums = []
    for row in np.array(data):
        newrow = []
        counter = 0
        for col in row:
            if col != -1.0:
                counter += 1
                if not flatten:
                    newrow.append(int(col))
                else:
                    newdata.append(int(col))
        nums.append(counter)
        if not flatten:
            newdata.append(newrow)

    return newdata, nums

def predict(parts):
    def feed(parts):
        data = []
        for id, part in parts:
            d = np.zeros((3, 11))
            for k, m in enumerate(part):
                d[k, m] = 1
            data.append(d)
        data = np.array(data)
        label = np.zeros((6, 6))
        for i in range(6):
            label[i, i] = 1
        return {'inputs:0': data, 'labels:0': label}

    def parse(results):
        total_cls = [] # 保存每个类别的工件id
        per_nums = [] # 每类中所使用的每类机器的数目
        all_cls_nums = {} # 保存所有类别中，所使用的机器的个数
        #print('工件类别如下：')
        for cls_index in range(6):
            #print('第%d类：' % cls_index)
            one_cls = []
            cls_m_num = {} # 类别中所使用的每类机器的数目
            for i, part_cls in enumerate(results):
                if part_cls == cls_index:
                    one_cls.append(parts[i][0])
                    #print(parts[part_i])
                    for m in parts[i][1]:
                        all_cls_nums.setdefault(m, 0)
                        all_cls_nums[m] += 1
                        if m in cls_m_num.keys():
                            cls_m_num[m] += 1
                        else:
                            cls_m_num[m] = 1
            #print(cls_m_num)
            per_nums.append(cls_m_num)
            total_cls.append(one_cls)
        return total_cls, per_nums, all_cls_nums

    sess = tf.Session()
    saver = tf.train.import_meta_graph('model/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint('model/'))
    o = sess.run('outputs:0', feed_dict=feed(parts))

    total_cls, per_nums, all_cls_nums = parse(o)

    # 计算每个类别中对一类机器的需求占对此类机器的总需求的比重
    all_machine_weights = []
    for num in per_nums:
        weights = {}
        for k, v in num.items():
            w = v / all_cls_nums[k]
            weights[k] = w
        all_machine_weights.append(weights)
    return all_machine_weights, total_cls

class CellSchedule():

    def __init__(self, parts, machine_process_t, m_cls_num, transform_time, paras, name):
        self.name = name
        self.paras = paras
        self.m_cls_num = m_cls_num  # 记录每类机器个数
        self.parts = parts
        self.parts_id = [i for i in range(len(parts))] #每类中的工件的id
        self.parts_cls = self.id2part()
        self.process_num = self.cac_process_num()
        self.spare_machine_num = self.cac_spare_machine_num()
        #self.cell = cells
        self.machine_process_t = machine_process_t
        self.transform_time = transform_time

    def m_cls2ids(self, m_cls):
        """
        将机器的类别根据其数量转为id列表
        :param m_cls: 机器种类
        :return:
        """
        #print('每类机器个数:', m_cls_num)
        m_num = self.m_cls_num[m_cls]
        prefix = sum(self.m_cls_num[0:m_cls])
        ids = []
        for i in range(m_num):
            id = i + prefix
            ids.append(id)
        return ids

    def id2part(self):
        p_cls = []
        #print(self.parts_id)
        for i in self.parts_id:
            ms = []
            for m in self.parts[i]:
                ms.append(self.m_cls2ids(m))
            p_cls.append(ms)
        #print('part single class:', p_cls)

        return p_cls

    def cac_process_num(self):
        """
        计算工件的工序数
        :return:
        """
        nums = []
        for part in self.parts_cls:
            nums.append(len(part))

        return nums

    def cac_spare_machine_num(self):
        """
        计算每个工件的工序的候选机器数目
        :return:
        """
        spare_nums = []
        for part in self.parts_cls:
            temp = []
            for process in part:
                temp.append(len(process))
            spare_nums.append(temp)

        return spare_nums

    def cell_alloc(self, parts):
        weights, parts_cls = predict(parts)
        print('工件分类:\n', parts_cls)
        machines, machine_nums = cac_cls_machine_num()
        cells = alloc_machine(weights, machines, machine_nums)
        print('虚拟单元：\n', cells)
        return parts_cls, cells

    def schedule(self):
        print('初调度分配虚拟单元：')
        print('----------------------------------------------------------------------------------------------------')
        parts = [(id, part) for id, part in enumerate(self.parts[:15])]
        part_cls, cells = self.cell_alloc(parts)
        print('----------------------------------------------------------------------------------------------------')
        data = {
            'cell': cells,
            'parts_process_num': self.process_num,
            'machine_num': len(self.machine_process_t),
            'spare_m_num': self.spare_machine_num,
            'parts_cls': self.parts_cls,
            'parts_id': self.parts_id,
            'name': self.name,
            'machine_process_t': self.machine_process_t,
            'transform_time': self.transform_time,
        }
        ade = ADE(paras=self.paras, data=data)
        ade.run()
        print('重调度分配虚拟单元：')
        print('----------------------------------------------------------------------------------------------------')
        parts = [(id, part) for id, part in enumerate(self.parts) if id not in self.paras['cancel_order']]
        part_cls, cells = self.cell_alloc(parts)
        print('----------------------------------------------------------------------------------------------------')
        data['name'] = 'reschedule'
        data['cell'] = cells
        ade_r = ADE(paras=self.paras, reschedule=True, data=data)
        ade_r.saved_objs = ade.saved_objs
        ade_r.run()
