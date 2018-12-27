from module.gene_data_and_train_net import cac_cls_machine_num, alloc_machine, writeAndPrintLog
from module.ade_fsm import ADE
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

def predict(parts, classes=6):
    def feed(parts):
        data = []
        for id, part in parts:
            d = np.zeros((3, 11))
            for k, m in enumerate(part):
                d[k, m] = 1
            data.append(d)
        data = np.array(data)
        label = np.zeros((classes, classes))
        for i in range(classes):
            label[i, i] = 1
        return {'inputs:0': data, 'labels:0': label}

    def parse(results):
        total_cls = [] # 保存每个类别的工件id
        per_nums = [] # 每类中所使用的每类机器的数目
        all_cls_nums = {} # 保存所有类别中，所使用的机器的个数
        #print('工件类别如下：')
        for cls_index in range(classes):
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

    def cell_alloc(self, parts, classes):
        weights, part_cell = predict(parts, classes)
        writeAndPrintLog('工件分类:\n{}'.format(part_cell), self.dispfunc)
        machines, machine_nums = cac_cls_machine_num()
        cells = alloc_machine(weights, machines, machine_nums)
        ncells = []
        for cid, machines in enumerate(cells):
            ncell = [[] for i in range(10)]
            for mid in machines:
                for mcls in range(10):
                    if mid in self.m_cls2ids(mcls):
                        ncell[mcls].append(mid)
            ncells.append(ncell)

        writeAndPrintLog('虚拟单元：\n{}'.format(cells), self.dispfunc)
        return part_cell, cells, ncells

    def regene_parts_cls_in_cell(self, parts_cls, part_cell, cells):
        def map_part_with_cell(pid):
            # 由工件id获取其单元id
            for cid, cell in enumerate(part_cell):
                if pid in cell:
                    return cid

        parts_cls_in_cell = []
        for pid, part in enumerate(parts_cls):
            process_j = []
            for j, process in enumerate(part):
                ms = []
                for machine in process:
                    if map_part_with_cell(pid) != None and machine in cells[map_part_with_cell(pid)]:
                        ms.append(machine)
                        #print('part-{},process-{},machine-{}=>{}'.format(pid, j, machine, cells[map_part_with_cell(pid)]))
                    else:
                        #print('part-{},process-{},machine-{}'.format(pid, j, machine))
                        pass
                    if map_part_with_cell(pid) == None:
                        #print(pid, len(process))
                        pass
                if ms == []: ms.append(-1)
                process_j.append(ms)
            parts_cls_in_cell.append(process_j)

        writeAndPrintLog('工件在单元中可选机器id:\n{}'.format(parts_cls_in_cell), self.dispfunc)
        return parts_cls_in_cell

    def schedule(self, dispfunc=None):
        self.dispfunc = dispfunc
        def cac_machine_num_in_cell(pid, part, reschedule=False):
            if reschedule and pid in self.paras['cancel_order']:
                num = self.spare_machine_num[pid]
                num = [0 for i in num]
                return num

            cid = 100
            for id, cell in enumerate(part_cell):
                if pid in cell:
                    cid = id
                    break
            part_spare_machine_num_in_cell = []
            for mcls in part:
                num = len(cells_m_in_cls[cid][mcls])
                part_spare_machine_num_in_cell.append(num)
            return part_spare_machine_num_in_cell

        writeAndPrintLog('初调度分配虚拟单元：', dispfunc)
        writeAndPrintLog('------------------------------------------------------------------------------------',
                         dispfunc)
        parts = [(id, part) for id, part in enumerate(self.parts[:15])]
        part_cell, cells, cells_m_in_cls = self.cell_alloc(parts, classes=self.paras['cell_size'])
        spare_machine_num_in_cells = []
        [spare_machine_num_in_cells.append(cac_machine_num_in_cell(pid, part)) for pid, part in parts]
        writeAndPrintLog('工件在各单元中可选机器数量：\n{}'.format(spare_machine_num_in_cells), dispfunc)
        parts_cls = self.regene_parts_cls_in_cell(self.parts_cls[:15], part_cell, cells)
        writeAndPrintLog('------------------------------------------------------------------------------------',
                         dispfunc)

        data = {
            'cell': cells,
            'cells_m_in_cls': cells_m_in_cls,
            'parts_process_num': self.process_num,
            'machine_num': len(self.machine_process_t),
            'spare_m_num': spare_machine_num_in_cells,
            'parts_cls': parts_cls, #包含了每道工序的待选机器id信息
            'parts_id': self.parts_id, #所有工件的id
            'name': self.name,
            'part_cell': part_cell,
            'machine_process_t': self.machine_process_t,
            'transform_time': self.transform_time,
            'dispfunc': self.dispfunc
        }
        ade = ADE(paras=self.paras, data=data)
        ade.run()
        writeAndPrintLog('重调度分配虚拟单元：', dispfunc)
        writeAndPrintLog('------------------------------------------------------------------------------------',
                         dispfunc)
        parts = [(id, part) for id, part in enumerate(self.parts) if id not in self.paras['cancel_order']]
        part_cell, cells, cells_m_in_cls = self.cell_alloc(parts, classes=self.paras['cell_size'])
        spare_machine_num_in_cells = []
        [spare_machine_num_in_cells.append(cac_machine_num_in_cell(pid, part, reschedule=True)) for pid, part in enumerate(self.parts)]
        writeAndPrintLog('工件在各单元中可选机器：\n{}'.format(spare_machine_num_in_cells), dispfunc)
        parts_cls = self.regene_parts_cls_in_cell(self.parts_cls, part_cell, cells)
        writeAndPrintLog('------------------------------------------------------------------------------------',
                         dispfunc)
        data = {
            'cell': cells,
            'cells_m_in_cls': cells_m_in_cls,
            'parts_process_num': self.process_num,
            'machine_num': len(self.machine_process_t),
            'spare_m_num': spare_machine_num_in_cells,
            'parts_cls': parts_cls,  # 包含了每道工序的待选机器id信息
            'parts_id': self.parts_id,  # 所有工件的id
            'name': 'reschedule',
            'part_cell': part_cell,
            'machine_process_t': self.machine_process_t,
            'transform_time': self.transform_time,
            'dispfunc': self.dispfunc
        }
        ade_r = ADE(paras=self.paras, reschedule=True, data=data)
        ade_r.saved_objs = ade.saved_objs
        ade_r.run()
