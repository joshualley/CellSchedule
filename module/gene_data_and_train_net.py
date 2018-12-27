import numpy as np
import pandas as pd
import os
import tensorflow as tf


def gene_machines(machine_class=10, total_num=30):
    """
    假定有10种机器，随机生成机器个数，并每台的加工时间
    格式： m_cls: [wt1,wt2,...],(len=randint(3, 6))
    :param machine_class: 
    :return: 
    """
    machines = []
    while True:
        m_cls_nums = np.random.randint(1,6,size=machine_class)
        if sum(m_cls_nums) == total_num:
            break

    for i in range(machine_class):
        machine_num = m_cls_nums[i]
        base_t = np.random.randint(20,61) * 2
        work_t = []
        for j in range(machine_num):
            for step in range(5):
                prop = np.random.random()
                if prop < 0.5:
                    base_t += 5
                else:
                    base_t -= 5
            work_t.append(base_t)
        machines.append(work_t)

    fn = 'data/machines.csv'
    if not os.path.exists(fn):
        machines = pd.DataFrame(machines)
        machines.to_csv(fn, index=None, header=None)
    else:
        machines = pd.read_csv(fn, header=None)
    machines[pd.isna(machines)] = -1
    nmachines = []
    count = 0
    for row in np.array(machines):
        nrow = []
        for col in row:
            if col != -1.0:
                count += 1
                nrow.append(int(col))
        nmachines.append(nrow)
    print('Machines:\n', nmachines, '\nnumber:', count)
    return nmachines

def gene_parts(fn='data/parts.csv', part_num=30, machines_num=10, process_range=(2, 4)):
    """
    生成工件，格式：part_i：[m1,m2,...](len=process_num)
    :param part_num: 
    :return: 
    """

    def to_vector(xs, deth):
        newx = np.zeros(deth)
        for x in xs:
            if not x == -1:
                newx[int(x)] = 1
        return newx

    parts = []
    #每个工件的工序数暂定2-3
    for i in range(part_num):
        process_num = np.random.randint(process_range[0], process_range[1])
        #print('part: %d,num: %d'%(i, process_num))
        machine_k = set()
        temp = np.random.randint(0, machines_num)
        while len(machine_k) < process_num:
            m_float = temp + np.random.randint(-process_num, process_num)
            if not m_float in range(0, machines_num):
                continue
            machine_k.add(m_float)
        #print('machine:', machine_k)
        parts.append(list(machine_k))

    if not os.path.exists(fn):
        parts_pf = pd.DataFrame(parts)
        parts_pf.to_csv(fn, index=None, header=None)
    else:
        parts_pf = pd.read_csv(fn, header=None)
    parts_pf[pd.isna(parts_pf)] = -1

    parts = np.array(parts_pf)
    nparts = []
    for row in parts:
        nrow = []
        for col in row:
            if not col == -1:
                nrow.append(int(col))
        nparts.append(nrow)

    return nparts

def transform_time(machine_num):
    fn = 'data/transform_time.csv'
    if not os.path.exists(fn):
        distance = np.zeros([machine_num ,machine_num])
        for m1 in range(machine_num):
            for m2 in range(machine_num):
                if m1 != m2:
                    m1_row = int((m1 + 1) / 6)
                    m1_col = int((m1 + 1) % 6)
                    m2_row = int((m2 + 1) / 6)
                    m2_col = int((m2 + 1) % 6)
                    distance[m1, m2] = abs(m2_row - m1_row) + abs(m2_col - m1_col)
        unit_time = 5
        transform_t = distance * unit_time
        transform_t = pd.DataFrame(transform_t)
        transform_t.to_csv(fn, header=None, index=None)
    else:
        transform_t = pd.read_csv(fn, header=None)
    print(transform_t)


def classifer_train(parts, epoches, classes, issave=False, seed=6):
    def net_build(inputs, labels):
        from tensorflow.contrib import slim
        net = slim.flatten(inputs)
        net = slim.fully_connected(net, num_outputs=32, activation_fn=tf.nn.tanh)
        net = slim.fully_connected(net, num_outputs=32, activation_fn=tf.nn.tanh)
        logits = slim.fully_connected(net, num_outputs=classes, activation_fn=tf.nn.softmax)
        loss = -tf.reduce_sum(labels * tf.log(logits))
        optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
        return logits, optimizer

    def feed(parts):
        data = []
        for part in parts:
            d = np.zeros((3, 11))
            for k,m in enumerate(part):
                d[k, m] = 1
            data.append(d)
        data = np.array(data)
        label = np.zeros((classes, classes))
        for i in range(classes):
            label[i, i] = 1
        return {inputs: data, labels: label}

    def parse(results):
        total_cls = [] # 保存每个类别的工件id
        per_nums = [] # 每类中所使用的每类机器的数目
        all_cls_nums = {} # 保存所有类别中，所使用的机器的个数
        print('工件类别如下：')
        for cls_index in range(classes):
            print('第%d类：' % cls_index)
            one_cls = []
            cls_m_num = {} # 类别中所使用的每类机器的数目
            for part_i, part_cls in enumerate(results):
                if part_cls == cls_index:
                    one_cls.append(part_i)
                    print(parts[part_i])
                    for m in parts[part_i]:
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

    tf.set_random_seed(seed)
    nparts = []
    for i in range(len(parts)):
        part = parts[i].copy()
        while len(part) < 3:
            part.append(10)#空缺处用10填补
        nparts.append(part)
    # 3工序11机器（0-9为机器类别，10为空）
    inputs = tf.placeholder(tf.float32, shape=[None, 3, 11], name='inputs')
    # 输出classes个类别
    labels = tf.placeholder(tf.float32, shape=[None, classes], name='labels')
    logits, optimizer = net_build(inputs, labels)
    out = tf.argmax(logits, 1, name='outputs')
    saver = tf.train.Saver()
    if not os.path.exists('model/'):
        os.mkdir('model/')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        feed_dict = feed(nparts[0:2:14])
        for epoch in range(epoches):
            sess.run(optimizer, feed_dict=feed_dict)
            if epoch % 10 == 0:
                o = sess.run(out, feed_dict=feed(nparts))
                print('{0}: {1}'.format(epoch, o))
        if issave:
            saver.save(sess, 'model/model')
        o = sess.run(out, feed_dict=feed(nparts))
    total_cls, per_nums, all_cls_nums = parse(o)

    # 计算每个类别中对一类机器的需求占对此类机器的总需求的比重
    all_machine_weights = []
    for num in per_nums:
        weights = {}
        for k,v in num.items():
            w = v / all_cls_nums[k]
            weights[k] = w
        all_machine_weights.append(weights)
    return all_machine_weights, total_cls

def cac_cls_machine_num():
    machines = []
    machine_nums = {}
    with open('data/machines.csv', 'rb') as f:
        lines = f.readlines()
        for k,line in enumerate(lines):
            line = line.strip().split(b',')
            line = [int(i) for i in line if i != b'']
            machines.append(line)
            machine_nums[k] = len(line)
    return machines, machine_nums

def alloc_machine(weights, machines, machine_nums):
    # 以存有二元组(机器加工时长，机器id)的list：ms来表示机器
    ms = []
    id = 0 # 机器id
    for m_cls in machines:
        l = len(m_cls)
        m = []
        for k in range(l):
            m.append((m_cls[k], id))
            id += 1
        ms.append(m)
    # 计算总需求数
    required_machine_num = {}
    for cls_weight in weights:
        for k, v in cls_weight.items():
            alloc_num = np.ceil(v * machine_nums[k])
            required_machine_num.setdefault(k, 0)
            required_machine_num[k] += int(alloc_num)

    for k, require_num in required_machine_num.items():
        while len(ms[k]) < require_num:
            m = sorted(ms[k])[0]
            ms[k].append(m)

    alloc_machines = []
    for cls_weight in weights:
        alloc_machine = []
        for k, v in cls_weight.items():
            alloc_num = np.ceil(v * machine_nums[k])
            for i in range(int(alloc_num)):
                if ms[k] != []:
                    time, id = ms[k].pop()
                    alloc_machine.append(id)
        alloc_machine = list(set(alloc_machine))
        alloc_machines.append(alloc_machine)

    return alloc_machines

def main():
    #machines = gene_machines(machine_class=10, total_num=30)
    #transform_time(machine_num=30)
    parts = gene_parts(part_num=20, process_range=(2, 4))
    weights, parts_cls = classifer_train(parts, epoches=200, classes=6, seed=100, issave=True)
    print('工件分类：\n', parts_cls)
    machines, machine_nums = cac_cls_machine_num()
    alloc_machines = alloc_machine(weights, machines, machine_nums)
    print('虚拟单元机器分配如下：')
    for cell in alloc_machines:
        print(cell)

def gene_data(mcls, mnum, pnum):
    machines = gene_machines(machine_class=mcls, total_num=mnum)
    transform_time(machine_num=mnum)
    gene_parts(part_num=pnum, process_range=(2, 4))

def train_classifer(classes, seed):
    parts = gene_parts(part_num=20, process_range=(2, 4))
    classifer_train(parts, epoches=200, classes=classes, seed=seed, issave=True)

