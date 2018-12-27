import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import copy
plt.rc('font',family='Times New Roman')

class State():
    M_BUSY = 0
    M_FREE = 1
    M_BREAKDOWN = 6
    P_PENDING = 2
    P_FINASH = 3
    P_TRANSFORM = 4
    P_WORKING = 5

class Machine(object):
    def __init__(self, id, process_time, transform_time, end_time=0):
        self.machine_id = id
        self.transform_time = transform_time
        self.process_time = process_time
        self.end_time = end_time
        self.state = State.M_FREE
        self.repair_t = 0
        self.break_st = []

    def update_state(self, part_obj, global_t):
        if self.state == State.M_BUSY:
            if global_t >= self.end_time:
                self.state = State.M_FREE

        if self.state == State.M_FREE:
            if global_t < self.end_time:
                self.state = State.M_BUSY

        if self.state == State.M_BREAKDOWN:
            self.repair_t -= 1
            if self.repair_t == 0:
                self.state = State.M_BUSY

    def process(self, part_obj, global_t):
        prop_breakdown = np.random.random()
        if prop_breakdown < 0.1:
            self.state = State.M_BREAKDOWN
            break_t = global_t + np.random.randint(0, int(self.process_time))
            #print('%d号机器在%d时刻故障' % (self.machine_id, break_t))
            part_obj.break_st.append([part_obj.current_process, break_t])
            self.repair_t = 30

        part_obj.start_time[part_obj.current_process] = global_t
        self.end_time = global_t + self.process_time + self.repair_t
        part_obj.wait_time[part_obj.current_process] += self.repair_t
        part_obj.end_time[part_obj.current_process] = self.end_time

        # 判断状态转换
        if part_obj.current_process == part_obj.max_process:
            part_obj.finish_t = self.end_time
            part_obj.state = State.P_FINASH
        else:
            mid_current = part_obj.choosed_machines[part_obj.current_process]
            mid_next = part_obj.choosed_machines[part_obj.current_process + 1]
            trans_t = self.transform_time[mid_current, mid_next]
            part_obj.transform_t[part_obj.current_process] = trans_t
            part_obj.start_time[part_obj.current_process + 1] = self.end_time + trans_t
            part_obj.state = State.P_TRANSFORM
            part_obj.current_process += 1

class Part(object):
    def __init__(self, id, choosed_machines, start_time=0):
        self.part_id = id
        self.current_process = 0
        self.max_process = len(choosed_machines) - 1
        self.choosed_machines = choosed_machines
        self.start_time = [0 for i in range(self.max_process+1)]
        self.start_time[self.current_process] = start_time
        self.end_time = [0 for i in range(self.max_process+1)]
        self.state = State.P_PENDING
        self.wait_time = [0 for i in range(self.max_process+1)]
        self.transform_t = [0 for i in range(self.max_process+1)]
        self.finish_t = 0
        self.break_st = []

    def run(self, global_t, machine_obj_list):
        #首先更新机器状态
        mid = self.choosed_machines[self.current_process]
        machine_obj_list[mid].update_state(self, global_t)
        #print("gt:%d, ft:%d" %(global_t, self.finish_t))
        if self.state == State.P_FINASH:
            if global_t >= self.finish_t:
                #print("Finish")
                self.end_time[self.current_process] = self.finish_t
                return False

        if self.state == State.P_PENDING:
            if machine_obj_list[mid].state == State.M_BUSY:
                self.wait_time[self.current_process] += 1
            if machine_obj_list[mid].state == State.M_FREE:
                self.state = State.P_WORKING

        if self.state == State.P_WORKING:
            #加工处理
            if machine_obj_list[mid].state == State.M_FREE:
                machine_obj_list[mid].process(part_obj=self, global_t=global_t)

        if self.state == State.P_TRANSFORM:
            if self.start_time[self.current_process-1] <= self.start_time[self.current_process] \
                    and global_t >= self.start_time[self.current_process]:
                self.state = State.P_PENDING

        return True

class ADE(object):

    def __init__(self, paras, reschedule=False, data={}):
        self.name = data['name']
        self.reschedule = reschedule
        self.per_gene_part_obj_set = []
        self.per_gene_individuals = []
        self.scale_factor_1 = 0.8
        self.scale_factor_2 = 0.8
        self.T = 90  # 起始温度
        self.T_end = 88
        self.temperature_factor = 0.9999  # 降温因子
        self.G = 0
        self.CR = paras['CR']
        self.Np = paras['Np']
        self.Gm = paras['Gm']
        self.lamda = paras['weight'] #目标函数权重因子
        self.reschedule_t = paras['reschedule_time']
        self.strategy = paras['strategy']
        self.fits = []
        self.all_gene_fits = []
        self.minfits = []
        self.avgfits = []

        self.canceled_order = paras['cancel_order']
        self.cell = data['cell']
        #self.cells_m_in_cls = data['cells_m_in_cls']
        #self.part_cell = data['part_cell']
        self.total_process_num = sum(data['parts_process_num'])
        self.machine_num = data['machine_num']
        self.machine_process_t = data['machine_process_t']
        self.transform_time = data['transform_time']

        self.process_spare_machine_num = data['spare_m_num']
        self.parts_process_num = data['parts_process_num']
        self.parts = data['parts_cls']
        #self.parts_cls_id = data['parts_cls_id']
        self.part_objs = []
        self.saved_objs = []

        if reschedule:
            self.parts_id = data['parts_id']
            self.part_num = len(data['parts_process_num'])
        else:
            self.parts_id = [i for i in range(15)]
            self.part_num = 15


        self.travel_t = np.zeros(self.Np)
        self.wait_t = np.zeros(self.Np)
        self.final_t = np.zeros(self.Np)

        self.X = None
        self.X_next_1 = None
        self.X_next_2 = None
        self.X_next = None

        self.Y = None
        self.Y_next_1 = None
        self.Y_next_2 = None
        self.Y_next = None

    def init_population(self):
        self.X = np.random.random((self.Np, self.part_num))
        self.X_next_1 = np.zeros((self.Np, self.part_num))
        self.X_next_2 = np.zeros((self.Np, self.part_num))
        self.X_next = np.zeros((self.Np, self.part_num))

        self.Y = []
        for k in range(self.Np):
            y = []
            for i in range(self.part_num):
                temp = list(np.random.random((1, self.parts_process_num[i])))
                y.append(temp)
            self.Y.append(y)
        self.Y = np.array(self.Y)
        self.Y_next_1 = np.array(self.Y)
        self.Y_next_2 = np.array(self.Y)
        self.Y_next = np.array(self.Y)

    def variation(self):
        operator = np.exp(1 - self.Gm / (self.Gm + 1 - self.G))
        F1 = self.scale_factor_1 * (2 ** operator)
        F2 = self.scale_factor_2 * (2 ** operator)
        for i in range(self.Np):
            j, k, p = 0, 0, 0
            while i == j or i == k or i == p or j == k or j == p or k == p:
                j, k, p = np.random.randint(0, self.Np, 3)
            self.X_next_1[i, :] = self.X[p, :] + F1 * (self.X[j, :] - self.X[k, :])
        self.X_next_1 = np.array([[np.random.random() if i > 1 or i < 0 else i for i in row] for row in self.X_next_1])
        # print(self.X_next_1.shape)

        for i in range(self.Np):
            j, k, p = 0, 0, 0
            while i == j or i == k or i == p or j == k or j == p or k == p:
                j, k, p = np.random.randint(0, self.Np, 3)
            for d in range(self.part_num):
                # print(self.Y[i,d,0])
                self.Y_next_1[i, d, 0] = self.Y[p, d, 0] + F2 * (self.Y[j, d, 0] - self.Y[k, d, 0])
                self.Y_next_1[i, d, 0] = np.array(
                    [np.random.random() if i > 1 or i < 0 else i for i in self.Y_next_1[i, d, 0]])

    def cross(self):
        for i in range(self.Np):
            for j in range(self.part_num):
                if np.random.random() < self.CR or np.random.randint(0, self.part_num) != j:
                    self.X_next_2[i, j] = self.X_next_1[i, j]
                else:
                    self.X_next_2[i, j] = self.X[i, j]

        for i in range(self.Np):
            for j in range(self.part_num):
                if np.random.random() < self.CR or np.random.randint(0, self.part_num) != j:
                    self.Y_next_2[i, j, 0] = self.Y_next_1[i, j, 0]
                else:
                    self.Y_next_2[i, j, 0] = self.Y[i, j, 0]

    def choose(self):
        for i in range(self.Np):
            ft1, wt1, tt1 = self.fitness(self.X[i, :], self.Y[i, :, 0])
            ft2, wt2, tt2 = self.fitness(self.X_next_2[i, :], self.Y_next_2[i, :, 0])
            fitness1 = self.lamda[0] * ft1 + self.lamda[1] * wt1 + (1 - self.lamda[0] - self.lamda[1]) * tt1
            fitness2 = self.lamda[0] * ft2 + self.lamda[1] * wt1 + (1 - self.lamda[0] - self.lamda[1]) * tt2
            #fitness1 = ft1
            #fitness2 = ft2
            if fitness2 < fitness1:
                self.X_next[i, :] = self.X_next_2[i, :]
                self.Y_next[i, :, 0] = self.Y_next_2[i, :, 0]
                self.final_t[i] = ft2
                self.wait_t[i] = wt2
                self.travel_t[i] = tt2
            else:
                if self.strategy == "anneal":
                    prop = np.random.random()
                    # 模拟退火优化策略
                    #prop_anneal = 1/(1 + np.exp((fitness2 - fitness1) / self.T))
                    prop_anneal = np.exp(-(fitness2 - fitness1) / self.T)
                    #print('prop:%2.2f, anneal:%2.2f' % (prop, prop_anneal))
                    if prop < prop_anneal:
                        #print('退火优化，prop:%2.2f, anneal:%2.2f' %(prop, prop_anneal))
                        self.X_next[i, :] = self.X_next_2[i, :]
                        self.Y_next[i, :, 0] = self.Y_next_2[i, :, 0]
                        self.final_t[i] = ft2
                        self.wait_t[i] = wt2
                        self.travel_t[i] = tt2
                    else:
                        self.X_next[i, :] = self.X[i, :]
                        self.Y_next[i, :, 0] = self.Y[i, :, 0]
                        self.final_t[i] = ft1
                        self.wait_t[i] = wt1
                        self.travel_t[i] = tt1

                if self.strategy == "greedy":
                    self.X_next[i, :] = self.X[i, :]
                    self.Y_next[i, :, 0] = self.Y[i, :, 0]
                    self.final_t[i] = ft1
                    self.wait_t[i] = wt1
                    self.travel_t[i] = tt1

    def fitness(self, X, Y):
        P, M = self.decode(X, Y)
        ft = 0
        wt = 0
        tt = 0
        machine_list = [Machine(i, t, self.transform_time) for i, t in enumerate(self.machine_process_t)]

        if self.reschedule:
            machine_list = copy.deepcopy(self.saved_objs[1])
            global_t = self.reschedule_t
            part_objs = []
            for id in self.parts_id:
                if id < 15:
                    p = self.saved_objs[0][id]
                    ms = p.choosed_machines
                    for i in range(p.max_process+1):
                        if i > p.current_process:

                            p.choosed_machines[i] = M[id][i]
                    for process_j in range(p.max_process):
                        mid = p.choosed_machines[process_j]
                        next_mid = p.choosed_machines[process_j + 1]
                        trans_t = self.transform_time[mid, next_mid]
                        if next_mid != ms[process_j+1]:
                            tt_already = global_t - p.end_time[process_j-1]
                        else:
                            tt_already = 0
                        p.transform_t[process_j] = trans_t + 2 * tt_already
                else:
                    p = Part(id=id, choosed_machines=M[id], start_time=self.reschedule_t)
                part_objs.append(copy.deepcopy(p))

        else:
            part_objs = [Part(id=id, choosed_machines=M[i]) for i, id in enumerate(self.parts_id)]
            global_t = 0

        while True:
            flag = [True for i in range(len(part_objs))]
            for i in P:
                if self.reschedule:
                    # 退订订单
                    if i in self.canceled_order:
                        #print("%d 号工件退订" % i)
                        flag[int(i)] = False
                        continue
                p = part_objs[int(i)]
                if flag[int(i)]:
                    flag[int(i)] = p.run(global_t, machine_list)

            if not self.reschedule and global_t == int(self.reschedule_t):
                self.saved_objs = []
                self.saved_objs.append(copy.deepcopy(part_objs))
                self.saved_objs.append(copy.deepcopy(machine_list))

            if not True in flag:
                break

            global_t += 1

        self.per_gene_part_obj_set.append(copy.deepcopy(part_objs))

        for k,p in enumerate(part_objs):
            #print('P', k,'st:', p.start_time, 'wt:', p.wait_time, 'tt:', p.transform_t, 'ft:', p.finish_t)
            if self.reschedule and p.part_id in self.canceled_order:
                continue
            ft = max(ft, p.finish_t)
            wt += sum(p.wait_time)
            tt += sum(p.transform_t)

        #print('ft:%d, wt:%d, tt:%d' %(ft, wt, tt))
        return ft, wt, tt

    def decode(self, X_, Y_):

        X, Y = X_.copy(), Y_.copy()
        x_index = np.argsort(X)
        A = [i for i in range(self.part_num)]

        for i in range(self.part_num):
            X[x_index[i]] = A[i]
        #print('X:', X)

        for i in range(self.part_num):
            Y[i] = np.ceil(Y[i] * self.process_spare_machine_num[i])

        Y = [list(i) for i in Y]
        for part_i in range(self.part_num):
            for process_j in range(self.parts_process_num[part_i]):
                Y[part_i][process_j] = self.parts[part_i][process_j][int(Y[part_i][process_j]) - 1]

        return X, Y

    def format_print(self, part_objs):
        process_num = 0

        for p in part_objs:
            if not self.reschedule:
                if p.part_id < 15:
                    process_num += p.max_process + 1
            else:
                if not p.part_id+1 in self.canceled_order:
                    process_num += p.max_process + 1

        result = np.zeros((process_num, 8))
        counter = 0
        for i in range(self.part_num):
            if self.reschedule:
                #退订订单
                if i in self.canceled_order:
                    print("%d 号工件退订" %(i))
                    continue
            part_i = part_objs[int(i)]
            for process_j in range(part_i.max_process + 1):
                mid = part_i.choosed_machines[process_j]
                st = part_i.start_time[process_j]
                et = st + self.machine_process_t[mid]
                pt = self.machine_process_t[mid]
                wt = part_i.wait_time[process_j]
                tt = part_i.transform_t[process_j]

                result[counter, 0] = i
                result[counter, 1] = process_j + 1
                result[counter, 2] = mid
                result[counter, 3] = st
                item = None
                for bt in part_i.break_st:
                    item = bt
                    if bt[0] == process_j:
                        result[counter, 4] = et + 30
                    else:
                        result[counter, 4] = et
                if not item:
                    result[counter, 4] = et
                result[counter, 5] = pt
                result[counter, 6] = wt
                result[counter, 7] = tt
                counter += 1

        pf = pd.DataFrame(result)
        pf.columns = ['工件号', '工序号', '机器编号', '起始时间',
                      '结束时间', '加工时间', '等待时间', '运输时间']
        print('调度结果：\n', pf)
        if not os.path.exists('result/'):
            os.mkdir('result')
        fn = 'result/'+ self.name +'.csv'
        pf.to_csv(fn)
        if not self.reschedule:
            fp = pf.where(pf['起始时间'] <= self.reschedule_t).dropna()
            print("已完工工序：\n", fp)
            fn = 'result/' + 'finish_process' + '.csv'
            fp.to_csv(fn)

        p_ft = []
        for p in self.part_objs:
            p_ft.append(p.finish_t)
        tft = max(p_ft)
        twt = sum(result[:, 6])
        ttt = sum(result[:, 7])
        print('完工时间：%d, 总等待时间：%d，总运输时间：%d' % (tft, twt, ttt))

        return tft

    def gant_chart(self):
        best_g = self.minfits.index(min(self.minfits))
        print('最优个体出现在第%d代，适应度为：%2.2f' %(best_g, min(self.all_gene_fits[best_g])))
        best_i = 2 * self.all_gene_fits[best_g].index(min(self.all_gene_fits[best_g])) + 1

        self.part_objs = self.per_gene_individuals[best_g][best_i]

        tft = self.format_print(self.part_objs)

        #虚拟单元调度图
        f1 = plt.figure()

        plt.xlabel('processing time', labelpad=20)
        #plt.ylabel('Machine', labelpad=20)
        plt.yticks([])
        plt.xticks([])
        plt.box(False)
        plt.margins(0)
        if self.reschedule:
            plt.title('Cell Reschedule')
        else:
            plt.title('Cell Schedule')
        cell_len = [len(c) for c in self.cell]
        max_len = max(cell_len)
        for cell_k, mids in enumerate(self.cell):
            s = "32" + str(cell_k + 1)
            ax = f1.add_subplot(int(s))
            s = "Cell-" + str(cell_k+1)
            plt.ylabel(s)
            h_cell = int(tft*2 / max_len)
            for i,mid in enumerate(mids):
                for part in self.part_objs:
                    if not self.reschedule:
                        if part.part_id >= 15:
                            continue
                    else:
                        # 退订工件订单
                        if part.part_id in self.canceled_order:
                            continue

                    for j,m in enumerate(part.choosed_machines):
                        if m == mid:
                            st = part.start_time[j]
                            et = part.end_time[j]
                            tpt = et - st
                            ax.broken_barh([(st, tpt)], [h_cell * i, h_cell-10], facecolors='w', edgecolors='b')
                            for bt in part.break_st:
                                if bt[0] == j:
                                    ax.broken_barh([(bt[1], 30)], [h_cell * i, h_cell-10], facecolors='r')
                                    #plt.text(bt[1] + 14, h_cell * i + h_cell / 2 - 20, 'E', fontsize=6)

                            #print("工件%d的工序%d加工时间：%d, 故障时间：%d" %(part.part_id, j,tpt, tpt-pt))
                            tx = str(part.part_id) + '-' + str(j + 1)
                            fs = 8
                            plt.text(st+tpt/2-len(tx)*fs/2, h_cell*i+h_cell/4, tx, fontsize=fs)


            ax.set_yticks(range(int(h_cell / 2), h_cell * max_len, h_cell))
            m_len = len(mids)
            #ax.set_yticks(range(m_len, h_cell * max_len, h_cell))
            #mids = [id for id in mids]
            ax.set_yticklabels(mids)

        if self.reschedule:
            plt.savefig('result/cell_reschedule.png')
        else:
            plt.savefig('result/cell_schedule.png')

        #总的调度图
        h = int(3*tft / (4*self.machine_num))
        f2 = plt.figure()
        ax = f2.add_subplot(111, aspect='equal')
        for part in self.part_objs:
            if not self.reschedule:
                if part.part_id >= 15:
                    continue
            else:
                #退订工件订单
                if part.part_id in self.canceled_order:
                    continue

            for process_j in range(part.max_process+1):
                mid = part.choosed_machines[process_j]
                st = part.start_time[process_j]
                et = part.end_time[process_j]
                pt = et - st
                #ax.broken_barh([(st, pt)], [h * mid, h], facecolors=colors[part.part_id])
                currentAxis = plt.gca()
                rect = patches.Rectangle((st, h*mid), pt, h-5, linewidth=1, edgecolor='b', facecolor='none')
                currentAxis.add_patch(rect)
                for bt in part.break_st:
                    if bt[0] == process_j:
                        print("机器%d在%d时刻加工工件%d的第%d道工序时发生故障"
                              %(part.choosed_machines[process_j]+1, bt[1], part.part_id, process_j+1))
                        ax.broken_barh([(bt[1], 30)], [h * mid, h-5], facecolors='r')
                        #plt.text(bt[1]+14, h * mid + h / 2 - 4, 'E', fontsize=8)
                tx = str(part.part_id) + '-' + str(process_j+1)
                fs = 8
                plt.text(st+pt/2-len(tx)*fs/2, h * mid +(h-fs)/2, tx, fontsize=fs)
            plt.xlabel('processing time')
            plt.ylabel('Machine')
            ax.set_yticks(range(int(h / 2), h * (self.machine_num + 1), h))
            ax.set_yticklabels(range(0, self.machine_num))
            plt.tight_layout(2)

        if self.reschedule:
            plt.title('Reschedule')
            plt.savefig('result/total_reschedule.png')
            plt.show()
        else:
            plt.title('Schedule')
            plt.savefig('result/total_schedule.png')

    def run(self):
        self.init_population()
        st = time.time()
        fts = []

        plt.ion()
        f = plt.figure()
        if self.reschedule:
            plt.title('Analysis of Rescheduling Astringency')
            print('重调度开始：')
        else:
            plt.title('Analysis of Scheduling Astringency')
            print('初调度开始：')

        while self.G < self.Gm and self.T > self.T_end:
            self.variation()
            self.cross()
            self.choose()
            self.X = self.X_next
            self.Y = self.Y_next

            time_step = 1
            temp = time.time() - st
            l_m = int(temp * self.Gm - temp * self.G) / 60
            l_s = int(temp * self.Gm - temp * self.G) % 60
            self.fits = list(map(lambda ft, wt, tt: self.lamda[0]*ft+self.lamda[1]*wt+(1-self.lamda[0]-self.lamda[1])*tt,
                                 self.final_t, self.wait_t, self.travel_t))
            fitness = min(self.fits)
            individuals_i = self.fits.index(fitness)

            ft = self.final_t[individuals_i]
            wt = self.wait_t[individuals_i]
            tt = self.travel_t[individuals_i]

            self.all_gene_fits.append(self.fits)

            self.minfits.append(fitness)
            self.avgfits.append(np.mean(self.fits))
            fts.append(ft)
            plt.plot(self.avgfits, '-r', linewidth=0.5)
            plt.plot(self.minfits, '-b', linewidth=0.5)
            plt.legend(['AvgFit', 'MinFit'], loc='upper right')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.pause(0.0001)
            print('[Name=%s,G=%d,T=%2.2fK] FT:%dmin, WT:%dmin, TT:%dmin, Fit:%2.2f, Time:%2.2fs, ETA:%dm %ds'
                  % (self.name, self.G, self.T, ft, wt, tt, fitness, temp * time_step, l_m, l_s))
            st = time.time()
            self.G += 1
            self.T *= self.temperature_factor
            self.per_gene_individuals.append(self.per_gene_part_obj_set)
            self.per_gene_part_obj_set = []

        plt.ioff()
        if self.reschedule:
            plt.savefig('result/reschedule_astringency.png')
        else:
            plt.savefig('result/schedule_astringency.png')
        self.gant_chart()

