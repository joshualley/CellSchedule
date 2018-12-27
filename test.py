from module.virtual_cell import load_data, CellSchedule
import pandas as pd
import numpy as np

def main():
    transform_time = np.array(pd.read_csv('data/transform_time.csv', header=None))
    #---------------------------------------------------------------------------------#
    ms_process_t, m_per_cls_num = load_data('data/machines.csv', flatten=True)      #
    print('机器加工时间:\n', ms_process_t)                                           #
    #---------------------------------------------------------------------------------#
    parts, part_process_num = load_data('data/parts.csv')
    print('工件:\n', parts)

    #设置ADE参数
    paras = {
        'CR':0.6,
        'Np':100,
        'Gm':2,
        'cell_size': 6,
        'reschedule_time':50,
        'weight': [0.9, 0.1],
        'strategy': 'greedy', # 'greedy'  or  'anneal'
        'cancel_order': [3,7], # 重调度时取消加工的工件（可在1~15任选）
    }
    c = CellSchedule(parts=parts,
                      machine_process_t=ms_process_t,
                      m_cls_num=m_per_cls_num,
                      transform_time=transform_time,
                      paras=paras,
                      name='schedule')
    print('每个工件的工序数:\n', c.process_num)
    print('每道工序的可选机器数:\n', c.spare_machine_num)
    c.schedule(print)

if __name__ == '__main__':
    main()