import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from utils import ToGraphData

warnings.filterwarnings('ignore')


# TODO
# 1. 考虑动态IP与静态IP问题
# 2. 考虑登录时间段问题
# 3. 考虑增强节点属性在编码中的影响力问题
# 4. 员工操作内容纳入图节点


def clean_system(data):
    # 添加系统名称
    sys_num_name_mapper = {'XTXN': '分布式消费信贷核心系统（生产+ 测试）',
                           'CLS': '操作平台',
                           'UBO': '统一业务运营管理平台（生产+测试）',
                           'ubo-ubomp-ubomp': '统一业务运营管理平台（生产+测试)',
                           'IAM': '统一认证',
                           'BRS': '般若',
                           'GDS': '综合数据查询系统',
                           'FTQ': '发欺诈天启',
                           'BDC': '资信平台',
                           'CMP': '综合资金管理系统（生产+ 测试）',
                           'XMS': '玄明',
                           'ddms': '电子档案管理子系统'}

    data['sys_name'] = data.sys_num.map(lambda x: sys_num_name_mapper[x] if x in sys_num_name_mapper else x)
    return data


def clean_staff(data):
    # 内部员工类型
    staff_char_name_mapper = {21: '人力外包',
                              24: '暑期实习生',
                              25: '其他',
                              22: '项目外包',
                              2: '派遣员工',
                              1: '行编',
                              23: '校招实习生'}
    data['emply_char_name'] = data.employment_char.map(lambda x: staff_char_name_mapper[int(x)] if pd.notna(x) else x)
    return data


def clean_operation_time(data):
    def is_work_hour(date_time):
        if 8 < date_time.hour < 22:
            return 1
        else:
            return 0

    # 将操作时间转化为datetime类型
    data['oper_time'] = data.oper_tm.map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    data['is_work_hour'] = data.oper_time.map(is_work_hour)
    return data


def clean_behaviors(data):
    # 提取主要操作行为类型
    def ext_main_behavior(x):
        behavior_pattern = re.compile('query|update|login|add|delete|download', flags=re.I)
        if not isinstance(x, (str, bytes)):
            return np.NaN

        behaviors = behavior_pattern.findall(x)
        if not behaviors:
            return np.NaN
        else:
            return behaviors[0]

    data['behav_cate_clean'] = data.behav_cate.map(lambda x: ext_main_behavior(x))
    return data


def clean_department(data):
    # 判断部门是否为外包部门
    data['is_outsource_dept'] = data.dept_name.map(lambda x: 1 if isinstance(x, (str, bytes)) and '外包' in x else 0)
    return data


def clean_ip_address(data):
    # 判断IP是否为行内IP
    inner_pattern = re.compile('^10\.108\.(?:[1-9]|1[2-5]|2[1-9]|3[0-68-9]|4[02457]|53|'
                               '6[01456]|7[2-5]|8[0-7]|9[2-5]|12[8-9]|13[0-9]|14[0-48-9]|15[0-9]|161|19[23])\.\d{1,3}$|'
                               '^10\.95\.(?:[0-9]|1[0-5])\.\d{1,3}$|'
                               '^10\.95\.253\.23[1-3]$|'
                               '^10\.109\.3\.(?:1[1—9]|20)$|'
                               '^10\.88\.0\.\d{1,3}$|'
                               '^10\.4\.0\.9$'
                               )

    wifi_pattern = re.compile('^10\.108\.'
                              '(?:53\.(?:[0-9]|1[0-9]|2[0-4])|'
                              '47\.\d{1,3}|(?:12[8-9]|13[0-5])\.\d{1,3}|'
                              '66\.(?:[0-9]|1[0-9]|2[0-4])|'
                              '14[0-3]\.\d{1,3}|'
                              '15[2-9]\.\d{1,3}|'
                              '161\.(?:[0-9]|1[0-9]|2[0-4]))$'
                              )

    data['is_inner_ip'] = (data.src_ip_addr.map(lambda x: 1
    if isinstance(x, (str, bytes)) and inner_pattern.search(x) else 0))
    data['is_wifi'] = (data.src_ip_addr.map(lambda x: 1
    if isinstance(x, (str, bytes)) and wifi_pattern.search(x) else 0))
    return data


# aggregate along staff
def agg_by_hash_index(staff_data):
    def gen_hash_index(data, index):
        desired_cols = ['src_ip_addr', 'behav_cate_clean', 'sys_name']
        hash_index = hash('|'.join(map(lambda x: str(x), data.loc[index, desired_cols].values)))
        return hash_index

    def get_uniq_value(rows):
        for row in rows:
            if pd.notna(row):
                return row
        return 'NaN'

    def get_multi_value(rows):
        counter = {}
        for row in rows:
            counter[row] = counter.get(row, 0) + 1

        ret = ''
        for k in counter:
            ret += '{}:{}\n'.format(k, counter[k])
        return ret

    def ext_day(datetime_):
        return datetime_[:10]

    staff_data.index = range(staff_data.shape[0])

    # 剔除在一分钟连续操作的行为，即排除多余的系统记录
    desired_indexes = [True]
    hash_indexes = [gen_hash_index(staff_data, 0)]

    for idx in range(1, staff_data.shape[0]):
        pre_hash_index = gen_hash_index(staff_data, idx - 1)
        cur_hash_index = gen_hash_index(staff_data, idx)
        hash_indexes.append(cur_hash_index)

        time_elapsed = staff_data.loc[idx, 'oper_time'] - staff_data.loc[idx - 1, 'oper_time']
        if cur_hash_index == pre_hash_index and time_elapsed.seconds <= 60:
            desired_indexes.append(False)
        else:
            desired_indexes.append(True)

    staff_data['hash_indexes'] = hash_indexes
    staff_data = staff_data.loc[desired_indexes, :]

    # aggregate the rows according to hash_index
    staff_data['behavior_counts'] = 1
    result = staff_data.groupby(by=['hash_indexes']).agg(
        dict(access_res=get_multi_value, acct_num=get_uniq_value, emply_num=get_uniq_value,
             emply_mobile_num=get_uniq_value, dept_name=get_uniq_value, src_ip_addr=get_uniq_value,
             src_mac_addr=get_uniq_value, oper_tm=ext_day, oper_prod_num=get_multi_value, oper_chan_id=get_multi_value,
             oper_mobile_num=get_multi_value, oper_id_card_num=get_multi_value, operdr_empl_char_cd=get_uniq_value,
             operdr_cust_ind=get_multi_value, sys_name=get_uniq_value, emply_char_name=get_uniq_value,
             behav_cate_clean=get_uniq_value, is_outsource_dept=get_uniq_value, is_inner_ip=get_uniq_value,
             is_wifi=get_uniq_value, behavior_counts=np.sum)
    )
    return result


def clean_one_day(data):
    # 添加系统名称
    data = clean_system(data)

    # 内部员工类型
    data = clean_staff(data)

    # 整理操作时间
    data = clean_operation_time(data)

    # 提取主要操作行为
    data = clean_behaviors(data)

    # 判断部门是否为外包部门
    data = clean_department(data)

    # 整理IP地址特征
    data = clean_ip_address(data)

    # 将员工编号设置为数据索引
    data.index = data.emply_num

    # 去除操作行为为空的观测
    data.dropna(subset=['behav_cate_clean'], inplace=True)

    # 汇总一天的所有员工、部门、系统、源机器信息
    graph_data = ToGraphData()
    for index in data.index.drop_duplicates():
        if pd.isna(index):
            continue

        data_per_staff = data.loc[index, :]
        if type(data_per_staff).__name__ != 'DataFrame':
            continue

        data_per_staff = data_per_staff.sort_values(by=['oper_time'])
        agg_data = agg_by_hash_index(data_per_staff)
        graph_data(agg_data)
    return graph_data


def clean_sequential_data(file_name):
    data = pd.read_csv(file_name)
    data['date'] = data.oper_tm.map(lambda x: x[:10])
    data.index = data.date

    results = {}
    for date in data.index.drop_duplicates():
        sub_data = data.loc[date, :]
        sub_data.index = range(sub_data.shape[0])
        graph_data = clean_one_day(sub_data)
        results[date] = graph_data
    return results
