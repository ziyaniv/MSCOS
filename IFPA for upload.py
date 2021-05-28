import scipy.special as sc_special
import numpy as np
import xlrd
import xlwt
from xlutils.copy import copy  #从xlutils模块导入copy
import pandas as pa
import sys
import random



class FPA:

    # iteration = 30  # 最大迭代次数

    p = 0.99
    # initpop=100 #初始化种群数和文件1Comparision_matrices的组合数相同
    def levy():  # n步，m维
        beta = 1.5
        sigma_u = (sc_special.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                sc_special.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
        sigma_v = 1
        u = np.random.normal(0, sigma_u)
        v = np.random.normal(0, sigma_v)
        # Ls = u / np.power(np.fabs(v),1)
        Ls = u / ((abs(v)) ** (1 / beta))  # levy飞行的分布
        # print("Ls is ",Ls)
        return Ls
    def find_col_num(sheet,a, col):#a是元素，b是总列数
        # topsis_xls = xlrd.open_workbook('topsis_matrice_xls_topsis_fpa.xls', 'r')
        # topsis_xls_sheet = topsis_xls.sheet_by_name("topsis_matrice")
        for i in range(col):
            if sheet.cell_value(0,i)==a:
                return i

    def find_row_num(sheet,a,row,col):#a是元素，row是总行数，col是所在列号
        # topsis_xls = xlrd.open_workbook('topsis_matrice_xls_topsis_fpa.xls', 'r')
        # topsis_xls_sheet = topsis_xls.sheet_by_name("topsis_matrice")

        for i in range(1,row):
            # print("col", col,"i",i)
            # print("a",a,"value",sheet.cell_value(i,col))
            if sheet.cell_value(i,col)==a:
                return i

    def find_max_data(a):  # a是列表
        max=0
        # print(a)
        for i in range(len(a)):
           if int(a[i])>int(max):
               max=a[i]
        return max


    def change_servicecomposition(sheet,list,station):#在表'sheet'中，将list替代station行的数据
        pass
        # topsis_matrice_xls = xlrd.open_workbook(sheet, 'r')
        # topsis_matrice_xls_w = copy(topsis_matrice_xls)
        # topsis_matrice_xls_w_sheet = topsis_matrice_xls_w.get_sheet('topsis_matrice')
        # cols=topsis_matrice_xls_w_sheet.ncols
        # for i in range(cols):
        #     topsis_matrice_xls_w_sheet.write(station,i,list[i])
        # topsis_matrice_xls_w.save(sheet+"_t.xls")


    def re_normal(v,st,a,b):
        #1、先替换

        topsis_xls = xlrd.open_workbook("1-3topsis_matrice_xls_topsis.xls", 'w')
        topsis_matrice_xls_n = copy(topsis_xls)  # 拷贝一份原来的excel用于暂时存储替代了新的组合的list
        topsis_matrice_xls_n_sheet = topsis_matrice_xls_n.get_sheet('topsis_matrice')
        for i in range(len(v)):
            topsis_matrice_xls_n_sheet.write(st,i,v[i])
        # print("list is",v,)
        # print("位置是",st)
        topsis_matrice_xls_n.save("topsis_matrice_xls_change1list.xls")
        topsis_matrice_xls_n1 = xlrd.open_workbook("topsis_matrice_xls_change1list.xls", 'r')
        topsis_sheet=topsis_matrice_xls_n1.sheet_by_name("topsis_matrice")
        topsis_normal_copy=copy(topsis_matrice_xls_n1)
        topsis_normal_sheet = topsis_normal_copy.get_sheet("topsis_matrice")

        rows=a
        cols = b
        # print("cols",cols)
        for i in range(0, cols-7):
            if (topsis_sheet. cell_value(0, i) == "sum(QoS_cost)" )or (topsis_sheet.cell_value(0,  i)=="distance similarity"):
                # for j in range(1,rows):
                list_value = topsis_sheet.col_values(i, 1, rows-2)
                # print("list_values",list_value)
                mindata = min(list_value)
                maxdata = max(list_value)
                # print("topsis_matrice_sheet.cell_values(j,i)",list_value[j])
                # print("len list_value ",len(list_value))
                for j in range(0, rows -3):
                    # print("j is ",j)
                    # print("j", j, "rows", rows, list_value[j])
                    data = (maxdata - list_value[j]) / (maxdata - mindata)
                    topsis_normal_sheet.write(j + 1, i, data)
        topsis_normal_copy.save("re_normal.xls")

    def find_martrix_min_value(data_matrix):
        '''''
        功能：找到矩阵最小值
        '''
        new_data = []
        for i in range(len(data_matrix)):
            new_data.append(min(data_matrix[i]))
        return min(new_data)
    def find_martrix_max_value(data_matrix):
        '''''
        功能：找到矩阵最大值
        '''
        new_data = []
        for i in range(len(data_matrix)):
            new_data.append(max(data_matrix[i]))
        return max(new_data)


    def topsis(v):
        topsis_xls=xlrd.open_workbook("1-3topsis_matrice_xls_topsis.xls",'r')
        topsis_xls_sheet=topsis_xls.sheet_by_name("topsis_matrice")
        rows=topsis_xls_sheet.nrows
        cols=topsis_xls_sheet.ncols
        topsis_max=topsis_xls_sheet.row_values(rows-2,0,120)
        print('topsis_max',topsis_max,len(topsis_max))
        topsis_min=topsis_xls_sheet.row_values(rows-1,0,120)
        print('topsis_min',topsis_min,len(topsis_min))
        d_up=np.sqrt(sum(np.power((np.array(v) - np.array(topsis_max)), 2)))#到正理想解的距离
        d_down = np.sqrt(sum(np.power((np.array(v) - np.array(topsis_min)), 2)))#到负理想解的距离
        d=(d_up/(d_up+d_down))*10000
        print("topsis is",d)
        return d

    def grey_corrlation():
        list_grey_coe_up = []  # 用于计算某个组合与理想解的灰色关联系数
        list_grey_coe_down = []  # 用于计算某个组合与负理想解的灰色关联系数
        g_list_up = []  # 用于保存到正理想解的灰色关联度
        g_list_down = []  # 用于保存到负理想解的灰色关联度
        grey_list = []  # 用于保存灰色关联贴近度

        topsis_matrice_xls = xlrd.open_workbook('1-2normal.xls', 'r')
        topsis_matrice_sheet = topsis_matrice_xls.sheet_by_name("topsis_matrice")
        grey_matric_xls=copy(topsis_matrice_xls)
        grey_matric_xls_sheet=grey_matric_xls.get_sheet("topsis_matrice")
        cols = topsis_matrice_sheet.ncols
        # print("--------------cols--------------",cols)
        rows = topsis_matrice_sheet.nrows
        matrix_grey_up = [[0 for i in range(cols)] for j in range(rows)]  # 用于存放每个元素与理想街的灰色关联度的矩阵
        matrix_grey_down = [[0 for i in range(cols)] for j in range(rows)]  # 用于存放每个元素与负理想街的灰色关联度的矩阵
        topsis_max=topsis_matrice_sheet.row_values(rows-2,0,cols)
        topsis_min = topsis_matrice_sheet.row_values(rows - 1, 0, cols )
        print("topsis_max",topsis_max)
        print("topsis_min",topsis_min)
        for j in range(cols-7):
            for c in range(1, rows):
                # print("c,rows",c,rows)
                # print("j,cols",j,cols)
                # print("topsis_max[j]",topsis_max[j])
                # print("topsis_matrice_sheet.cell_value(c, j)",topsis_matrice_sheet.cell_value(c, j))
                matrix_grey_up[c - 1][j] = abs(topsis_max[j] - topsis_matrice_sheet.cell_value(c, j))
                # print("topsis_min[j]", topsis_min[j])
                # print("topsis_matrice_sheet.cell_value(c, j)", topsis_matrice_sheet.cell_value(c, j))
                # print("topsis_matrice_sheet.cell_value(c, j)", topsis_matrice_sheet.cell_value(c, j))
                matrix_grey_down[c - 1][j] = abs(topsis_min[j] - topsis_matrice_sheet.cell_value(c, j))
        # print("matrix_grey_up",matrix_grey_up)
        # print("matrix_grey_down",matrix_grey_down)
        for g in range(cols):
            for h in range(rows):
                # print("g,h",g,h)

                a = FPA.find_martrix_min_value(matrix_grey_up)
                b = FPA.find_martrix_max_value(matrix_grey_up)
                c = FPA.find_martrix_min_value(matrix_grey_down)
                d = FPA.find_martrix_max_value(matrix_grey_down)
                # print("a",a,"b",b,"c",c,"d",d,"matrix_grey_up[h][g]",matrix_grey_up[h][g])

                list_grey_coe_up.append((a + 0.5 * b) / ((matrix_grey_up[h][g]) + 0.5 * b))

                list_grey_coe_down.append((c + 0.5 * d) / (
                        (matrix_grey_down[h][g]) + 0.5 * d))
            g_up = sum(list_grey_coe_up) / len(list_grey_coe_up)
            g_list_up.append(g_up)
            # print("与正理想解的灰色关联度",g_up)
            g_down = sum(list_grey_coe_down) / len(list_grey_coe_down)
            g_list_down.append(g_down)
            grey_matric_xls_sheet.write(h + 1, cols + 2, g_up)
            grey_matric_xls_sheet.write(h + 1, cols + 3, g_down)

        for ro in range(rows - 1):

            g_cd = ((g_list_up[ro]) / (g_list_up[ro] + g_list_down[ro])) * 10000  # 第ro组的综合灰色贴近度
            grey_list.append(g_cd)
        print("新组合的grey值为：",grey_list[])




    def evaluation(c1,levy_next):
        v1=[]
        t_list = []#r任务参数列表
        task_cuttool = xlrd.open_workbook('task-cuttool.xls', 'r')
        task_sheet = task_cuttool.sheet_by_name('Sheet1')
        task_num = task_sheet.nrows
        for i in range(task_sheet.ncols):
            if task_sheet.cell_value(0, i) == 'Func_CuttingSpeed (mm/min)':
                t_cs_col = i
            if task_sheet.cell_value(0, i) == 'Func_FeedRate (mm/r)':
                t_fr_col = i
            if task_sheet.cell_value(0, i) == 'Func_CuttingDepth (mm)':
                t_cd_col = i

        for j in range(1, task_num):
            t_list.append(task_sheet.cell_value(j, t_cs_col))
            t_list.append(task_sheet.cell_value(j, t_fr_col))
            t_list.append(task_sheet.cell_value(j, t_cd_col))  # 此时A为任务参数的对比列
        print("任务参数列表是:",t_list)

        task_cuttool = xlrd.open_workbook('task-cuttool.xls', 'r') #打开刀具的参数所在的excel
        task_cuttool_topsis = xlrd.open_workbook('1-3topsis_matrice_xls_topsis.xls', 'r')  # 打开刀具的参数所在的excel
        sheet1 = task_cuttool.sheet_by_name('Sheet1')
        sheet2 = task_cuttool.sheet_by_name('Sheet2')
        t_rows=sheet2.nrows
        ideal_xls = task_cuttool_topsis.sheet_by_name('topsis_matrice')
        rows=ideal_xls.nrows
        cols=ideal_xls.ncols
        # print("rows",rows)
        # print("cols",cols)
        # i_s_list=ideal_xls.row_values(rows-2,0,cols-7)#正理想解存放的list
        # n_i_s_list=ideal_xls.row_values(rows-1,0,cols-7)#负理想解存放的list
        # print("正理想解的list",i_s_list,len(i_s_list))
        # print("负理想解的list",n_i_s_list,len(n_i_s_list))
        len_c=len(c1) #计算组合的长度，注意，每个服务有5个参数
        print("len_c",len_c)
        A=[]#用于单个任务刀具的参数
        B=[]#用于存储服务的参数
        for i in range(len_c):
            cs_id=FPA.find_col_num(sheet1,'Func_CuttingSpeed (mm/min)',11)#共有12列
            fr_id = FPA.find_col_num(sheet1,'Func_FeedRate (mm/r)', 11)  # 共有12列
            fr_cd = FPA.find_col_num(sheet1,'Func_CuttingDepth (mm)', 11)
            A.append(sheet1.cell_value(1,cs_id))
            A.append(sheet1.cell_value(1, fr_id))
            A.append(sheet1.cell_value(1, fr_cd))
            ct_cs_id = FPA.find_col_num(sheet2,'Func_CuttingSpeed (mm/min)', 11)  # 共有12列
            ct_fr_id = FPA.find_col_num(sheet2,'Func_FeedRate (mm/r)', 11)  # 共有12列
            ct_fr_cd = FPA.find_col_num(sheet2,'Func_CuttingDepth (mm)', 11)
            print("c1[i]",c1[i])
            # find_row_num(sheet, a, row, col):  # a是元素，row是总行数，col是所在列号
            # print("t_rows",t_rows)
            ct_id=FPA.find_row_num(sheet2,int(c1[i]),t_rows,0)
            # print("ct_id",ct_id)
            B.append(sheet2.cell_value(ct_id,ct_cs_id))
            B.append(sheet2.cell_value(ct_id, ct_fr_id))
            B.append(sheet2.cell_value(ct_id, ct_fr_cd))

            d_sim = np.sqrt(sum(np.power((np.array(A) - np.array(B)), 2)))  # c1功能参数与任务的欧式距离
            v1.append(d_sim)
            dot_product, square_sum_x, square_sum_y = 0, 0, 0
            for i_a in range(len(A)):
                dot_product += A[i_a] * B[i_a]
                square_sum_x += A[i_a] * A[i_a]
                square_sum_y += B[i_a] * B[i_a]
            cos_sim = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))
            v1.append(cos_sim)
            ct_relia_id = FPA.find_col_num(sheet2,'reliability', 11)  # 共有12列
            ct_recyc_id = FPA.find_col_num(sheet2,'recyclable', 11)
            ct_cost_id = FPA.find_col_num(sheet2,'cost', 11)
            v1.append(sheet2.cell_value(ct_id,ct_cost_id))
            v1.append(sheet2.cell_value(ct_id, ct_relia_id))
            v1.append(sheet2.cell_value(ct_id, ct_recyc_id))

        print("v1",v1,len(v1))

        # FPA.re_normal(v1,levy_next,rows,cols)

        FPA.topsis(v1)
        FPA.grey_corrlation()


    def calculation_parameters(work_xls):
        sheet=work_xls.get_sheet("topsis_matrice")
        xls_name_list = ['distance similarity', 'Cosine similarity', 'sum(QoS_cost)', 'accrued(reliability)',
                         'accrued(recyclable)']
        l = len(xls_name_list)
        tasknum=24

        task_cuttool = xlrd.open_workbook('task-cuttool.xls', 'r')
        cuttool_sheet = task_cuttool.sheet_by_name('Sheet2')
        for i in range(cuttool_sheet.ncols):
            if cuttool_sheet.cell_value(0, i) == 'Func_CuttingSpeed (mm/min)':
                c_cs_col = i
            if cuttool_sheet.cell_value(0, i) == 'Func_FeedRate (mm/r)':
                c_fr_col = i
            if cuttool_sheet.cell_value(0, i) == 'Func_CuttingDepth (mm)':
                c_cd_col = i
            if cuttool_sheet.cell_value(0, i) == 'reliability':
                c_relia_col = i
            if cuttool_sheet.cell_value(0, i) == 'recyclable':
                c_recyc_col = i
            if cuttool_sheet.cell_value(0, i) == 'cost':
                c_cost_col = i
        task_sheet = task_cuttool.sheet_by_name('Sheet1')

        for i in range(task_sheet.ncols):
            if task_sheet.cell_value(0, i) == 'Func_CuttingSpeed (mm/min)':
                t_cs_col = i
            if task_sheet.cell_value(0, i) == 'Func_FeedRate (mm/r)':
                t_fr_col = i
            if task_sheet.cell_value(0, i) == 'Func_CuttingDepth (mm)':
                t_cd_col = i
        cuttool_rows = cuttool_sheet.nrows
        task_rows = task_sheet.nrows
        initpop_num=sheet.ncol
        for i in range(1, initpop_num + 1):
            for j in range(1, task_rows):
                sheet.write(i, j - 1, initpop_list[i - 1][j - 1])
                A = []  # 记录任务的参数
                B = []  # 记录刀具的参数
                relia = []  # 记录刀具QOS-relia的参数
                recyc = []  # 记录刀具QOS-recyc的参数
                cost = []  # 记录刀具QOS-cost的参数
                # print("j",j)
                A.append(task_sheet.cell_value(j, t_cs_col))
                A.append(task_sheet.cell_value(j, t_fr_col))
                A.append(task_sheet.cell_value(j, t_cd_col))  # 此时A为任务参数的对比列
                # print("A is",A )
                for cl in range(1, cuttool_rows):
                    # print("cuttool_rows",cuttool_rows)
                    # print("i,j,cl",i,j,cl)
                    # print("initpop_list",initpop_list[i][j-1])
                    if initpop_list[i - 1][j - 1] == cuttool_sheet.cell_value(cl, 0):
                        B.append(cuttool_sheet.cell_value(cl, c_cs_col))
                        B.append(cuttool_sheet.cell_value(cl, c_fr_col))
                        B.append(cuttool_sheet.cell_value(cl, c_cd_col))
                        relia.append(cuttool_sheet.cell_value(cl, c_relia_col))
                        recyc.append(cuttool_sheet.cell_value(cl, c_recyc_col))
                        cost.append(cuttool_sheet.cell_value(cl, c_cost_col))
                        break
                # print("B is", B)
                # print("relia is", relia)
                # print("recyc is", recyc)
                # print("cost is", cost)
                d_sim = np.sqrt(sum(np.power((np.array(A) - np.array(B)), 2)))  # 功能参数的欧式距离
                # print("d_sim",d_sim)

                dot_product, square_sum_x, square_sum_y = 0, 0, 0
                for i_a in range(len(A)):
                    dot_product += A[i_a] * B[i_a]
                    square_sum_x += A[i_a] * A[i_a]
                    square_sum_y += B[i_a] * B[i_a]
                cos_sim = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))

                # vector_a = np.mat(A)
                # vector_b = np.mat(B)
                # num = float(A * B.T)
                # denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
                # cos = num / denom
                # cos_sim = 0.5 + 0.5 * cos #功能参数的余弦相似度

                # print("cos_sim",cos_sim)
                relia_mul = np.cumprod(relia)
                # print('relia_mul',relia_mul)
                recyc_mul = np.cumprod(recyc)
                # print('recyc_mul', recyc_mul)
                cost_add = sum(cost)
                # print('cost_add',cost_add)

                sheet1.write((i), (l * (j - 1) + 0), d_sim)
                sheet1.write((i), (l * (j - 1) + 1), cos_sim)
                sheet1.write((i), (l * (j - 1) + 2), cost_add)
                sheet1.write((i), (l * (j - 1) + 3), relia_mul[0])
                sheet1.write((i), (l * (j - 1) + 4), recyc_mul[0])
            # print(i)
        topsis_matrice_xls.save('1-1topsis_matrice_xls.xls')


    def Crossover(): #异花授粉
        topsis_xls = xlrd.open_workbook('1-3topsis_matrice_xls_topsis.xls', 'r')
        topsis_xls_sheet = topsis_xls.sheet_by_name("topsis_matrice")
        topsis_composition_sheet = topsis_xls.sheet_by_name("service_composition")
        # topsis_xls = xlrd.open_workbook('topsis_matrice_xls_topsis_fpa.xls', 'r')
        # topsis_xls_sheet = topsis_xls.sheet_by_name("topsis_matrice")
        cols = topsis_xls_sheet.ncols
        rows = topsis_xls_sheet.nrows
        col = FPA.find_col_num(topsis_xls_sheet,'topsis_grey', cols)
        initpop = topsis_xls_sheet.nrows - 1
        # print("initpop",initpop)
        #区域内读取topsis最大值
        spannum=10 #spannum划分的区域个数
        span=int(initpop/spannum )#span为区域跨度
        # print("span",span)
        topsis_grey_list = topsis_xls_sheet.col_values(col, 1, initpop+1)
        # print("topsis_grey_list", topsis_grey_list,len(topsis_grey_list))
        for i in range(0,spannum):
            c1=[]
            c2=[]
            topsis_grey_list = topsis_xls_sheet.col_values(col, (i)*span+1, (i+1)*span+1) #xls第一行为题目，所以行数要加1
            max_tg = FPA.find_max_data(topsis_grey_list) #找到该区域中的最大值
            composition_row = FPA.find_row_num(topsis_xls_sheet,max_tg, rows, col) # 找到该最大值对应的服务组合号，即行号
            print("服务组合号为",composition_row)
            levy_next = (int(FPA.levy() * initpop % initpop) + composition_row) % initpop #结合最大服务组合号与levy飞行得回的位置
            print("levy_next", levy_next)
            list_levy = topsis_composition_sheet.row_values(levy_next-1, 0, cols) #得回levy对应的服务组合
            list_max= topsis_composition_sheet.row_values(composition_row, 0, cols)#得回composition_row对应的服务组合
            # print("list_max", list_max)
            # print("list_levy", list_levy)
            for j in range(0,len(list_levy)-1,2):
                c1.append(list_levy[j])
                c1.append(list_max[j+1])
                c2.append(list_max[j])
                c2.append(list_levy[j + 1])

            # print("c1 is", c1)
            # print("c2 is", c2)
            #或得C1与C2两个组合的参数指标
            FPA.evaluation(c1,levy_next)




        # print("Crossover OK")
    def self_pollination():

        print("self_pollination OK")

    def FPA():
        iterations=2
        # topsis_xls = xlrd.open_workbook('topsis_matrice_xls_topsis_fpa.xls', 'r')
        # topsis_xls_sheet=topsis_xls.sheet_by_name("topsis_matrice")
        # cols=topsis_xls_sheet.ncols
        # rows=topsis_xls_sheet.nrows
        # col=FPA.find_col_num('topsis_grey',cols)
        # initpop=topsis_xls_sheet.nrows-1
        # # print("topsis_xls_sheet.nrows",topsis_xls_sheet.nrows-1)
        # topsis_grey_list=topsis_xls_sheet.col_values(col,1,rows-1)
        # # print("topsis_grey_list",topsis_grey_list)
        # max_tg=FPA.find_max_data(topsis_grey_list) #最大的topsis_grey相关系数
        # print("max_tg",max_tg)
        # max_row=FPA.find_row_num(max_tg,rows,col)
        for ita in range ( iterations):
            if random.random()>FPA.p:
                FPA.self_pollination()
            else:
                FPA.Crossover()






        return


if __name__ == '__main__':

    FPA.FPA()