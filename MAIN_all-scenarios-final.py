from docplex.mp.model import Model
import numpy as np
import random
import math
import platform
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import os
import time
from matplotlib import rcParams

rcParams['font.family'] = "serif"

big_number = 100000
Nb_exp = 100
dataset_collection = {}
dataset = pd.DataFrame()

COST_RANDOM = []
THROUGHPUT_RANDOM = []
ADMITTANCE_RANDOM = []
RB_utilization = []
fairness_EDGE = []
fairness_RANDOM = []
fairness_OPTIMAL = []
fairness_REGIONAL = []

RB_utilization_1_all_runs = []  # rb utilization at each BS
RB_utilization_2_all_runs = []
RB_utilization_3_all_runs = []
RB_utilization_4_all_runs = []

RB_demand_1_all_runs = []
RB_demand_2_all_runs = []
RB_demand_3_all_runs = []
RB_demand_4_all_runs = []

GOPS_utilization_1_all_runs = []  # gops utilization at each server
GOPS_utilization_2_all_runs = []
GOPS_utilization_3_all_runs = []
GOPS_utilization_4_all_runs = []

GOPS_utilization_all_edge_all_runs = []

COST_EDGE = []
THROUGHPUT_EDGE = []
ADMITTANCE_EDGE = []

COST_OPTIMAL = []
THROUGHPUT_OPTIMAL = []
ADMITTANCE_OPTIMAL = []

COST_regional = []
THROUGHPUT_regional = []
ADMITTANCE_regional = []

RATIO_eMBB_optimal = []
RATIO_eMBB_random = []
RATIO_eMBB_edge = []
RATIO_eMBB_regional = []

RATIO_uRLLC_optimal = []
RATIO_uRLLC_random = []
RATIO_uRLLC_edge = []
RATIO_uRLLC_regional = []

RATIO_mMTC_optimal = []
RATIO_mMTC_random = []
RATIO_mMTC_edge = []
RATIO_mMTC_regional = []

ratio_CU_edge_eMbb_optimal_all_runs = []
ratio_CU_edge_uRLLC_optimal_all_runs = []
ratio_CU_edge_mMTC_optimal_all_runs = []

TIME_MILP = []
time_MILP = []

for tt in range(Nb_exp):
    data_edge = []
    CU_edge_eMbb = []
    CU_edge_uRLLC = []
    CU_edge_MTC = []

    Nb_Admitted_slices = []

    loc_x_reg = []
    loc_y_reg = []
    loc_x_edge = []
    loc_y_edge = []

    link_utilization_random = []
    link_utilization_edge = []
    link_utilization_optimal = []
    link_utilization_regional = []

    RB_utilization_1 = []
    RB_utilization_2 = []
    RB_utilization_3 = []
    RB_utilization_4 = []

    RB_1_TOTALL = []
    RB_2_TOTALL = []
    RB_3_TOTALL = []
    RB_4_TOTALL = []

    GOPS_utilization_1 = []
    GOPS_utilization_2 = []
    GOPS_utilization_3 = []
    GOPS_utilization_4 = []

    GOPS_utilization_all_edge = []

    # GOPS_utilization_random = []
    # GOPS_utilization_edge = []
    # GOPS_utilization_optimal = []
    # GOPS_utilization_regional = []

    ratio_admitted_random = []
    ratio_admitted_eMBB_random = []
    ratio_admitted_uRLLC_random = []
    ratio_admitted_mMTC_random = []

    ratio_admitted_edge = []
    ratio_admitted_eMBB_edge = []
    ratio_admitted_uRLLC_edge = []
    ratio_admitted_mMTC_edge = []

    ratio_admitted_optimal = []
    ratio_admitted_eMBB_optimal = []
    ratio_admitted_uRLLC_optimal = []
    ratio_admitted_mMTC_optimal = []

    ratio_admitted_regional = []
    ratio_admitted_eMBB_regional = []
    ratio_admitted_uRLLC_regional = []
    ratio_admitted_mMTC_regional = []

    ratio_CU_reg_eMbb_random = []
    ratio_CU_reg_uRLLC_random = []
    ratio_CU_reg_MTC_random = []
    ratio_CU_edge_eMbb_random = []
    ratio_CU_edge_uRLLC_random = []
    ratio_CU_edge_MTC_random = []

    ratio_CU_reg_eMbb_edge = []
    ratio_CU_reg_uRLLC_edge = []
    ratio_CU_reg_MTC_edge = []
    ratio_CU_edge_eMbb_edge = []
    ratio_CU_edge_uRLLC_edge = []
    ratio_CU_edge_MTC_edge = []

    ratio_CU_reg_eMbb_optimal = []
    ratio_CU_reg_uRLLC_optimal = []
    ratio_CU_reg_MTC_optimal = []
    ratio_CU_edge_eMbb_optimal = []
    ratio_CU_edge_uRLLC_optimal = []
    ratio_CU_edge_MTC_optimal = []

    ratio_CU_reg_eMbb_regional = []
    ratio_CU_reg_uRLLC_regional = []
    ratio_CU_reg_MTC_regional = []
    ratio_CU_edge_eMbb_regional = []
    ratio_CU_edge_uRLLC_regional = []
    ratio_CU_edge_MTC_regional = []

    cost_random = []
    cost_edge = []
    cost_optimal = []
    cost_regional = []

    total_throughput_random = []
    total_throughput_edge = []
    total_throughput_optimal = []
    total_throughput_regional = []

    fairness_edge = []
    fairness_optimal = []
    fairness_random = []
    fairness_regional = []

    time_MILP = []

    n_servers = 4
    n_servers_edge = 3

    N = np.arange(10,150,10)
    # ratio_eMBB=ratio_uRLLC=ratio_MTC = np.empty((0), float)
    # objectivee= np.empty((0), int)
    Number_slices = [n for n in N]
    ###servers locations are fixed for all the experiments even when increasing the nb of UEs
    x_center = 0.500
    y_center = 0.500
    servers_edge = [i for i in range(n_servers_edge)]
    for i in range(n_servers_edge):
        r = random.randint(5,10)
        ang = random.uniform(0,1) * 2 * math.pi
        loc_x_edge.append(r * math.cos(ang)+x_center)
        loc_y_edge.append(r * math.sin(ang)+y_center)

    for i in range(n_servers_edge,n_servers):
        r = random.randint(40,80)
        ang = random.uniform(0,1) * 2 * math.pi
        loc_x_reg.append(r * math.cos(ang)+x_center)
        loc_y_reg.append(r * math.sin(ang)+y_center)

    for NN in Number_slices:

        # print ('n_slices=',NN )

        m = Model(name='RA-ORAN_optimal')
        m.parameters.timelimit = 60

        m0 = Model(name='RA-ORAN_random')
        m0.parameters.timelimit = 60

        m1 = Model(name='RA-ORAN_all_edge')
        m1.parameters.timelimit = 60

        m2 = Model(name='RA-ORAN_all_regional')
        m2.parameters.timelimit = 60

        n_servers = 4

        Nb_BS = 4  # number of base_stations(supposed, it can be between a random interval
        RB_utilization = []

        # 25,25,50 INDUSTRIAL area
        # 50,30,20

        if (NN / 10) % 2 == 0:
            n_eMBB = int(NN * 0.25)
        else:
            n_eMBB = int(NN * 0.25)+1

        n_uRLLC = int(NN * 0.25)
        n_MTC = int(NN * 0.5)

        n_slices = n_eMBB+n_MTC+n_uRLLC

        servers = [i for i in range(n_servers)]
        slices = [i for i in range(n_slices)]

        i = 0
        S = np.empty([n_servers * n_servers,2],dtype=int)
        for s in range(n_servers):
            for s1 in range(n_servers):
                S[i] = (s,s1)
                i = i+1
        loc_x_BS = []
        loc_y_BS = []

        rnd = np.random
        # rnd.seed(0)
        loc_x = rnd.rand(n_slices) * 1
        loc_y = rnd.rand(n_slices) * 1
        loc_x_BS = [0.250,0.250,0.750,0.750]
        loc_y_BS = [0.250,0.750,0.250,0.750]

        RB_max = 100

        RB_total = 100 * Nb_BS  # 20MHZ bandwith= 100 RBs each BS

        N_RB_rem_embb = [50,50,50,50]
        N_RB_rem_urllc = [25,25,25,25]
        N_RB_rem_mmtc = [25,25,25,25]

        RB_eMBB = []
        RB_uRLLC = []
        RB_MTC = []

        ### TO DONT WASTE RESOURCES, REMAINING RBs OF URLLC AND MMTC ARE ASSIGNED TO EMBB USERS
        for i in range(n_eMBB,n_slices):  # allocate uRLLC and mmTC before eMBB
            if i < n_eMBB+n_uRLLC:
                if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1,5)
                    if N_RB_rem_urllc[0]-r >= 0:
                        RB_uRLLC.append(r)
                        N_RB_rem_urllc[0] -= r
                    elif N_RB_rem_urllc[0] > 0:  # giving the remaining RBs to the last slice(to not waste RBs)
                        RB_uRLLC.append(N_RB_rem_urllc[0])
                        N_RB_rem_urllc[0] = 0
                    else:
                        RB_uRLLC.append(0)
                elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1,5)
                    if N_RB_rem_urllc[1]-r >= 0:
                        RB_uRLLC.append(r)
                        N_RB_rem_urllc[1] -= r
                    elif N_RB_rem_urllc[1] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_uRLLC.append(N_RB_rem_urllc[1])
                        N_RB_rem_urllc[1] = 0
                    else:
                        RB_uRLLC.append(0)
                elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1,5)
                    if N_RB_rem_urllc[2]-r >= 0:
                        RB_uRLLC.append(r)
                        N_RB_rem_urllc[2] -= r
                    elif N_RB_rem_urllc[2] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_uRLLC.append(N_RB_rem_urllc[2])
                        N_RB_rem_urllc[2] = 0
                    else:
                        RB_uRLLC.append(0)
                elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1,5)
                    if N_RB_rem_urllc[3]-r >= 0:
                        RB_uRLLC.append(r)
                        N_RB_rem_urllc[3] -= r
                    elif N_RB_rem_urllc[3] > 0:  # giving the remaining RBs to the last slice(to not waste RBs)
                        RB_uRLLC.append(N_RB_rem_urllc[3])
                        N_RB_rem_urllc[3] = 0
                    else:
                        RB_uRLLC.append(0)
            elif i < n_eMBB+n_uRLLC+n_MTC:
                if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1,5)
                    if N_RB_rem_mmtc[0]-r >= 0:
                        RB_MTC.append(r)
                        N_RB_rem_mmtc[0] -= r
                    elif N_RB_rem_mmtc[0] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_MTC.append(N_RB_rem_mmtc[0])
                        N_RB_rem_mmtc[0] = 0
                    else:
                        RB_MTC.append(0)
                elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1,5)
                    if N_RB_rem_mmtc[1]-r >= 0:
                        RB_MTC.append(r)
                        N_RB_rem_mmtc[1] -= r
                    elif N_RB_rem_mmtc[1] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_MTC.append(N_RB_rem_mmtc[1])
                        N_RB_rem_mmtc[1] = 0
                    else:
                        RB_MTC.append(0)
                elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1,5)
                    if N_RB_rem_mmtc[2]-r >= 0:
                        RB_MTC.append(r)
                        N_RB_rem_mmtc[2] -= r
                    elif N_RB_rem_mmtc[2] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_MTC.append(N_RB_rem_mmtc[2])
                        N_RB_rem_mmtc[2] = 0
                    else:
                        RB_MTC.append(0)
                elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1,5)
                    if N_RB_rem_mmtc[3]-r >= 0:
                        RB_MTC.append(r)
                        N_RB_rem_mmtc[3] -= r
                    elif N_RB_rem_mmtc[3] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_MTC.append(N_RB_rem_mmtc[3])
                        N_RB_rem_mmtc[3] = 0

                    else:
                        RB_MTC.append(0)

        N_RB_rem_embb = N_RB_rem_embb[0]+np.add(N_RB_rem_mmtc,N_RB_rem_urllc)  # 50+remaining RBs or 250+remaining RBs

        for i in range(n_eMBB):
            if loc_x[i] < 0.500 and loc_y[i] < 0.500:  # this user is ascociated to the first BS
                r = random.randint(10,20)
                if N_RB_rem_embb[0]-r >= 0:
                    RB_eMBB.append(r)
                    N_RB_rem_embb[0] -= r
                elif N_RB_rem_embb[0] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                    RB_eMBB.append(N_RB_rem_embb[0])
                    N_RB_rem_embb[0] = 0
                else:
                    RB_eMBB.append(0)
            elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                r = random.randint(10,20)
                if N_RB_rem_embb[1]-r >= 0:
                    RB_eMBB.append(r)
                    N_RB_rem_embb[1] -= r
                elif N_RB_rem_embb[1] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                    RB_eMBB.append(N_RB_rem_embb[1])
                    N_RB_rem_embb[1] = 0
                else:
                    RB_eMBB.append(0)
            elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                r = random.randint(10,20)
                if N_RB_rem_embb[2]-r >= 0:
                    RB_eMBB.append(r)
                    N_RB_rem_embb[2] -= r
                elif N_RB_rem_embb[2] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                    RB_eMBB.append(N_RB_rem_embb[2])
                    N_RB_rem_embb[2] = 0
                else:
                    RB_eMBB.append(0)
            elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                r = random.randint(10,20)
                if N_RB_rem_embb[3]-r >= 0:
                    RB_eMBB.append(r)
                    N_RB_rem_embb[3] -= r
                elif N_RB_rem_embb[3] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                    RB_eMBB.append(N_RB_rem_embb[3])
                    N_RB_rem_embb[3] = 0
                else:
                    RB_eMBB.append(0)
        RB_slices = np.concatenate((RB_eMBB,RB_uRLLC,RB_MTC))
        # # print("N_slices=",NN)
        # print('RBs',RB_slices)
        RB_1 = RB_2 = RB_3 = RB_4 = 0
        ##collecting RB Load:
        for i in range(n_slices):
            if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                RB_1 += RB_slices[i]
            elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                RB_2 += RB_slices[i]
            elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                RB_3 += RB_slices[i]
            elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                RB_4 += RB_slices[i]
        # print("N_slices=", NN)
        # print('RBs', RB_slices)
        # print(RB_1,RB_2,RB_3,RB_4)
        RB_utilization_1.append(RB_1)  # out of 100 RBs of each BS
        RB_utilization_2.append(RB_2)
        RB_utilization_3.append(RB_3)
        RB_utilization_4.append(RB_4)
        ########now runnning 3 different scenarios but with same topology of the network

        ########################TOTAL RB demand

        RB_1_total_demand = 0
        RB_2_total_demand = 0
        RB_3_total_demand = 0
        RB_4_total_demand = 0

        for i in range(n_slices):
            if i < n_eMBB:
                if loc_x[i] < 0.500 and loc_y[i] < 0.500:  # this user is ascociated to the first BS
                    r = random.randint(10,20)
                    RB_1_total_demand += r
                elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                    r = random.randint(10,20)
                    RB_2_total_demand += r
                elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                    r = random.randint(10,20)
                    RB_3_total_demand += r
                elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                    r = random.randint(10,20)
                    RB_4_total_demand += r
            if i < n_eMBB+n_uRLLC:
                if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1,5)
                    RB_1_total_demand += r
                elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1,5)
                    RB_2_total_demand += r
                elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1,5)
                    RB_3_total_demand += r
                elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1,5)
                    RB_4_total_demand += r
            elif i < n_eMBB+n_uRLLC+n_MTC:
                if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1,5)
                    RB_1_total_demand += r
                elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1,5)
                    RB_2_total_demand += r
                elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1,5)
                    RB_3_total_demand += r
                elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1,5)
                    RB_4_total_demand += r
        RB_1_TOTALL.append(RB_1_total_demand)
        RB_2_TOTALL.append(RB_2_total_demand)
        RB_3_TOTALL.append(RB_3_total_demand)
        RB_4_TOTALL.append(RB_4_total_demand)

        ###############

        MMM = []

        scenarios = [n for n in range(0,4)]
        for scenario in scenarios:  ##we will run4 scenarios: all edge, optimal , random, all regional

            n_servers = 4
            if scenario == 0:  # random
                n_servers_edge = 3
                n_servers = 4
                i = 0
                for s in range(n_servers):
                    for s1 in range(n_servers):
                        S[i] = (s,s1)
                        i = i+1
                # desicion variables
                theta_CU_random = m0.binary_var_matrix(range(n_slices),range(n_servers),name='theta_CU')
                theta_DU_random = m0.binary_var_matrix(range(n_slices),range(n_servers),name='theta_DU')

                z_random = m0.binary_var_cube(range(n_slices),range(n_servers),range(n_servers),name='z_random')

                c1 = m0.add_constraints(m0.sum(theta_CU_random[i,s] for s in range(n_servers)) <= 1 for i in range(n_slices))
                c2 = m0.add_constraints(m0.sum(theta_DU_random[i,s] for s in range(n_servers)) <= 1 for i in range(n_slices))
                #             c3=m.add_constraints(m.sum(theta_DU[i,s] for s in range(n_servers_edge,n_servers)) == 0 for i in range(n_slices)) #not necessarly a solution can be found for DU
                # c3=m.add_constraints(m.sum(theta_DU[i,s] for s in range(0,n_servers_edge)) == 1 for i in range(n_slices))
                c55 = m0.add_constraints(m0.sum(theta_CU_random[i,s] for s in range(n_servers)) == m0.sum(theta_DU_random[i,s] for s in range(n_servers)) for i in range(n_slices))

            if scenario == 1:  ##all edge
                # n_servers_edge=n_servers
                n_servers = 3  # total servers at edge

                i = 0
                S = np.empty([n_servers * n_servers,2],dtype=int)
                for s in range(n_servers):
                    for s1 in range(n_servers):
                        S[i] = (s,s1)
                        i = i+1

                # desicion variables
                theta_CU_edge = m1.binary_var_matrix(range(n_slices),range(n_servers),name='theta_CU')
                theta_DU_edge = m1.binary_var_matrix(range(n_slices),range(n_servers),name='theta_DU')

                z_edge = m1.binary_var_cube(range(n_slices),range(n_servers),range(n_servers),name='z_edge')

                c1 = m1.add_constraints(m1.sum(theta_CU_edge[i,s] for s in range(n_servers)) <= 1 for i in range(n_slices))
                c2 = m1.add_constraints(m1.sum(theta_DU_edge[i,s] for s in range(n_servers)) <= 1 for i in range(n_slices))
                c3 = m1.add_constraints(m1.sum(theta_DU_edge[i,s] for s in range(n_servers_edge,n_servers)) == 0 for i in range(n_slices))  # necessarly DU cann't be at regional
                # c3=m.add_constraints(m.sum(theta_DU[i,s] for s in range(0,n_servers_edge)) == 1 for i in range(n_slices))
                c55 = m1.add_constraints(m1.sum(theta_CU_edge[i,s] for s in range(n_servers)) == m1.sum(theta_DU_edge[i,s] for s in range(n_servers)) for i in range(n_slices))

            if scenario == 2:  # optimal with
                n_servers = 4
                n_servers_edge = 3
                i = 0
                S = np.empty([n_servers * n_servers,2],dtype=int)
                for s in range(n_servers):
                    for s1 in range(n_servers):
                        S[i] = (s,s1)
                        i = i+1

                # desicion variables
                theta_CU_optimal = m.binary_var_matrix(range(n_slices),range(n_servers),name='theta_CU')
                theta_DU_optimal = m.binary_var_matrix(range(n_slices),range(n_servers),name='theta_DU')

                z_optimal = m.binary_var_cube(range(n_slices),range(n_servers),range(n_servers),name='z_optimal')

                c1 = m.add_constraints(m.sum(theta_CU_optimal[i,s] for s in range(n_servers)) <= 1 for i in range(n_slices))
                c2 = m.add_constraints(m.sum(theta_DU_optimal[i,s] for s in range(n_servers)) <= 1 for i in range(n_slices))
                c3 = m.add_constraints(m.sum(theta_DU_optimal[i,s] for s in range(n_servers_edge,n_servers)) == 0 for i in range(n_slices))  # not necessarly a solution can be found for DU
                # c3=m.add_constraints(m.sum(theta_DU[i,s] for s in range(0,n_servers_edge)) == 1 for i in range(n_slices))
                c55 = m.add_constraints(m.sum(theta_CU_optimal[i,s] for s in range(n_servers)) == m.sum(theta_DU_optimal[i,s] for s in range(n_servers)) for i in range(n_slices))

            if scenario == 3:  # all regional
                n_servers = 4
                n_servers_edge = 3
                i = 0
                S = np.empty([n_servers * n_servers,2],dtype=int)
                for s in range(n_servers):
                    for s1 in range(n_servers):
                        S[i] = (s,s1)
                        i = i+1

                # desicion variables
                theta_CU_regional = m2.binary_var_matrix(range(n_slices),range(n_servers),name='theta_CU')
                theta_DU_regional = m2.binary_var_matrix(range(n_slices),range(n_servers),name='theta_DU')

                z_regional = m2.binary_var_cube(range(n_slices),range(n_servers),range(n_servers),name='z_regional')

                c1 = m2.add_constraints(m2.sum(theta_CU_regional[i,s] for s in range(n_servers)) <= 1 for i in range(n_slices))
                c2 = m2.add_constraints(m2.sum(theta_DU_regional[i,s] for s in range(n_servers)) <= 1 for i in range(n_slices))
                c3 = m2.add_constraints(m2.sum(theta_DU_regional[i,s] for s in range(n_servers_edge,n_servers)) == 0 for i in range(n_slices))  # not necessarly a solution can be found for DU
                c999 = m2.add_constraints(m2.sum(theta_CU_regional[i,s] for s in range(0,n_servers_edge)) == 0 for i in range(n_slices))
                c55 = m2.add_constraints(m2.sum(theta_CU_regional[i,s] for s in range(n_servers)) == m2.sum(theta_DU_regional[i,s] for s in range(n_servers)) for i in range(n_slices))

            for i in range(n_slices):
                if RB_slices[i] == 0:
                    if scenario == 0:
                        m0.add_constraints(theta_DU_random[i,s] == 0 for s in range(n_servers))
                        m0.add_constraints(theta_CU_random[i,s] == 0 for s in range(n_servers))
                    elif scenario == 1:
                        m1.add_constraints(theta_DU_edge[i,s] == 0 for s in range(n_servers))
                        m1.add_constraints(theta_CU_edge[i,s] == 0 for s in range(n_servers))
                    elif scenario == 2:
                        m.add_constraints(theta_DU_optimal[i,s] == 0 for s in range(n_servers))
                        m.add_constraints(theta_CU_optimal[i,s] == 0 for s in range(n_servers))
                    else:
                        m2.add_constraints(theta_DU_regional[i,s] == 0 for s in range(n_servers))
                        m2.add_constraints(theta_CU_regional[i,s] == 0 for s in range(n_servers))

            #############link_capacity_constraint
            N_sym = 14  # number of symbols per sub-frame
            N_SC = 12  # number of subcarrier per RB
            A = 4  # number of Antennas
            BTW = 32  # number of I Q bits
            l = [i for i in range(n_slices)]  # midhaul link capacity needed by each slice btw CU and DU
            for i in range(n_slices):
                if i < n_eMBB:
                    l[i] = 14000 * 1502 / (1509 * 1000000)  # Gbps
                elif i < (n_eMBB+n_uRLLC):
                    l[i] = 3752 * 1502 / (1509 * 1000000)  # Gbps
                else:
                    l[i] = 3752 * 1502 / (1509 * 1000000)  # Gbps

            ############Model link capacity C between servers
            Capacity = np.ones((n_servers,n_servers))  # link capacity available between 2 servers
            for s in range(n_servers):
                for s1 in range(n_servers):
                    if s1 != s:
                        if s < n_servers_edge:
                            if s1 < n_servers_edge:
                                Capacity[s][s1] = random.randint(1,10)  # Gbps edge-edge (max range = 10Gbps)
                            else:
                                Capacity[s][s1] = random.randint(10,20)  # edge-regional (max range = 20 Gbps)
                        else:
                            if s1 < n_servers_edge:
                                Capacity[s][s1] = random.randint(10,20)  # reg-edge
                    else:  # same server=> link capacity very high number suppose 1000(exchange of information in the same server)
                        Capacity[s][s1] = 1000
            if scenario == 1:
                cc = m1.add_constraints(z_edge[i,s,s1] <= ((theta_DU_edge[i,s]+theta_CU_edge[i,s1]) / 2) for i in range(n_slices) for s,s1 in S)
                cccc = m1.add_constraints(z_edge[i,s,s1] >= (theta_DU_edge[i,s]+theta_CU_edge[i,s1]-1) for i in range(n_slices) for s,s1 in S)
                c4 = m1.add_constraints(m1.sum((z_edge[i,s,s1]+z_edge[i,s1,s]) * l[i] for i in range(n_slices)) <= Capacity[s][s1] for s,s1 in S if s != s1)
                ccc = m1.add_constraints(m1.sum(z_edge[i,s,s1] * l[i] for i in range(n_slices)) <= Capacity[s][s1] for s,s1 in S if s == s1)

            if scenario == 2:
                cc = m.add_constraints(z_optimal[i,s,s1] <= ((theta_DU_optimal[i,s]+theta_CU_optimal[i,s1]) / 2) for i in range(n_slices) for s,s1 in S)
                cccc = m.add_constraints(z_optimal[i,s,s1] >= (theta_DU_optimal[i,s]+theta_CU_optimal[i,s1]-1) for i in range(n_slices) for s,s1 in S)
                c4 = m.add_constraints(m.sum((z_optimal[i,s,s1]+z_optimal[i,s1,s]) * l[i] for i in range(n_slices)) <= Capacity[s][s1] for s,s1 in S if s != s1)
                ccc = m.add_constraints(m.sum(z_optimal[i,s,s1] * l[i] for i in range(n_slices)) <= Capacity[s][s1] for s,s1 in S if s == s1)

            if scenario == 3:
                cc = m2.add_constraints(z_regional[i,s,s1] <= ((theta_DU_regional[i,s]+theta_CU_regional[i,s1]) / 2) for i in range(n_slices) for s,s1 in S)
                cccc = m2.add_constraints(z_regional[i,s,s1] >= (theta_DU_regional[i,s]+theta_CU_regional[i,s1]-1) for i in range(n_slices) for s,s1 in S)
                c4 = m2.add_constraints(m2.sum((z_regional[i,s,s1]+z_regional[i,s1,s]) * l[i] for i in range(n_slices)) <= Capacity[s][s1] for s,s1 in S if s != s1)
                ccc = m2.add_constraints(m2.sum(z_regional[i,s,s1] * l[i] for i in range(n_slices)) <= Capacity[s][s1] for s,s1 in S if s == s1)

            #################c5_server_capacity constraint
            R = np.ones(n_servers)
            R_DU = np.ones(n_slices)
            R_CU = np.ones(n_slices)
            alpha_DU = 0.4  # Scaling factor for the DU functionalities over all funct
            alpha_CU = 0.1  #
            Code_rate = [438,466,517,567,616,666,719,772,822,873,910,948]
            spectral_eff = [2.5664,2.7305,3.0293,3.3223,3.6094,3.9023,4.2129,4.5234,4.8164,5.1152,5.3320,5.5547]
            C = []
            S_E = []

            for i in range(n_slices):
                MCS = random.randint(17,28)
                if scenario == 2:
                    MMM.append(MCS)
                C.append(Code_rate[MCS-17] / 1024)
                S_E.append(spectral_eff[MCS-17])
            M = 6  # modulation bits log2(64)
            L = 2  # number of MIMO layers
            A = 4  # number of Antennas
            ###total computetional power demand by all experiment can be 1800GOPS

            for i in range(n_servers):
                if i < n_servers_edge:
                    R[i] = random.randint(100,200)  # GOPS available at edge cloud servers
                else:
                    R[i] = random.randint(1000,2000)  # GOPS available at regional cloud servers

            for i in range(n_slices):
                if i < n_eMBB:
                    R_DU[i] = alpha_DU * (3 * A+A ** 2+M * C[i] * L / 3) * RB_eMBB[i] / 10  # GOPS needed by DU functionalities for slice eMBB
                elif i < (n_eMBB+n_uRLLC):
                    R_DU[i] = alpha_DU * (3 * A+A ** 2+M * C[i] * L / 3) * RB_uRLLC[i-n_eMBB] / 10
                else:
                    R_DU[i] = alpha_DU * (3 * A+A ** 2+M * C[i] * L / 3) * RB_MTC[i-n_eMBB-n_uRLLC] / 10

            for i in range(n_slices):
                if i < n_eMBB:
                    R_CU[i] = alpha_CU * (3 * A+A ** 2+M * C[i] * L / 3) * RB_eMBB[i] / 10  # GOPS needed by CU functionalities for slice eMBB
                elif i < (n_eMBB+n_uRLLC):
                    R_CU[i] = alpha_CU * (3 * A+A ** 2+M * C[i] * L / 3) * RB_uRLLC[i-n_eMBB] / 10
                else:
                    R_CU[i] = alpha_CU * (3 * A+A ** 2+M * C[i] * L / 3) * RB_MTC[i-n_eMBB-n_uRLLC] / 10

            if scenario == 1:
                c5 = m1.add_constraints((m1.sum((theta_DU_edge[i,s] * R_DU[i]+theta_CU_edge[i,s] * R_CU[i]) for i in range(n_slices)) <= R[s] for s in range(n_servers)))
            if scenario == 2:
                c5 = m.add_constraints((m.sum((theta_DU_optimal[i,s] * R_DU[i]+theta_CU_optimal[i,s] * R_CU[i]) for i in range(n_slices)) <= R[s] for s in range(n_servers)))
            if scenario == 3:
                c5 = m2.add_constraints((m2.sum((theta_DU_regional[i,s] * R_DU[i]+theta_CU_regional[i,s] * R_CU[i]) for i in range(n_slices)) <= R[s] for s in range(n_servers)))

            # c6 Latency constraint
            delta = np.zeros((n_servers,n_servers))
            delta_max = np.ones(n_slices)
            for i in range(n_slices):
                if i < n_eMBB:
                    delta_max[i] = 500  # random.randint(4,10)*1000 #micro seconds
                elif i < (n_eMBB+n_uRLLC):
                    delta_max[i] = random.randint(100,300)  # 100 #random.randint(1,2)*1000
                else:
                    delta_max[i] = 1000  # random.randint(2,4)*1000

            for s in range(n_servers):
                for s1 in range(n_servers):
                    if s < n_servers_edge:
                        if s1 < n_servers_edge:
                            delta[s][s1] = 5 * math.sqrt((loc_x_edge[s]-loc_x_edge[s1]) ** 2+(
                                    loc_y_edge[s]-loc_y_edge[s1]) ** 2)  # random.randint(50,100)  # us edge-edge (5us/km => 10-20 km betwwen edge-edge server)
                        else:
                            delta[s][s1] = 5 * math.sqrt((loc_x_edge[s]-loc_x_reg[s1-n_servers_edge]) ** 2+(
                                    loc_y_edge[s]-loc_y_reg[s1-n_servers_edge]) ** 2)  # random.randint(200,400) #edge-regional(5us/km => 40-80 km betwwen edge-edge server)
                    else:
                        if s1 < n_servers_edge:
                            delta[s][s1] = 5 * math.sqrt((loc_x_edge[s1]-loc_x_reg[s-n_servers_edge]) ** 2+(loc_y_edge[s1]-loc_y_reg[s-n_servers_edge]) ** 2)  # random.randint(200,400) #reg-edge
                    #             else:
                    #                 delta[s][s1]= random.randint(1000,2000) #reg-reg
                    if s == s1:
                        delta[s][s1] = 0

            if scenario == 1:
                c6 = m1.add_constraints(delta[s,s1] * (theta_CU_edge[i,s]+theta_DU_edge[i,s1]-1) <= delta_max[i] for i in range(n_slices) for s,s1 in
                S)  # c6=m1.add_constraints(delta[s,s1]*(theta_DU_edge[i,s])<=delta_max[i] for i in range(n_slices) for s,s1 in S)
            if scenario == 2:
                c6 = m.add_constraints(delta[s,s1] * (theta_CU_optimal[i,s]+theta_DU_optimal[i,s1]-1) <= delta_max[i] for i in range(n_slices) for s,s1 in
                S)  # c6=m.add_constraints(delta[s,s1]*(theta_DU_optimal[i,s])<=delta_max[i] for i in range(n_slices) for s,s1 in S)
            if scenario == 3:
                c6 = m2.add_constraints(delta[s,s1] * (theta_CU_regional[i,s]+theta_DU_regional[i,s1]-1) <= delta_max[i] for i in range(n_slices) for s,s1 in
                S)  # c6=m.add_constraints(delta[s,s1]*(theta_DU_optimal[i,s])<=delta_max[i] for i in range(n_slices) for s,s1 in S)

            nearest_edge = np.ones(Nb_BS)
            for b in range(Nb_BS):
                minimum = -1
                nearest = 0
                for s in range(n_servers_edge):
                    distance = math.sqrt((loc_x_BS[b]-loc_x_edge[s]) ** 2+(loc_y_BS[b]-loc_y_edge[s]) ** 2)
                    if distance < minimum:
                        minimum = distance
                        nearest = s
                nearest_edge[b] = s  # save for each bs the nearest edge server

            C_F = np.ones((n_slices,n_servers))  # link capacity available between 2 servers
            for i in range(n_slices):
                for s in range(n_servers):
                    if s < n_servers_edge:
                        if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                            C_F[i][s] = 1 / math.sqrt((loc_x_BS[0]-loc_x_edge[s]) ** 2+(loc_y_BS[0]-loc_y_edge[s]) ** 2)  # edge
                        elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                            C_F[i][s] = 1 / math.sqrt((loc_x_BS[1]-loc_x_edge[s]) ** 2+(loc_y_BS[1]-loc_y_edge[s]) ** 2)  # edge
                        elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                            C_F[i][s] = 1 / math.sqrt((loc_x_BS[2]-loc_x_edge[s]) ** 2+(loc_y_BS[2]-loc_y_edge[s]) ** 2)  # edge
                        elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                            C_F[i][s] = 1 / math.sqrt((loc_x_BS[3]-loc_x_edge[s]) ** 2+(loc_y_BS[3]-loc_y_edge[s]) ** 2)  # edge
                    else:
                        C_F[i][s] = 1  # regional

            # C_F = np.ones((n_slices, n_servers))  # link capacity available between 2 servers
            # for i in range(n_slices):
            #     for s in range(n_servers):
            #         if s < n_servers_edge:
            #             C_F[i][s] = 1  # edge other than the nearest
            #             if loc_x[i] < 0.500 and loc_y[i] < 0.500:
            #                 if s == nearest_edge[0]:
            #                     C_F[i][s] = 3  # edge
            #             elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
            #                 if s == nearest_edge[1]:
            #                     C_F[i][s] = 3  # edge
            #             elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
            #                 if s == nearest_edge[2]:
            #                     C_F[i][s] = 3  # edge
            #             elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
            #                 if s == nearest_edge[3]:
            #                     C_F[i][s] = 3  # edge
            #         else:
            #             C_F[i][s] = 5  # regional

            priority_matrix = np.ones(n_slices)
            for i in range(n_slices):
                if i < n_eMBB:
                    priority_matrix[i] = 10  # to maximize the throughput we give higher priority for eMBB users
                elif i < n_eMBB+n_uRLLC:
                    priority_matrix[i] = 10  # 3
                else:
                    priority_matrix[i] = 1

            ###OBJECTIVEEEEE

            if scenario == 0:  # random
                # user_satisfaction = m0.sum(theta_CU_random[i, s]* priority_matrix[i] for i in range(n_slices) for s in range(n_servers))
                user_satisfaction = m0.sum(theta_DU_random[i,s] * priority_matrix[i]+theta_CU_random[i,s] * priority_matrix[i] for i in range(n_slices) for s in range(n_servers))

                m0.maximize(user_satisfaction)
                solution = m0.solve()
            elif scenario == 1:
                # user_satisfaction= m1.sum(theta_CU_edge[i,s] for i in range(n_slices) for s in range(n_servers))
                # user_satisfaction = m1.sum(C_F[i, s] * theta_CU_edge[i, s] + C_F[i, s] * theta_DU_edge[i, s] for i in range(n_slices) for s in range(n_servers))
                user_satisfaction = m1.sum(C_F[i,s] * theta_CU_edge[i,s] * priority_matrix[i]+C_F[i,s] * theta_DU_edge[i,s] * priority_matrix[i] for i in range(n_slices) for s in range(n_servers))

                m1.maximize(user_satisfaction)
                solution = m1.solve()
            elif scenario == 2:  # optimal
                # user_satisfaction = m.sum(C_F[i, s] * theta_CU_optimal[i, s] + C_F[i, s] * theta_DU_optimal[i, s] for i in range(n_slices) for s in range(n_servers))
                user_satisfaction = m.sum(
                    C_F[i,s] * theta_CU_optimal[i,s] * priority_matrix[i]+C_F[i,s] * theta_DU_optimal[i,s] * priority_matrix[i] for i in range(n_slices) for s in range(n_servers))
                # user_satisfaction= m.sum(C_F[i,s]*theta_CU_optimal[i,s]  for i in range(n_slices) for s in range(n_servers))
                # user_satisfaction= m.sum(theta_CU_optimal[i,s] for i in range(n_slices) for s in range(n_servers))
                m.maximize(user_satisfaction)
                t = time.time()
                solution = m.solve()
                time_MILP.append((time.time()-t) * 1000)
                # print("time_MILP",time_MILP)
                objectivee = m.objective_value
            elif scenario == 3:  # regional
                user_satisfaction = m2.sum(
                    C_F[i,s] * theta_CU_regional[i,s] * priority_matrix[i]+C_F[i,s] * theta_DU_regional[i,s] * priority_matrix[i] for i in range(n_slices) for s in range(n_servers))
                # user_satisfaction= m2.sum(C_F[i,s]*theta_CU_regional[i,s]  for i in range(n_slices) for s in range(n_servers))

                m2.maximize(user_satisfaction)
                solution = m2.solve()

            Cost_matrix = np.ones(n_servers)  # link capacity available between 2 servers

            for s in range(n_servers):
                if s < n_servers_edge:
                    Cost_matrix[s] = 1.59  # edge
                else:
                    Cost_matrix[s] = 0.5  # regional

            solution_theta_DU = np.ones((n_slices,n_servers))
            for i in range(n_slices):
                for j in range(n_servers):
                    if scenario == 0:
                        solution_theta_DU[i][j] = (m0.solution.get_value("theta_DU_{}_{}".format(i,j)))
                    if scenario == 1:
                        solution_theta_DU[i][j] = (m1.solution.get_value("theta_DU_{}_{}".format(i,j)))
                    if scenario == 2:
                        solution_theta_DU[i][j] = (m.solution.get_value("theta_DU_{}_{}".format(i,j)))
                    if scenario == 3:
                        solution_theta_DU[i][j] = (m2.solution.get_value("theta_DU_{}_{}".format(i,j)))

            solution_theta_CU = np.ones((n_slices,n_servers))
            for i in range(n_slices):
                for j in range(n_servers):
                    if scenario == 0:
                        solution_theta_CU[i][j] = (m0.solution.get_value("theta_CU_{}_{}".format(i,j)))
                    if scenario == 1:
                        solution_theta_CU[i][j] = (m1.solution.get_value("theta_CU_{}_{}".format(i,j)))

                    if scenario == 2:
                        solution_theta_CU[i][j] = (m.solution.get_value("theta_CU_{}_{}".format(i,j)))
                    if scenario == 3:
                        solution_theta_CU[i][j] = (m2.solution.get_value("theta_CU_{}_{}".format(i,j)))

            if scenario == 0:  # for random scenario

                for i in range(n_slices):
                    for s in range(n_servers):
                        for s1 in range(n_servers):
                            if delta[s,s1] * (solution_theta_CU[i,s] * solution_theta_DU[i,s1]) > delta_max[i]:
                                # print("delta violated")
                                solution_theta_DU[i] = solution_theta_CU[i] = 0

                rem_server_cap = R  # remaining server capacity
                for s in range(n_servers):
                    for i in range(n_slices):
                        rem_server_cap[s] -= solution_theta_DU[i,s] * R_DU[i]+solution_theta_CU[i,s] * R_CU[i]
                        if rem_server_cap[s] <= 0:
                            # print('no enough RBs')
                            # if server capacity is not satisfied
                            solution_theta_DU[i,s] = solution_theta_CU[i,s] = 0  # print(solution_theta_CU)  # print(solution_theta_DU)

            ##throughput calculation
            throughput = 0
            for i in range(n_slices):
                for s in range(n_servers):
                    # throughput += N_SC*N_sym*RB_slices[i]*S_E[i]*L*1600*solution_theta_CU[i,s]/1000000##in Mbps
                    throughput += L * M * C[i] * N_SC * N_sym * RB_slices[i] * solution_theta_CU[i,s] * (1-0.14) / 1000  ##in Mbps

            n_admitted_eMBB = sum(solution_theta_CU[i,s] for i in range(n_eMBB) for s in range(n_servers))
            n_admitted_uRLLC = sum(solution_theta_CU[i,s] for i in range(n_eMBB,n_eMBB+n_uRLLC) for s in range(n_servers))
            n_admitted_mMTC = sum(solution_theta_CU[i,s] for i in range(n_eMBB+n_uRLLC,n_slices) for s in range(n_servers))

            ######fairness index of admitted slices
            s = n_admitted_eMBB / n_eMBB+n_admitted_mMTC / n_MTC+n_admitted_uRLLC / n_uRLLC
            x = (n_admitted_eMBB / n_eMBB) ** 2+(n_admitted_mMTC / n_MTC) ** 2+(n_admitted_uRLLC / n_uRLLC) ** 2

            fairness = (s * s) / (3 * x)

            linkk_util = np.zeros((n_servers,n_servers))
            GOPS_util = np.zeros(n_servers)
            # just targeting to collect GOPS load for n=140 users

            for s,s1 in S:
                linkk_util[s][s1] = sum(l[i] * solution_theta_CU[i,s] * solution_theta_DU[i,s1] for i in range(n_slices)) / Capacity[s][s1] * 100
            for s in range(n_servers):
                GOPS_util[s] = ((sum(R_CU[i] * solution_theta_CU[i,s] for i in range(n_slices))+sum(R_DU[i] * solution_theta_DU[i,s] for i in range(n_slices))) / R[s]) * 100

            # objectivee=np.append(objectivee,m.objective_value)
            Nb_servers_edge_CU = np.ones(3)
            Nb_servers_reg_CU = np.ones(3)
            Nb_servers_edge_CU = [sum(solution_theta_CU[i,s] for i in range(n_eMBB) for s in range(n_servers_edge)),

                sum(solution_theta_CU[i,s] for i in range(n_eMBB,n_eMBB+n_uRLLC) for s in range(n_servers_edge)),
                sum(solution_theta_CU[i,s] for i in range(n_eMBB+n_uRLLC,n_slices) for s in range(n_servers_edge))]
            Nb_servers_reg_CU = [sum(solution_theta_CU[i,s] for i in range(n_eMBB) for s in range(n_servers_edge,n_servers)),
                sum(solution_theta_CU[i,s] for i in range(n_eMBB,n_eMBB+n_uRLLC) for s in range(n_servers_edge,n_servers)),
                sum(solution_theta_CU[i,s] for i in range(n_eMBB+n_uRLLC,n_slices) for s in range(n_servers_edge,n_servers))]

            CC = sum(Cost_matrix[s] * R_CU[i] * solution_theta_CU[i,s] for i in range(n_slices) for s in range(n_servers))
            # we take into consideration the cost of deployment of just CU because the DU is always at the edge.
            # CC = sum(Cost_matrix[s]*R_CU[i]*solution_theta_CU[i,s] for i in range(n_slices) for s in range(n_servers))+sum(Cost_matrix[s]*R_DU[i]*solution_theta_DU[i,s] for i in range(n_slices) for s in range(n_servers))

            if scenario == 0:

                link_utilization_random.append(linkk_util)
                # GOPS_utilization_random.append(GOPS_util)
                # print(GOPS_utilization_random)

                total_throughput_random.append(throughput)

                ratio_CU_edge_eMbb_random.append(Nb_servers_edge_CU[0] / n_admitted_eMBB * 100 if n_admitted_eMBB > 0 else 0)
                ratio_CU_edge_uRLLC_random.append(Nb_servers_edge_CU[1] / n_admitted_uRLLC * 100 if n_admitted_uRLLC > 0 else 0)
                ratio_CU_edge_MTC_random.append(Nb_servers_edge_CU[2] / n_admitted_mMTC * 100 if n_admitted_mMTC > 0 else 0)

                ratio_CU_reg_eMbb_random.append(Nb_servers_reg_CU[0] / n_admitted_eMBB * 100 if n_admitted_eMBB > 0 else 0)
                ratio_CU_reg_uRLLC_random.append(Nb_servers_reg_CU[1] / n_admitted_uRLLC * 100 if n_admitted_uRLLC > 0 else 0)
                ratio_CU_reg_MTC_random.append(Nb_servers_reg_CU[2] / n_admitted_mMTC * 100 if n_admitted_mMTC > 0 else 0)

                ratio_admitted_random.append(sum(solution_theta_CU[i,s] for i in range(n_slices) for s in range(n_servers)) / n_slices * 100)
                ratio_admitted_eMBB_random.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB) for s in range(n_servers)) / n_eMBB * 100)
                ratio_admitted_uRLLC_random.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB,n_eMBB+n_uRLLC) for s in range(n_servers)) / n_uRLLC * 100)  # if n_uRLLC>0 else 0)
                ratio_admitted_mMTC_random.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB+n_uRLLC,n_slices) for s in range(n_servers)) / n_MTC * 100)

                cost_random.append(CC)

                fairness_random.append(fairness)


            elif scenario == 1:

                link_utilization_edge.append(linkk_util)
                # GOPS_utilization_edge.append(GOPS_util)

                total_throughput_edge.append(throughput)
                ratio_CU_edge_eMbb_edge.append(Nb_servers_edge_CU[0] / n_admitted_eMBB * 100 if n_admitted_eMBB > 0 else 0)
                ratio_CU_edge_uRLLC_edge.append(Nb_servers_edge_CU[1] / n_admitted_uRLLC * 100 if n_admitted_uRLLC > 0 else 0)
                ratio_CU_edge_MTC_edge.append(Nb_servers_edge_CU[2] / n_admitted_mMTC * 100 if n_admitted_mMTC > 0 else 0)

                ratio_CU_reg_eMbb_edge.append(Nb_servers_reg_CU[0] / n_admitted_eMBB * 100 if n_admitted_eMBB > 0 else 0)
                ratio_CU_reg_uRLLC_edge.append(Nb_servers_reg_CU[1] / n_admitted_uRLLC * 100 if n_admitted_uRLLC > 0 else 0)
                ratio_CU_reg_MTC_edge.append(Nb_servers_reg_CU[2] / n_admitted_mMTC * 100 if n_admitted_uRLLC > 0 else 0)

                ratio_admitted_edge.append(sum(solution_theta_CU[i,s] for i in range(n_slices) for s in range(n_servers)) / n_slices * 100)
                ratio_admitted_eMBB_edge.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB) for s in range(n_servers)) / n_eMBB * 100)
                ratio_admitted_uRLLC_edge.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB,n_eMBB+n_uRLLC) for s in range(n_servers)) / n_uRLLC * 100)
                ratio_admitted_mMTC_edge.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB+n_uRLLC,n_slices) for s in range(n_servers)) / n_MTC * 100)

                cost_edge.append(CC)

                fairness_edge.append(fairness)

            elif scenario == 2:

                link_utilization_optimal.append(linkk_util)
                # GOPS_utilization_optimal.append(GOPS_util)
                total_throughput_optimal.append(throughput)

                ratio_CU_edge_eMbb_optimal.append(Nb_servers_edge_CU[0] / n_admitted_eMBB * 100 if n_admitted_eMBB > 0 else 0)
                ratio_CU_edge_uRLLC_optimal.append(Nb_servers_edge_CU[1] / n_admitted_uRLLC * 100 if n_admitted_uRLLC > 0 else 0)
                ratio_CU_edge_MTC_optimal.append(Nb_servers_edge_CU[2] / n_admitted_mMTC * 100 if n_admitted_mMTC > 0 else 0)

                ratio_CU_reg_eMbb_optimal.append(Nb_servers_reg_CU[0] / n_admitted_eMBB * 100 if n_admitted_eMBB > 0 else 0)
                ratio_CU_reg_uRLLC_optimal.append(Nb_servers_reg_CU[1] / n_admitted_uRLLC * 100 if n_admitted_uRLLC > 0 else 0)
                ratio_CU_reg_MTC_optimal.append(Nb_servers_reg_CU[2] / n_admitted_mMTC * 100 if n_admitted_mMTC > 0 else 0)

                ratio_admitted_optimal.append(sum(solution_theta_CU[i,s] for i in range(n_slices) for s in range(n_servers)) / n_slices * 100)
                ratio_admitted_eMBB_optimal.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB) for s in range(n_servers)) / n_eMBB * 100)
                ratio_admitted_uRLLC_optimal.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB,n_eMBB+n_uRLLC) for s in range(n_servers)) / n_uRLLC * 100)
                ratio_admitted_mMTC_optimal.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB+n_uRLLC,n_slices) for s in range(n_servers)) / n_MTC * 100)

                cost_optimal.append(CC)

                fairness_optimal.append(fairness)

                GOPS_utilization_1.append(GOPS_util[0])
                GOPS_utilization_2.append(GOPS_util[1])
                GOPS_utilization_3.append(GOPS_util[2])
                GOPS_utilization_4.append(GOPS_util[3])
                # the utilization of all edge servers GOPS resources
                GOPS_utilization_all_edge.append((sum(R_CU[i] * solution_theta_CU[i,s]+R_DU[i] * solution_theta_DU[i,s] for i in range(n_slices) for s in range(n_servers_edge)) / sum(
                    R[j] for j in range(n_servers_edge))) * 100)
                # the dataframe row are users and columns are the features

                nearest_edge = np.ones(Nb_BS)
                relative_dist_BS_server = []
                for b in range(Nb_BS):
                    for s in range(n_servers):
                        if s < n_servers_edge:
                            relative_dist_BS_server.append(math.sqrt((loc_x_BS[b]-loc_x_edge[s]) ** 2+(loc_y_BS[b]-loc_y_edge[s]) ** 2))
                        else:
                            relative_dist_BS_server.append(math.sqrt((loc_x_BS[b]-loc_x_reg[s-n_servers_edge]) ** 2+(loc_y_BS[b]-loc_y_reg[s-n_servers_edge]) ** 2))

                relative_dist_server_server = []
                loc_x_servers = loc_x_edge+loc_x_reg
                loc_y_servers = loc_y_edge+loc_y_reg

                relative_dist_server_server.append(math.sqrt((loc_x_servers[0]-loc_x_servers[1]) ** 2+(loc_y_servers[0]-loc_y_servers[1]) ** 2))
                relative_dist_server_server.append(math.sqrt((loc_x_servers[0]-loc_x_servers[2]) ** 2+(loc_y_servers[0]-loc_y_servers[2]) ** 2))
                relative_dist_server_server.append(math.sqrt((loc_x_servers[0]-loc_x_servers[3]) ** 2+(loc_y_servers[0]-loc_y_servers[3]) ** 2))
                relative_dist_server_server.append(math.sqrt((loc_x_servers[1]-loc_x_servers[2]) ** 2+(loc_y_servers[1]-loc_y_servers[2]) ** 2))
                relative_dist_server_server.append(math.sqrt((loc_x_servers[1]-loc_x_servers[3]) ** 2+(loc_y_servers[1]-loc_y_servers[3]) ** 2))
                relative_dist_server_server.append(math.sqrt((loc_x_servers[2]-loc_x_servers[3]) ** 2+(loc_y_servers[2]-loc_y_servers[3]) ** 2))

                BS_association = []
                for i in range(n_slices):
                    if loc_x[i] < 0.500 and loc_y[i] < 0.500:  # this user is ascociated to the first BS
                        BS_association.append(0)
                    elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                        BS_association.append(1)
                    elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                        BS_association.append(2)
                    elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                        BS_association.append(3)

                GOPS_av = []
                LOC = []
                LOC_new = []
                Capacity_av = []
                delta_av = []
                # GOPS_av = np.tile(R, (n_slices, 1))
                # CCC = Capacity.flatten()  # from 2d array to 1d
                # ddd= delta.flatten()
                loc_x_EDGE = []
                loc_y_EDGE = []
                loc_x_REG = []
                loc_y_REG = []
                total_number_users = []
                slice_type = []
                for i in range(n_slices):
                    GOPS_av.append(R)
                    # Capacity_av.append(CCC)
                    # delta_av.append(ddd)
                    loc_x_EDGE.append(loc_x_edge)
                    loc_y_EDGE.append(loc_y_edge)
                    loc_x_REG.append(loc_x_reg)
                    loc_y_REG.append(loc_y_reg)
                    ind_CU = ind_DU = 0
                    # print(solution_theta_CU)
                    temp = np.zeros(12)  # 12 elements vector for the location

                    total_number_users.append(NN)

                    if (1 in solution_theta_CU[i]) and (1 in solution_theta_DU[i]):
                        ind_CU = np.where(solution_theta_CU[i] == 1)[0][0]
                        ind_DU = np.where(solution_theta_DU[i] == 1)[0][0]
                        # temp[ind_DU+ind_CU*3]=1
                        # LOC.append(temp)
                        LOC.append([ind_CU+1,ind_DU+1])

                    else:
                        # LOC.append(np.zeros(12))
                        LOC.append(np.zeros(2))

                for ll in LOC:
                    if ll[0] != 0 and ll[1] != 0:
                        LOC_new.append((ll[1]-1+(ll[0]-1) * 3+1))
                    else:
                        LOC_new.append(0)

                slice_type = np.concatenate(([1] * n_eMBB,[2] * n_uRLLC,[3] * n_MTC))

                C_F = np.ones((n_slices,n_servers))  # link capacity available between 2 servers

                for i in range(n_slices):
                    for s in range(n_servers):
                        if s < n_servers_edge:
                            if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                                C_F[i][s] = 1 / math.sqrt((loc_x_BS[0]-loc_x_edge[s]) ** 2+(loc_y_BS[0]-loc_y_edge[s]) ** 2)  # edge
                            elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                                C_F[i][s] = 1 / math.sqrt((loc_x_BS[1]-loc_x_edge[s]) ** 2+(loc_y_BS[1]-loc_y_edge[s]) ** 2)  # edge
                            elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                                C_F[i][s] = 1 / math.sqrt((loc_x_BS[2]-loc_x_edge[s]) ** 2+(loc_y_BS[2]-loc_y_edge[s]) ** 2)  # edge
                            elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                                C_F[i][s] = 1 / math.sqrt((loc_x_BS[3]-loc_x_edge[s]) ** 2+(loc_y_BS[3]-loc_y_edge[s]) ** 2)  # edge
                        else:
                            C_F[i][s] = 1  # regional

                priority_matrix = np.ones(n_slices)
                for i in range(n_slices):
                    if i < n_eMBB:
                        priority_matrix[i] = 10  # to maximize the throughput we give higher priority for eMBB users
                    elif i < n_eMBB+n_uRLLC:
                        priority_matrix[i] = 10  # 3
                    else:
                        priority_matrix[i] = 1

                df1 = pd.DataFrame({'user_loc_x': loc_x,'user_loc_y': loc_y,# 'RU_loc':[],
                                       'loc_edge_x_1': [loc_x_EDGE[0][0]] * n_slices,'loc_edge_x_2': [loc_x_EDGE[0][1]] * n_slices,'loc_edge_x_3': [loc_x_EDGE[0][2]] * n_slices,
                                       'loc_edge_y_1': [loc_y_EDGE[0][0]] * n_slices,'loc_edge_y_2': [loc_y_EDGE[0][1]] * n_slices,'loc_edge_y_3': [loc_y_EDGE[0][2]] * n_slices,
                                       'loc_reg_x': [loc_x_REG[0][0]] * n_slices,'loc_reg_y': [loc_y_REG[0][0]] * n_slices,

                                       'USer_association_BS': BS_association,

                                       'rel_dist_Server_BS_0': relative_dist_BS_server[0],'rel_dist_Server_BS_1': relative_dist_BS_server[1],'rel_dist_Server_BS_2': relative_dist_BS_server[2],
                                       'rel_dist_Server_BS_3': relative_dist_BS_server[3],'rel_dist_Server_BS_4': relative_dist_BS_server[4],'rel_dist_Server_BS_5': relative_dist_BS_server[5],
                                       'rel_dist_Server_BS_6': relative_dist_BS_server[6],'rel_dist_Server_BS_7': relative_dist_BS_server[7],'rel_dist_Server_BS_8': relative_dist_BS_server[8],
                                       'rel_dist_Server_BS_9': relative_dist_BS_server[9],'rel_dist_Server_BS_10': relative_dist_BS_server[10],'rel_dist_Server_BS_11': relative_dist_BS_server[11],
                                       'rel_dist_Server_BS_12': relative_dist_BS_server[12],'rel_dist_Server_BS_13': relative_dist_BS_server[13],'rel_dist_Server_BS_14': relative_dist_BS_server[14],
                                       'rel_dist_Server_BS_15': relative_dist_BS_server[15],

                                       'rel_dist_ser_ser_0': relative_dist_server_server[0],'rel_dist_ser_ser_1': relative_dist_server_server[1],'rel_dist_ser_ser_2': relative_dist_server_server[2],
                                       'rel_dist_ser_ser_3': relative_dist_server_server[3],'rel_dist_ser_ser_4': relative_dist_server_server[4],'rel_dist_ser_ser_5': relative_dist_server_server[5],

                                       'RB': RB_slices,'MCS': MMM,# 'slice': [],
                                       'GOPS_available_1': [GOPS_av[0][0]] * n_slices,'GOPS_available_2': [GOPS_av[0][1]] * n_slices,'GOPS_available_3': [GOPS_av[0][2]] * n_slices,
                                       'GOPS_available_4': [GOPS_av[0][3]] * n_slices,'GOPS_required_CU': R_CU,

                                       'GOPS_required_DU': R_DU,

                                       # 'link_cap_av': Capacity_av,
                                       # 'link_required': l,
                                       # 'link_latency': delta_av, #can be learned from the locations of server and user, i suppose no need for it
                                       'link_latency_0_0': [delta[0][0]] * n_slices,'link_latency_0_1': [delta[0][1]] * n_slices,'link_latency_0_2': [delta[0][2]] * n_slices,
                                       'link_latency_0_3': [delta[0][3]] * n_slices,# 'link_latency_1_0': [delta[1][0]]*n_slices,
                                       'link_latency_1_1': [delta[1][1]] * n_slices,'link_latency_1_2': [delta[1][2]] * n_slices,'link_latency_1_3': [delta[1][3]] * n_slices,
                                       'link_latency_2_2': [delta[2][2]] * n_slices,'link_latency_2_3': [delta[2][3]] * n_slices,'link_latency_3_3': [delta[3][3]] * n_slices,
                                       'TOTAL_GOPS_required_CU': [np.sum(R_CU)] * n_slices,'TOTAL_GOPS_required_DU': [np.sum(R_DU)] * n_slices,'max_latency': delta_max,
                                       'total_num_users': total_number_users,'slice_type': slice_type,'priority': priority_matrix,'C_F_s_0': C_F[:,0],'C_F_s_1': C_F[:,1],'C_F_s_2': C_F[:,2],
                                       'C_F_s_3': C_F[:,3],# 'objective_value':objectivee,
                                       'LOCATION': LOC_new})

                if NN < 140:
                    # Desired number of rows
                    desired_rows = 140

                    # Calculate the difference between the current number of rows and the desired number
                    difference = desired_rows-len(df1)

                    # Create a new dataframe with zeros
                    zeros_df = pd.DataFrame(0,index=range(difference),columns=df1.columns)
                    # zeros_df=zeros_df.loc['LOCATION']

                    # Concatenate the original dataframe with the zeros dataframe to reach the desired number of rows
                    df_with_zeros = pd.concat([df1,zeros_df],axis=0)

                    df1 = df_with_zeros

                dataset = dataset.append(df1,ignore_index=True)  # dataset_collection[Nb_exp]=df1



            elif scenario == 3:

                link_utilization_regional.append(linkk_util)
                # GOPS_utilization_regional=np.append(GOPS_utilization_regional,GOPS_util)
                total_throughput_regional.append(throughput)

                ratio_CU_edge_eMbb_regional.append(Nb_servers_edge_CU[0] / n_admitted_eMBB * 100 if n_admitted_eMBB > 0 else 0)
                ratio_CU_edge_uRLLC_regional.append(Nb_servers_edge_CU[1] / n_admitted_uRLLC * 100 if n_admitted_uRLLC > 0 else 0)
                ratio_CU_edge_MTC_regional.append(Nb_servers_edge_CU[2] / n_admitted_mMTC * 100 if n_admitted_mMTC > 0 else 0)

                ratio_CU_reg_eMbb_regional.append(Nb_servers_reg_CU[0] / n_admitted_eMBB * 100 if n_admitted_eMBB > 0 else 0)
                ratio_CU_reg_uRLLC_regional.append(Nb_servers_reg_CU[1] / n_admitted_uRLLC * 100 if n_admitted_uRLLC > 0 else 0)
                ratio_CU_reg_MTC_regional.append(Nb_servers_reg_CU[2] / n_admitted_mMTC * 100 if n_admitted_mMTC > 0 else 0)

                ratio_admitted_regional.append(sum(solution_theta_CU[i,s] for i in range(n_slices) for s in range(n_servers)) / n_slices * 100)
                ratio_admitted_eMBB_regional.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB) for s in range(n_servers)) / n_eMBB * 100)
                ratio_admitted_uRLLC_regional.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB,n_eMBB+n_uRLLC) for s in range(n_servers)) / n_uRLLC * 100)
                ratio_admitted_mMTC_regional.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB+n_uRLLC,n_slices) for s in range(n_servers)) / n_MTC * 100)

                cost_regional.append(CC)

                fairness_regional.append(fairness)

    ratio_CU_edge_eMbb_optimal_all_runs.append(ratio_CU_edge_eMbb_optimal)
    ratio_CU_edge_uRLLC_optimal_all_runs.append(ratio_CU_edge_uRLLC_optimal)
    ratio_CU_edge_mMTC_optimal_all_runs.append(ratio_CU_edge_MTC_optimal)

    RATIO_eMBB_random.append(ratio_admitted_eMBB_random)
    RATIO_eMBB_edge.append(ratio_admitted_eMBB_edge)
    RATIO_eMBB_optimal.append(ratio_admitted_eMBB_optimal)
    RATIO_eMBB_regional.append(ratio_admitted_eMBB_regional)

    RATIO_uRLLC_random.append(ratio_admitted_uRLLC_random)
    RATIO_uRLLC_edge.append(ratio_admitted_uRLLC_edge)
    RATIO_uRLLC_optimal.append(ratio_admitted_uRLLC_optimal)
    RATIO_uRLLC_regional.append(ratio_admitted_uRLLC_regional)

    RATIO_mMTC_random.append(ratio_admitted_mMTC_random)
    RATIO_mMTC_edge.append(ratio_admitted_mMTC_edge)
    RATIO_mMTC_optimal.append(ratio_admitted_mMTC_optimal)
    RATIO_mMTC_regional.append(ratio_admitted_mMTC_regional)

    COST_RANDOM.append(cost_random)
    COST_EDGE.append(cost_edge)
    COST_OPTIMAL.append(cost_optimal)
    COST_regional.append(cost_regional)

    fairness_RANDOM.append(fairness_random)
    fairness_EDGE.append(fairness_edge)
    fairness_OPTIMAL.append(fairness_optimal)
    fairness_REGIONAL.append(fairness_regional)

    THROUGHPUT_RANDOM.append(total_throughput_random)
    THROUGHPUT_EDGE.append(total_throughput_edge)
    THROUGHPUT_OPTIMAL.append(total_throughput_optimal)
    THROUGHPUT_regional.append(total_throughput_regional)

    ADMITTANCE_RANDOM.append(ratio_admitted_random)
    ADMITTANCE_EDGE.append(ratio_admitted_edge)
    ADMITTANCE_OPTIMAL.append(ratio_admitted_optimal)
    ADMITTANCE_regional.append(ratio_admitted_regional)
    TIME_MILP.append(time_MILP)

    # GOPS_utilization_random_all_runs.append(GOPS_utilization_random)
    # GOPS_utilization_edge_all_runs.append(GOPS_utilization_edge)
    # GOPS_utilization_optimal_all_runs.append(GOPS_utilization_optimal)
    # GOPS_utilization_regional_all_runs.append(GOPS_utilization_regional)

    RB_utilization_1_all_runs.append(RB_utilization_1)
    RB_utilization_2_all_runs.append(RB_utilization_2)
    RB_utilization_3_all_runs.append(RB_utilization_3)
    RB_utilization_4_all_runs.append(RB_utilization_4)

    GOPS_utilization_1_all_runs.append(GOPS_utilization_1)
    GOPS_utilization_2_all_runs.append(GOPS_utilization_2)
    GOPS_utilization_3_all_runs.append(GOPS_utilization_3)
    GOPS_utilization_4_all_runs.append(GOPS_utilization_4)

    GOPS_utilization_all_edge_all_runs.append(GOPS_utilization_all_edge)

    RB_demand_1_all_runs.append(RB_1_TOTALL)
    RB_demand_2_all_runs.append(RB_2_TOTALL)
    RB_demand_3_all_runs.append(RB_3_TOTALL)
    RB_demand_4_all_runs.append(RB_4_TOTALL)

df = pd.DataFrame(dataset)
# df_one_hot = pd.get_dummies(df, columns=['RB'])
os.makedirs('Desktop',exist_ok=True)
df.to_csv('test_data_milpVSrnn.csv')
df.to_csv(index=True)

I_ratio_random = []
I_ratio_edge = []
I_ratio_optimal = []
I_ratio_regional = []

I_cost_random = []
I_cost_edge = []
I_cost_optimal = []
I_cost_regional = []

I_fairness_random = []
I_fairness_edge = []
I_fairness_optimal = []
I_fairness_regional = []

I_throughput_random = []
I_throughput_edge = []
I_throughput_optimal = []
I_throughput_regional = []

I_ratio_embb_optimal = []
I_ratio_embb_random = []
I_ratio_embb_edge = []
I_ratio_embb_regional = []

I_ratio_uRLLC_optimal = []
I_ratio_uRLLC_random = []
I_ratio_uRLLC_edge = []
I_ratio_uRLLC_regional = []

I_ratio_mMTC_optimal = []
I_ratio_mMTC_random = []
I_ratio_mMTC_edge = []
I_ratio_mMTC_regional = []

I_RB_1 = []
I_RB_2 = []
I_RB_3 = []
I_RB_4 = []

I_RB_1_demand = []
I_RB_2_demand = []
I_RB_3_demand = []
I_RB_4_demand = []

I_GOPS_1 = []
I_GOPS_2 = []
I_GOPS_3 = []
I_GOPS_4 = []

I_GOPS_all_edge = []

I_time_MILP = []
I_time_RNN = []

I_ratio_RNN = []
I_cost_RNN = []
I_fairness_RNN = []
I_throughput_RNN = []
I_ratio_embb_RNN = []
I_ratio_uRLLC_RNN = []
I_ratio_mMTC_RNN = []

I_time_reduction = []

####RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from tqdm import tqdm  # For a nice progress bar!
import tensorboardX

# Set device
if torch.cuda.is_available():
    print('CUDA is available')

device = "cuda" if torch.cuda.is_available() else "cpu"
# Hyperparameters
input_size = 93  # 53#38
sequence_length = 140
hidden_size = 100
num_layers = 1
output_size = 13
num_classes = 13  ### is it the number of possible labels(we have 13 possibility of locations)????
learning_rate = 0.005
batch_size = 14  # i choose it to be a divisor of numer of samples otehr wise
num_epochs = 2000

# Load the data
df1 = pd.read_csv("test_data_milpVSrnn.csv")
df1.drop(df1.columns[0],axis=1,inplace=True)  ##drop the indexes of the dataframe
# df=df.drop(df.columns[0], axis=1, inplace=True)  ##drop the indexes of the dataframe
Initial_features = []
for BATCH in range(int(df1.shape[0] / sequence_length) // int(batch_size)):  # to drop the last batch
    nested_list = []
    for index in range(batch_size):
        list = []
        for j in range(sequence_length):
            list.append([df1.iloc[j+sequence_length * index+batch_size * sequence_length * BATCH]])
        nested_list.append(list)  # collect each batch of data into nested_list
    Initial_features.append(nested_list)
Initial_features = np.array(Initial_features).squeeze(axis=3)  ##before we where saving them as tensor, and did the squeeze before the last append => on dim=2
# print('iiiiii',Initial_features.shape)
##collect some data we will use when checking the output of the the RNN (features before transformation into OneHotVector)


df = pd.read_csv("test_data_milpVSrnn.csv")
# Extract the features
LOCATIONS = df[['LOCATION']]

features = df.drop(['LOCATION','user_loc_x','user_loc_y','loc_edge_x_1','loc_edge_x_2','loc_edge_x_3','loc_edge_y_1','loc_edge_y_2','loc_edge_y_3','loc_reg_x','loc_reg_y'],axis=1)
arr = df['RB'].values
arr_groups = arr.reshape(-1,140)
sum_groups = arr_groups.sum(axis=1)
repeated_sums = np.repeat(sum_groups,140)
features['Sum_RBS'] = repeated_sums / 400

arr = df['max_latency'].values
arr_groups = arr.reshape(-1,140)
sum_groups = arr_groups.sum(axis=1)
repeated_sums = np.repeat(sum_groups,140)
features['Sum_max_latency'] = repeated_sums / 10000

features = pd.get_dummies(features,columns=['RB','MCS','USer_association_BS','slice_type','priority'])

# features['RB'] = features['RB'] / 20
# features['MCS'] = features['MCS'] / 28

# features['slice_type'] = features['slice_type'] / 3

# features['priority'] = features['priority'] / 10

# features['USer_association_BS'] = features['USer_association_BS'] / 3


features['GOPS_required_DU'] = features['GOPS_required_DU'] / 30
features['GOPS_required_CU'] = features['GOPS_required_CU'] / 10

features['total_num_users'] = features['total_num_users'] / 100

features['max_latency'] = features['max_latency'] / 1000

features['TOTAL_GOPS_required_CU'] = features['TOTAL_GOPS_required_CU'] / 200
features['TOTAL_GOPS_required_DU'] = features['TOTAL_GOPS_required_DU'] / 500

features['GOPS_available_1'] = features['GOPS_available_1'] / 2000
features['GOPS_available_2'] = features['GOPS_available_2'] / 2000
features['GOPS_available_3'] = features['GOPS_available_3'] / 2000
features['GOPS_available_4'] = features['GOPS_available_4'] / 2000

features['rel_dist_Server_BS_0'] = features['rel_dist_Server_BS_0'] / 120
features['rel_dist_Server_BS_1'] = features['rel_dist_Server_BS_1'] / 120
features['rel_dist_Server_BS_2'] = features['rel_dist_Server_BS_2'] / 120
features['rel_dist_Server_BS_3'] = features['rel_dist_Server_BS_3'] / 120
features['rel_dist_Server_BS_4'] = features['rel_dist_Server_BS_4'] / 120
features['rel_dist_Server_BS_5'] = features['rel_dist_Server_BS_5'] / 120
features['rel_dist_Server_BS_6'] = features['rel_dist_Server_BS_6'] / 120
features['rel_dist_Server_BS_7'] = features['rel_dist_Server_BS_7'] / 120
features['rel_dist_Server_BS_8'] = features['rel_dist_Server_BS_8'] / 120
features['rel_dist_Server_BS_9'] = features['rel_dist_Server_BS_9'] / 120
features['rel_dist_Server_BS_10'] = features['rel_dist_Server_BS_10'] / 120
features['rel_dist_Server_BS_11'] = features['rel_dist_Server_BS_11'] / 120
features['rel_dist_Server_BS_12'] = features['rel_dist_Server_BS_12'] / 120
features['rel_dist_Server_BS_13'] = features['rel_dist_Server_BS_13'] / 120
features['rel_dist_Server_BS_14'] = features['rel_dist_Server_BS_14'] / 120
features['rel_dist_Server_BS_15'] = features['rel_dist_Server_BS_15'] / 120

features['rel_dist_ser_ser_0'] = features['rel_dist_ser_ser_0'] / 120
features['rel_dist_ser_ser_1'] = features['rel_dist_ser_ser_1'] / 120
features['rel_dist_ser_ser_2'] = features['rel_dist_ser_ser_2'] / 120
features['rel_dist_ser_ser_3'] = features['rel_dist_ser_ser_3'] / 120
features['rel_dist_ser_ser_4'] = features['rel_dist_ser_ser_4'] / 120
features['rel_dist_ser_ser_5'] = features['rel_dist_ser_ser_5'] / 120

features['link_latency_0_0'] = features['link_latency_0_0'] / 600
features['link_latency_0_1'] = features['link_latency_0_1'] / 600
features['link_latency_0_2'] = features['link_latency_0_2'] / 600
features['link_latency_0_3'] = features['link_latency_0_3'] / 600
features['link_latency_1_1'] = features['link_latency_1_1'] / 600
features['link_latency_1_2'] = features['link_latency_1_2'] / 600
features['link_latency_1_3'] = features['link_latency_1_3'] / 600
features['link_latency_2_2'] = features['link_latency_2_2'] / 600
features['link_latency_2_3'] = features['link_latency_2_3'] / 600
features['link_latency_3_3'] = features['link_latency_3_3'] / 600

features.to_csv('normalized_data.csv',index=False)


class CustomStarDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv("normalized_data.csv")
        self.df_labels = LOCATIONS
        self.df.drop(self.df.columns[0],axis=1,inplace=True)  ##drop the indexes of the dataframe
        # fill data from dataframe as a 3D tensor [40,32,19] to be the input of LSTM
        LIST_data = []
        LIST_target = []
        for BATCH in range(int(df.shape[0] / sequence_length) // int(batch_size)):  # to drop the last batch
            nested_list = []
            for index in range(batch_size):
                list = []
                for j in range(sequence_length):
                    list.append([self.df.iloc[j+sequence_length * index+batch_size * sequence_length * BATCH]])
                nested_list.append(list)  # collect each batch of data into nested_list
            nested_list = torch.tensor(nested_list).squeeze(dim=2).float()
            LIST_data.append(nested_list)

            # fill the targets as tensor
            nested_target = []
            for index in range(batch_size):
                list = []
                for j in range(sequence_length):
                    list.append([self.df_labels.iloc[j+sequence_length * index+batch_size * sequence_length * BATCH]])  ####wasss WRONGGG!!!!!!!!!!!! :)
                nested_target.append(list)
            nested_target = torch.tensor(nested_target).squeeze(dim=2).long()
            nested_target = nested_target.squeeze(dim=2)
            LIST_target.append(nested_target)

        self.test = LIST_data
        self.test_labels = LIST_target

    def __len__(self):
        return len(self.test)

    def __getitem__(self,idx):
        return self.test[idx],self.test_labels[idx]
        return self


TEST_DATA = CustomStarDataset()


def generate_mask(batch_sequence_lengths,max_sequence_length):
    mask = torch.zeros((len(batch_sequence_lengths),max_sequence_length),dtype=torch.float)
    for i,sequence_length in enumerate(batch_sequence_lengths):
        mask[i,:sequence_length] = 1
    return mask


# batch_sequence_lengths= [20,40,60,80,100]*(batch_size//5) #decesending order because my data are sorted in descending befor pack_padded
batch_sequence_lengths = [10,20,30,40,50,60,70,80,90,100,110,120,130,140] * (batch_size // 14)  # decesending order because my data are sorted in descending befor pack_padded

mask = generate_mask(batch_sequence_lengths,140)

criterion = nn.CrossEntropyLoss()


class BRNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(BRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2,num_classes)  # fully connected

    def forward(self,X,X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence

        h0 = torch.zeros(self.num_layers * 2,X.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2,X.size(0),self.hidden_size).to(device)

        batch_size,seq_len,_ = X.size()

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X,X_lengths,batch_first=True,enforce_sorted=False)  ###!!!!.to(device)

        # now run through LSTM
        X,_ = self.lstm(X,(h0,c0))

        # undo the packing operation
        X,_ = torch.nn.utils.rnn.pad_packed_sequence(X,batch_first=True)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1,X.shape[2])

        # run through actual linear layer
        X = self.fc(X)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X = X.view(batch_size,seq_len,num_classes)

        out = X
        return out


state_dict = torch.load('model-3.pt')

rnn_model = BRNN(input_size,hidden_size,num_layers,num_classes).to(device)
# Load the saved parameters into the new model
rnn_model.load_state_dict(state_dict)
# Set the model to evaluation mode
rnn_model.eval()

FAIRNESS_RNN = []
COST_RNN = []
THROUGHPUT_RNN = []
ADMITTANCE_RNN = []

RATIO_eMBB_RNN = []

RATIO_uRLLC_RNN = []

RATIO_mMTC_RNN = []

TIME_RNN = []

ratio_CU_edge_eMbb_RNN_all_runs = []
ratio_CU_edge_uRLLC_RNN_all_runs = []
ratio_CU_edge_mMTC_RNN_all_runs = []

ratio_CU_reg_eMbb_RNN_all_runs = []
ratio_CU_reg_uRLLC_RNN_all_runs = []
ratio_CU_reg_mMTC_RNN_all_runs = []

time_RNN = []

ratio_admitted_RNN = []

ratio_admitted_eMBB_RNN = []
ratio_admitted_uRLLC_RNN = []
ratio_admitted_mMTC_RNN = []

cost_RNN = []
total_throughput_RNN = []
fairness_RNN = []
Code_rate = [438,466,517,567,616,666,719,772,822,873,910,948]
spectral_eff = [2.5664,2.7305,3.0293,3.3223,3.6094,3.9023,4.2129,4.5234,4.8164,5.1152,5.3320,5.5547]
n_servers = 4
n_servers_edge = 3
counter = 0  ##to specify the number of users

with torch.no_grad():
    for batch_idx,data in enumerate(tqdm(TEST_DATA.test)):  ##for i, batch in enumerate(dataloader):
        ##data in this case is one sample in the following order{20,40,60,80,100}

        data = data.to(device=device)
        targets = TEST_DATA.test_labels[batch_idx]
        targets = targets.to(device=device)

        features = Initial_features[batch_idx]
        features = np.array(features)

        t = time.time()
        scores = rnn_model(data,batch_sequence_lengths)
        rnn_exec_time_1_batch = (time.time()-t) * 1000
        # print('rnn_exec_time_1_batch',rnn_exec_time_1_batch)
        # time_RNN.append((time.time()-t)*100)

        # print(time_RNN)
        _,max_index = torch.max(scores,dim=-1,keepdim=False)  ## returns the max_value,max_index tuple element wise over the all sequences of the batch [32,100]
        predictions = max_index
        # print('PREDECTION', predictions)
        # print('TARGET',targets)

        num_correct = 0
        num_samples = 0

        for i in range(len(batch_sequence_lengths)):
            NN = batch_sequence_lengths[i]
            num_correct += (predictions[i,:NN] == targets[i,:NN]).sum()
            num_samples += NN
        accuracy = (num_correct / num_samples) * 100
        print("Accuracy_test:",accuracy)
        # Element-wise multiply scores with the mask having already same shape of scores

        if mask.ndim < 3:
            mask = mask.unsqueeze(-1).expand_as(scores).to(device=device)

        scores = scores * mask
        loss = criterion(scores.reshape(-1,num_classes),targets.reshape(-1))
        print('loss_test',loss)

        predictions = predictions.flatten()

        ##get theta_cu and du from predictions along with n_slices.......
        predictions = predictions.cpu()
        predictions = predictions.numpy()  ##convert output to numpy

        for ii in range(len(batch_sequence_lengths)):
            NN = batch_sequence_lengths[ii]
            if (NN / 10) % 2 == 0:
                n_eMBB = int(NN * 0.25)
            else:
                n_eMBB = int(NN * 0.25)+1

            n_uRLLC = int(NN * 0.25)
            n_MTC = int(NN * 0.5)
            n_slices = n_eMBB+n_MTC+n_uRLLC
            PP = predictions[ii * 140:(ii+1) * 140]
            pred = PP[:NN]  # remove padding from output

            solution_theta_DU = np.zeros((n_slices,n_servers))
            solution_theta_CU = np.zeros((n_slices,n_servers))
            for i in range(n_slices):
                if pred[i] != 0:
                    if (pred[i] % 3) != 0:
                        ind_DU = int((pred[i] % 3)-1)
                    else:
                        ind_DU = 2

                    # print(']]]]]]', ind_DU)
                    ind_CU = int((pred[i]-1-ind_DU) / 3)
                    # print('+++++++++++', ind_CU)

                    solution_theta_CU[i,ind_CU] = 1  # append vector of length n_server with 1 on the choosen CU
                    solution_theta_DU[i,ind_DU] = 1  ##else Prediction reamin zero=> user not allocated

            FEATURE = features[ii]
            # FEATURE = np.array(FEATURE)
            FEATURE = FEATURE[:NN]

            C = []
            S_E = []
            # print(n_slices)
            for i in range(n_slices):
                # print(features[i])

                MCS = int(FEATURE[i,34])  ##get MCS from dataset ,I THINK @ INDEX 10 OF THE INPUT TENSOR,
                # MCS = 20
                # print(MCS)
                C.append(Code_rate[MCS-17] / 1024)
                S_E.append(spectral_eff[MCS-17])
            M = 6  # modulation bits log2(64)
            L = 2  # number of MIMO layers
            A = 4  # number of Antennas

            ####MUST check if the predictions fits the delay constraints and gops capacity.. As done in the random scenario
            R = FEATURE[0,35:39]  # GOPS availbale at each server : vector of size n_servers
            rem_server_cap = R  # remaining server capacity
            R_DU = FEATURE[:,40]
            R_CU = FEATURE[:,39]

            RBss = FEATURE[:,33]
            for i in range(n_slices):
                if RBss[i] == 0:  ##if user have no resource blocks, set the solution to zero
                    solution_theta_CU[i] = solution_theta_DU[i] = 0  ##sets the whole row in solution to zero ##i.e the user i is not admitted

            # for s in range(n_servers):
            #     for i in range(n_slices):
            #         rem_server_cap[s] -= solution_theta_DU[i, s] * R_DU[i] + solution_theta_CU[i, s] * R_CU[i]
            #         if rem_server_cap[s] <= 0:
            #             # print('no enough RBs')
            #             # if server capacity is not satisfied
            #             solution_theta_DU[i] = solution_theta_CU[i] = 0

            for s in range(n_servers):
                for i in range(n_slices):
                    rem_server_cap[s] -= solution_theta_DU[i,s] * R_DU[i]+solution_theta_CU[i,s] * R_CU[i]
                    if rem_server_cap[s] <= 0:
                        # print('no enough RBs')
                        # if server capacity is not satisfied
                        solution_theta_DU[i,s] = solution_theta_CU[i,s] = 0

            delta_max = FEATURE[:,53]
            loc_x_edge = FEATURE[0,2:5]
            loc_y_edge = FEATURE[0,5:8]
            loc_x_reg = np.array([FEATURE[0,8]])
            loc_y_reg = np.array([FEATURE[0,9]])
            for s in range(n_servers):
                for s1 in range(n_servers):
                    if s < n_servers_edge:
                        if s1 < n_servers_edge:
                            delta[s][s1] = 5 * math.sqrt((loc_x_edge[s]-loc_x_edge[s1]) ** 2+(
                                    loc_y_edge[s]-loc_y_edge[s1]) ** 2)  # random.randint(50,100)  # us edge-edge (5us/km => 10-20 km betwwen edge-edge server)
                        else:
                            delta[s][s1] = 5 * math.sqrt((loc_x_edge[s]-loc_x_reg[s1-n_servers_edge]) ** 2+(
                                    loc_y_edge[s]-loc_y_reg[s1-n_servers_edge]) ** 2)  # random.randint(200,400) #edge-regional(5us/km => 40-80 km betwwen edge-edge server)
                    else:
                        if s1 < n_servers_edge:
                            delta[s][s1] = 5 * math.sqrt((loc_x_edge[s1]-loc_x_reg[s-n_servers_edge]) ** 2+(loc_y_edge[s1]-loc_y_reg[s-n_servers_edge]) ** 2)  # random.randint(200,400) #reg-edge
                    if s == s1:
                        delta[s][s1] = 0

            for i in range(n_slices):
                for s in range(n_servers):
                    for s1 in range(n_servers):
                        if delta[s,s1] * (solution_theta_CU[i,s] * solution_theta_DU[i,s1]) > delta_max[i]:
                            # print("delta violated")
                            solution_theta_DU[i] = solution_theta_CU[i] = 0

            # print('CU',solution_theta_CU)
            # print('DU',solution_theta_DU)

            ##throughput calculation

            # RBss = FEATURE [:,33]
            throughput = 0
            for i in range(n_slices):
                for s in range(n_servers):
                    # throughput += N_SC*N_sym*RB_slices[i]*S_E[i]*L*1600*solution_theta_CU[i,s]/1000000##in Mbps
                    throughput += L * M * C[i] * N_SC * N_sym * RBss[i] * solution_theta_CU[i,s] * (1-0.14) / 1000  ##in Mbps
            #
            n_admitted_eMBB = sum(solution_theta_CU[i,s] for i in range(n_eMBB) for s in range(n_servers))
            n_admitted_uRLLC = sum(solution_theta_CU[i,s] for i in range(n_eMBB,n_eMBB+n_uRLLC) for s in range(n_servers))
            n_admitted_mMTC = sum(solution_theta_CU[i,s] for i in range(n_eMBB+n_uRLLC,n_slices) for s in range(n_servers))

            ######fairness index of admitted slices
            s = n_admitted_eMBB / n_eMBB+n_admitted_mMTC / n_MTC+n_admitted_uRLLC / n_uRLLC
            x = (n_admitted_eMBB / n_eMBB) ** 2+(n_admitted_mMTC / n_MTC) ** 2+(n_admitted_uRLLC / n_uRLLC) ** 2

            fairness = (s * s) / (3 * x)

            # objectivee=np.append(objectivee,m.objective_value)
            Cost_matrix = np.ones(n_servers)  # link capacity available between 2 servers
            n_servers_edge = 3
            n_servers = 4

            for s in range(n_servers):
                if s < n_servers_edge:
                    Cost_matrix[s] = 1.59  # edge
                else:
                    Cost_matrix[s] = 0.5  # regional

            CC = sum(Cost_matrix[s] * R_CU[i] * solution_theta_CU[i,s] for i in range(n_slices) for s in range(n_servers))
            # print('n_slices', n_slices)
            # print(batch_idx)

            ratio_admitted_RNN.append(sum(solution_theta_CU[i,s] for i in range(n_slices) for s in range(n_servers)) / n_slices * 100)
            # print(ratio_admitted_RNN)

            ratio_admitted_eMBB_RNN.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB) for s in range(n_servers)) / n_eMBB * 100)
            ratio_admitted_uRLLC_RNN.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB,n_eMBB+n_uRLLC) for s in range(n_servers)) / n_uRLLC * 100)
            ratio_admitted_mMTC_RNN.append(sum(solution_theta_CU[i,s] for i in range(n_eMBB+n_uRLLC,n_slices) for s in range(n_servers)) / n_MTC * 100)

            time_RNN.append(rnn_exec_time_1_batch)  ##same time for all experiments

            total_throughput_RNN.append(throughput)
            cost_RNN.append(CC)
            fairness_RNN.append(fairness)

        # print(batch_idx)
        # if (batch_idx+1) % len(Number_slices) == 0:  ######every 5 instances means after each experiment append to all_runs
        # print('collect all runss')
        # append to all_runs
        RATIO_eMBB_RNN.append(ratio_admitted_eMBB_RNN)
        RATIO_uRLLC_RNN.append(ratio_admitted_uRLLC_RNN)
        RATIO_mMTC_RNN.append(ratio_admitted_mMTC_RNN)
        COST_RNN.append(cost_RNN)
        FAIRNESS_RNN.append(fairness_RNN)
        THROUGHPUT_RNN.append(total_throughput_RNN)
        ADMITTANCE_RNN.append(ratio_admitted_RNN)

        TIME_RNN.append(time_RNN)

        # re-initialize for next Exp
        ratio_admitted_RNN = []
        ratio_admitted_eMBB_RNN = []
        ratio_admitted_uRLLC_RNN = []
        ratio_admitted_mMTC_RNN = []
        cost_RNN = []
        total_throughput_RNN = []
        fairness_RNN = []
        counter = 0

        time_RNN = []

##############################################PLOTTING
reduction = np.subtract(TIME_MILP,TIME_RNN) / TIME_MILP * 100
# print("time_MILP",TIME_MILP)
# print('rnn_exec_time_1_batch',TIME_RNN)

for N in range(len(Number_slices)):
    ratio_values_embb_optimal = []
    ratio_values_embb_edge = []
    ratio_values_embb_random = []
    ratio_values_embb_regional = []
    ratio_values_embb_RNN = []

    RB_values_1 = []
    RB_values_2 = []
    RB_values_3 = []
    RB_values_4 = []

    ratio_values_uRLLC_optimal = []
    ratio_values_uRLLC_RNN = []
    ratio_values_uRLLC_edge = []
    ratio_values_uRLLC_random = []
    ratio_values_uRLLC_regional = []

    ratio_values_mMTC_optimal = []
    ratio_values_mMTC_RNN = []
    ratio_values_mMTC_edge = []
    ratio_values_mMTC_random = []
    ratio_values_mMTC_regional = []

    Admittance_values_random = []
    Admittance_values_edge = []
    Admittance_values_optimal = []
    Admittance_values_regional = []

    throughput_values_random = []
    throughput_values_edge = []
    throughput_values_optimal = []
    throughput_values_regional = []

    cost_values_random = []
    cost_values_edge = []
    cost_values_optimal = []
    cost_values_regional = []

    fairness_values_random = []
    fairness_values_edge = []
    fairness_values_optimal = []
    fairness_values_regional = []

    time_values_MILP = []
    time_values_RNN = []
    ratio_values_embb_RNN = []
    ratio_values_uRLLC_RNN = []
    ratio_values_mMTC_RNN = []
    Admittance_values_RNN = []
    throughput_values_RNN = []
    cost_values_RNN = []
    fairness_values_RNN = []

    time_values_reduction = []

    GOPS_values_1 = []
    GOPS_values_2 = []
    GOPS_values_3 = []
    GOPS_values_4 = []

    GOPS_values_all_edge = []

    RB_demand_values_4 = []
    RB_demand_values_3 = []
    RB_demand_values_2 = []
    RB_demand_values_1 = []

    for i in range(Nb_exp):
        Admittance_values_random.append(ADMITTANCE_RANDOM[i][N])
        Admittance_values_edge.append(ADMITTANCE_EDGE[i][N])
        Admittance_values_optimal.append(ADMITTANCE_OPTIMAL[i][N])
        Admittance_values_regional.append(ADMITTANCE_regional[i][N])

        throughput_values_random.append(THROUGHPUT_RANDOM[i][N])
        throughput_values_edge.append(THROUGHPUT_EDGE[i][N])
        throughput_values_optimal.append(THROUGHPUT_OPTIMAL[i][N])
        throughput_values_regional.append(THROUGHPUT_regional[i][N])

        cost_values_random.append(COST_RANDOM[i][N])
        cost_values_edge.append(COST_EDGE[i][N])
        cost_values_optimal.append(COST_OPTIMAL[i][N])
        cost_values_regional.append(COST_regional[i][N])

        fairness_values_random.append(fairness_RANDOM[i][N])
        fairness_values_edge.append(fairness_EDGE[i][N])
        fairness_values_optimal.append(fairness_OPTIMAL[i][N])
        fairness_values_regional.append(fairness_REGIONAL[i][N])

        ratio_values_embb_optimal.append(RATIO_eMBB_optimal[i][N])
        ratio_values_embb_edge.append(RATIO_eMBB_edge[i][N])
        ratio_values_embb_random.append(RATIO_eMBB_random[i][N])
        ratio_values_embb_regional.append(RATIO_eMBB_regional[i][N])

        ratio_values_uRLLC_regional.append(RATIO_uRLLC_regional[i][N])
        ratio_values_uRLLC_optimal.append(RATIO_uRLLC_optimal[i][N])
        ratio_values_uRLLC_edge.append(RATIO_uRLLC_edge[i][N])
        ratio_values_uRLLC_random.append(RATIO_uRLLC_random[i][N])

        ratio_values_mMTC_optimal.append(RATIO_mMTC_optimal[i][N])
        ratio_values_mMTC_edge.append(RATIO_mMTC_edge[i][N])
        ratio_values_mMTC_random.append(RATIO_mMTC_random[i][N])
        ratio_values_mMTC_regional.append(RATIO_mMTC_regional[i][N])

        RB_values_1.append(RB_utilization_1_all_runs[i][N])
        RB_values_2.append(RB_utilization_2_all_runs[i][N])
        RB_values_3.append(RB_utilization_3_all_runs[i][N])
        RB_values_4.append(RB_utilization_4_all_runs[i][N])

        RB_demand_values_4.append(RB_demand_4_all_runs[i][N])
        RB_demand_values_3.append(RB_demand_3_all_runs[i][N])

        RB_demand_values_2.append(RB_demand_2_all_runs[i][N])
        RB_demand_values_1.append(RB_demand_1_all_runs[i][N])

        time_values_MILP.append(TIME_MILP[i][N])
        time_values_RNN.append(TIME_RNN[i][N])
        # reduction = np.subtract(TIME_MILP,TIME_RNN)/TIME_MILP*100
        time_values_reduction.append(reduction[i][N] if reduction[i][N] > 0 else 0)
        # print(np.subtract(TIME_RNN,TIME_MILP)/TIME_MILP*100)

        Admittance_values_RNN.append(ADMITTANCE_RNN[i][N])
        throughput_values_RNN.append(THROUGHPUT_RNN[i][N])
        cost_values_RNN.append(COST_RNN[i][N])
        fairness_values_RNN.append(FAIRNESS_RNN[i][N])
        ratio_values_embb_RNN.append(RATIO_eMBB_RNN[i][N])
        ratio_values_uRLLC_RNN.append(RATIO_uRLLC_RNN[i][N])
        ratio_values_mMTC_RNN.append(RATIO_mMTC_RNN[i][N])

        GOPS_values_1.append(GOPS_utilization_1_all_runs[i][N])
        GOPS_values_2.append(GOPS_utilization_2_all_runs[i][N])
        GOPS_values_3.append(GOPS_utilization_3_all_runs[i][N])
        GOPS_values_4.append(GOPS_utilization_4_all_runs[i][N])

        GOPS_values_all_edge.append(GOPS_utilization_all_edge_all_runs[i][N])

    I_RB_1.append(st.t.interval(confidence=0.95,df=len(RB_values_1)-1,loc=np.mean(RB_values_1),scale=st.sem(RB_values_1)))
    I_RB_2.append(st.t.interval(confidence=0.95,df=len(RB_values_2)-1,loc=np.mean(RB_values_2),scale=st.sem(RB_values_2)))
    I_RB_3.append(st.t.interval(confidence=0.95,df=len(RB_values_3)-1,loc=np.mean(RB_values_3),scale=st.sem(RB_values_3)))
    I_RB_4.append(st.t.interval(confidence=0.95,df=len(RB_values_4)-1,loc=np.mean(RB_values_4),scale=st.sem(RB_values_4)))

    I_RB_1_demand.append(st.t.interval(confidence=0.95,df=len(RB_demand_values_1)-1,loc=np.mean(RB_demand_values_1),scale=st.sem(RB_demand_values_1)))
    I_RB_2_demand.append(st.t.interval(confidence=0.95,df=len(RB_demand_values_2)-1,loc=np.mean(RB_demand_values_2),scale=st.sem(RB_demand_values_2)))
    I_RB_3_demand.append(st.t.interval(confidence=0.95,df=len(RB_demand_values_3)-1,loc=np.mean(RB_demand_values_3),scale=st.sem(RB_demand_values_3)))
    I_RB_4_demand.append(st.t.interval(confidence=0.95,df=len(RB_demand_values_4)-1,loc=np.mean(RB_demand_values_4),scale=st.sem(RB_demand_values_4)))

    I_GOPS_1.append(st.t.interval(confidence=0.95,df=len(GOPS_values_1)-1,loc=np.mean(GOPS_values_1),scale=st.sem(GOPS_values_1)))
    I_GOPS_2.append(st.t.interval(confidence=0.95,df=len(GOPS_values_2)-1,loc=np.mean(GOPS_values_2),scale=st.sem(GOPS_values_2)))
    I_GOPS_3.append(st.t.interval(confidence=0.95,df=len(GOPS_values_3)-1,loc=np.mean(GOPS_values_3),scale=st.sem(GOPS_values_3)))
    I_GOPS_4.append(st.t.interval(confidence=0.95,df=len(GOPS_values_4)-1,loc=np.mean(GOPS_values_4),scale=st.sem(GOPS_values_4)))
    I_GOPS_all_edge.append(st.t.interval(confidence=0.95,df=len(GOPS_values_all_edge)-1,loc=np.mean(GOPS_values_all_edge),scale=st.sem(GOPS_values_all_edge)))
    I_ratio_random.append(st.t.interval(confidence=0.95,df=len(Admittance_values_random)-1,loc=np.mean(Admittance_values_random),scale=st.sem(Admittance_values_random)))
    I_ratio_edge.append(st.t.interval(confidence=0.95,df=len(Admittance_values_edge)-1,loc=np.mean(Admittance_values_edge),scale=st.sem(Admittance_values_edge)))
    I_ratio_optimal.append(st.t.interval(confidence=0.95,df=len(Admittance_values_optimal)-1,loc=np.mean(Admittance_values_optimal),scale=st.sem(Admittance_values_optimal)))
    I_ratio_regional.append(st.t.interval(confidence=0.95,df=len(Admittance_values_regional)-1,loc=np.mean(Admittance_values_regional),scale=st.sem(Admittance_values_regional)))

    # print(I_ratio_mMTC_optimal)
    #     print(len(I_ratio_mMTC_optimal))
    # print(ratio_values_mMTC_optimal)
    # print(len(ratio_values_mMTC_optimal))

    I_throughput_random.append(st.t.interval(confidence=0.95,df=len(throughput_values_random)-1,loc=np.mean(throughput_values_random),scale=st.sem(throughput_values_random)))
    I_throughput_edge.append(st.t.interval(confidence=0.95,df=len(throughput_values_edge)-1,loc=np.mean(throughput_values_edge),scale=st.sem(throughput_values_edge)))
    I_throughput_optimal.append(st.t.interval(confidence=0.95,df=len(throughput_values_optimal)-1,loc=np.mean(throughput_values_optimal),scale=st.sem(throughput_values_optimal)))
    I_throughput_regional.append(st.t.interval(confidence=0.95,df=len(throughput_values_regional)-1,loc=np.mean(throughput_values_regional),scale=st.sem(throughput_values_regional)))

    I_cost_regional.append(st.t.interval(confidence=0.95,df=len(cost_values_regional)-1,loc=np.mean(cost_values_regional),scale=st.sem(cost_values_regional)))
    I_cost_random.append(st.t.interval(confidence=0.95,df=len(cost_values_random)-1,loc=np.mean(cost_values_random),scale=st.sem(cost_values_random)))
    I_cost_edge.append(st.t.interval(confidence=0.95,df=len(cost_values_edge)-1,loc=np.mean(cost_values_edge),scale=st.sem(cost_values_edge)))
    I_cost_optimal.append(st.t.interval(confidence=0.95,df=len(cost_values_optimal)-1,loc=np.mean(cost_values_optimal),scale=st.sem(cost_values_optimal)))

    I_fairness_random.append(st.t.interval(confidence=0.95,df=len(fairness_values_random)-1,loc=np.mean(fairness_values_random),scale=st.sem(fairness_values_random)))
    I_fairness_edge.append(st.t.interval(confidence=0.95,df=len(fairness_values_edge)-1,loc=np.mean(fairness_values_edge),scale=st.sem(fairness_values_edge)))
    I_fairness_optimal.append(st.t.interval(confidence=0.95,df=len(fairness_values_optimal)-1,loc=np.mean(fairness_values_optimal),scale=st.sem(fairness_values_optimal)))
    I_fairness_regional.append(st.t.interval(confidence=0.95,df=len(fairness_values_regional)-1,loc=np.mean(fairness_values_regional),scale=st.sem(fairness_values_regional)))

    I_ratio_embb_regional.append(st.t.interval(confidence=0.95,df=len(ratio_values_embb_regional)-1,loc=np.mean(ratio_values_embb_regional),scale=st.sem(ratio_values_embb_regional)))
    I_ratio_embb_optimal.append(st.t.interval(confidence=0.95,df=len(ratio_values_embb_optimal)-1,loc=np.mean(ratio_values_embb_optimal),scale=st.sem(ratio_values_embb_optimal)))
    I_ratio_embb_random.append(st.t.interval(confidence=0.95,df=len(ratio_values_embb_random)-1,loc=np.mean(ratio_values_embb_random),scale=st.sem(ratio_values_embb_random)))
    I_ratio_embb_edge.append(st.t.interval(confidence=0.95,df=len(ratio_values_embb_edge)-1,loc=np.mean(ratio_values_embb_edge),scale=st.sem(ratio_values_embb_edge)))
    I_time_MILP.append(st.t.interval(confidence=0.95,df=len(time_values_MILP)-1,loc=np.mean(time_values_MILP),scale=st.sem(time_values_MILP)))
    I_time_RNN.append(st.t.interval(confidence=0.95,df=len(time_values_RNN)-1,loc=np.mean(time_values_RNN),scale=st.sem(time_values_RNN)))
    I_time_reduction.append(st.t.interval(confidence=0.95,df=len(time_values_reduction)-1,loc=np.mean(time_values_reduction),scale=st.sem(time_values_reduction)))

    I_ratio_RNN.append(st.t.interval(confidence=0.95,df=len(Admittance_values_RNN)-1,loc=np.mean(Admittance_values_RNN),scale=st.sem(Admittance_values_RNN)))

    I_ratio_embb_RNN.append(st.t.interval(confidence=0.95,df=len(ratio_values_embb_RNN)-1,loc=np.mean(ratio_values_embb_RNN),scale=st.sem(ratio_values_embb_RNN)))

    I_ratio_uRLLC_RNN.append(st.t.interval(confidence=0.95,df=len(ratio_values_uRLLC_RNN)-1,loc=np.nanmean(ratio_values_uRLLC_RNN),scale=st.sem(ratio_values_uRLLC_RNN,nan_policy='omit')))

    I_ratio_mMTC_RNN.append(st.t.interval(confidence=0.95,df=len(ratio_values_mMTC_RNN)-1,loc=np.mean(ratio_values_mMTC_RNN),scale=st.sem(ratio_values_mMTC_RNN)))

    I_throughput_RNN.append(st.t.interval(confidence=0.95,df=len(throughput_values_RNN)-1,loc=np.mean(throughput_values_RNN),scale=st.sem(throughput_values_RNN)))
    I_cost_RNN.append(st.t.interval(confidence=0.95,df=len(cost_values_RNN)-1,loc=np.mean(cost_values_RNN),scale=st.sem(cost_values_RNN)))
    I_fairness_RNN.append(st.t.interval(confidence=0.95,df=len(fairness_values_RNN)-1,loc=np.mean(fairness_values_RNN),scale=st.sem(fairness_values_RNN)))

    for i in range(len(I_ratio_RNN)):
        if math.isnan(I_ratio_RNN[i][0]):
            I_ratio_RNN[i] = Admittance_values_RNN[0],Admittance_values_RNN[0]

    I_ratio_uRLLC_optimal.append(st.t.interval(confidence=0.95,df=len(ratio_values_uRLLC_optimal)-1,loc=np.nanmean(ratio_values_uRLLC_optimal),scale=st.sem(ratio_values_uRLLC_optimal,nan_policy='omit')))
    I_ratio_uRLLC_random.append(st.t.interval(confidence=0.95,df=len(ratio_values_uRLLC_random)-1,loc=np.nanmean(ratio_values_uRLLC_random),scale=st.sem(ratio_values_uRLLC_random,nan_policy='omit')))
    I_ratio_uRLLC_edge.append(st.t.interval(confidence=0.95,df=len(ratio_values_uRLLC_edge)-1,loc=np.nanmean(ratio_values_uRLLC_edge),scale=st.sem(ratio_values_uRLLC_edge,nan_policy='omit')))
    I_ratio_uRLLC_regional.append(st.t.interval(confidence=0.95,df=len(ratio_values_uRLLC_regional)-1,loc=np.nanmean(ratio_values_uRLLC_regional),scale=st.sem(ratio_values_uRLLC_regional,nan_policy='omit')))

    I_ratio_mMTC_regional.append(st.t.interval(confidence=0.95,df=len(ratio_values_mMTC_regional)-1,loc=np.mean(ratio_values_mMTC_regional),scale=st.sem(ratio_values_mMTC_regional)))
    I_ratio_mMTC_optimal.append(st.t.interval(confidence=0.95,df=len(ratio_values_mMTC_optimal)-1,loc=np.mean(ratio_values_mMTC_optimal),scale=st.sem(ratio_values_mMTC_optimal)))
    I_ratio_mMTC_random.append(st.t.interval(confidence=0.95,df=len(ratio_values_mMTC_random)-1,loc=np.mean(ratio_values_mMTC_random),scale=st.sem(ratio_values_mMTC_random)))
    I_ratio_mMTC_edge.append(st.t.interval(confidence=0.95,df=len(ratio_values_mMTC_edge)-1,loc=np.mean(ratio_values_mMTC_edge),scale=st.sem(ratio_values_mMTC_edge)))

    #     print(I_ratio_edge)
    #     print(Admittance_values_edge,Admittance_values_optimal)

    for i in range(len(I_ratio_uRLLC_optimal)):
        if math.isnan(I_ratio_uRLLC_optimal[i][0]):
            I_ratio_uRLLC_optimal[i] = ratio_values_uRLLC_optimal[0],ratio_values_uRLLC_optimal[0]
        if math.isnan(I_ratio_uRLLC_edge[i][0]):
            I_ratio_uRLLC_edge[i] = ratio_values_uRLLC_edge[0],ratio_values_uRLLC_edge[0]
        if math.isnan(I_ratio_uRLLC_regional[i][0]):
            I_ratio_uRLLC_regional[i] = ratio_values_uRLLC_regional[0],ratio_values_uRLLC_regional[0]
        if math.isnan(I_ratio_uRLLC_random[i][0]):
            I_ratio_uRLLC_random[i] = ratio_values_uRLLC_random[0],ratio_values_uRLLC_random[0]
    for i in range(len(I_ratio_mMTC_optimal)):
        if math.isnan(I_ratio_mMTC_optimal[i][0]):
            I_ratio_mMTC_optimal[i] = ratio_values_mMTC_optimal[0],ratio_values_mMTC_optimal[0]
        if math.isnan(I_ratio_mMTC_edge[i][0]):
            I_ratio_mMTC_edge[i] = ratio_values_mMTC_edge[0],ratio_values_mMTC_edge[0]
        if math.isnan(I_ratio_mMTC_regional[i][0]):
            I_ratio_mMTC_regional[i] = ratio_values_mMTC_regional[0],ratio_values_mMTC_regional[0]
        if math.isnan(I_ratio_mMTC_random[i][0]):
            I_ratio_mMTC_random[i] = ratio_values_mMTC_random[0],ratio_values_mMTC_random[0]
    for i in range(len(I_ratio_embb_optimal)):
        if math.isnan(I_ratio_embb_optimal[i][0]):
            I_ratio_embb_optimal[i] = ratio_values_embb_optimal[0],ratio_values_embb_optimal[0]
        if math.isnan(I_ratio_embb_edge[i][0]):
            I_ratio_embb_edge[i] = ratio_values_embb_edge[0],ratio_values_embb_edge[0]
        if math.isnan(I_ratio_embb_regional[i][0]):
            I_ratio_embb_regional[i] = ratio_values_embb_regional[0],ratio_values_embb_regional[0]
        if math.isnan(I_ratio_embb_random[i][0]):
            I_ratio_embb_random[i] = ratio_values_embb_random[0],ratio_values_embb_random[0]
    # print(Admittance_values_optimal[0])
    for i in range(len(I_ratio_optimal)):
        if math.isnan(I_ratio_optimal[i][0]):
            I_ratio_optimal[i] = Admittance_values_optimal[0],Admittance_values_optimal[0]
        if math.isnan(I_ratio_edge[i][0]):
            I_ratio_edge[i] = Admittance_values_edge[0],Admittance_values_edge[0]
        if math.isnan(I_ratio_random[i][0]):
            I_ratio_random[i] = Admittance_values_random[0],Admittance_values_random[0]
        if math.isnan(I_ratio_regional[i][0]):
            I_ratio_regional[i] = Admittance_values_regional[0],Admittance_values_regional[0]

# print(I_ratio_uRLLC_edge)
lower_ratio_RNN = []
upper_ratio_RNN = []
lower_ratio_random = []
upper_ratio_random = []
lower_ratio_edge = []
upper_ratio_edge = []
lower_ratio_optimal = []
upper_ratio_optimal = []
lower_ratio_regional = []
upper_ratio_regional = []

for i in range(len(Number_slices)):
    lower_ratio_random.append(I_ratio_random[i][0])
    upper_ratio_random.append(I_ratio_random[i][1] if I_ratio_random[i][1] < 100 else 100)
    lower_ratio_edge.append(I_ratio_edge[i][0])
    upper_ratio_edge.append(I_ratio_edge[i][1] if I_ratio_edge[i][1] < 100 else 100)
    lower_ratio_optimal.append(I_ratio_optimal[i][0])
    upper_ratio_optimal.append(I_ratio_optimal[i][1] if I_ratio_optimal[i][1] < 100 else 100)
    lower_ratio_regional.append(I_ratio_regional[i][0])
    upper_ratio_regional.append(I_ratio_regional[i][1] if I_ratio_regional[i][1] < 100 else 100)
    lower_ratio_RNN.append(I_ratio_RNN[i][0])
    upper_ratio_RNN.append(I_ratio_RNN[i][1] if I_ratio_RNN[i][1] < 100 else 100)

r = []
r1 = []
r0 = []
r2 = []
r3 = []

for lower,upper,y in zip(lower_ratio_random,upper_ratio_random,range(len(lower_ratio_random))):
    plt.plot((y,y),(lower,upper),'v:',color='tomato',label='Random',markersize=13)
    r0.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_edge,upper_ratio_edge,range(len(lower_ratio_edge))):
    plt.plot((y,y),(lower,upper),'d--',color='royalblue',label='All_edge',markersize=13)
    r1.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_regional,upper_ratio_regional,range(len(lower_ratio_regional))):
    plt.plot((y,y),(lower,upper),'^-.',color='y',label='Static',markersize=13)
    r2.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_optimal,upper_ratio_optimal,range(len(lower_ratio_optimal))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='Optimal',markersize=13)
    r.append(lower+(upper-lower) / 2)
for lower,upper,y in zip(lower_ratio_RNN,upper_ratio_RNN,range(len(lower_ratio_RNN))):
    plt.plot((y,y),(lower,upper),'^--',color='purple',label='RNN',markersize=13)
    r3.append(lower+(upper-lower) / 2)
plt.plot(r0,':',color='tomato')
plt.plot(r1,'--',color='royalblue')
plt.plot(r2,'-.',color='y')
plt.plot(r,'-',color='seagreen')
plt.plot(r3,'--',color='purple')

plt.xticks(range(len(lower_ratio_optimal)),Number_slices,fontsize=25)
plt.ylim([0,100])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('Admittance Ratio (%)',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

# print()

lower_ratio_embb_optimal = []
upper_ratio_embb_optimal = []
lower_ratio_embb_RNN = []
upper_ratio_embb_RNN = []
lower_ratio_embb_edge = []
upper_ratio_embb_edge = []
lower_ratio_embb_random = []
upper_ratio_embb_random = []

lower_ratio_embb_regional = []
upper_ratio_embb_regional = []

for i in range(len(Number_slices)):
    lower_ratio_embb_optimal.append(I_ratio_embb_optimal[i][0] if I_ratio_embb_optimal[i][0] < 100 else 100)
    upper_ratio_embb_optimal.append(I_ratio_embb_optimal[i][1] if I_ratio_embb_optimal[i][1] < 100 else 100)
    lower_ratio_embb_edge.append(I_ratio_embb_edge[i][0] if I_ratio_embb_edge[i][0] < 100 else 100)
    upper_ratio_embb_edge.append(I_ratio_embb_edge[i][1] if I_ratio_embb_edge[i][1] < 100 else 100)
    lower_ratio_embb_random.append(I_ratio_embb_random[i][0] if I_ratio_embb_random[i][0] < 100 else 100)
    upper_ratio_embb_random.append(I_ratio_embb_random[i][1] if I_ratio_embb_random[i][1] < 100 else 100)
    lower_ratio_embb_regional.append(I_ratio_embb_regional[i][0] if I_ratio_embb_regional[i][0] < 100 else 100)
    upper_ratio_embb_regional.append(I_ratio_embb_regional[i][1] if I_ratio_embb_regional[i][1] < 100 else 100)
    lower_ratio_embb_RNN.append(I_ratio_embb_RNN[i][0] if I_ratio_embb_RNN[i][1] < 100 else 100)
    upper_ratio_embb_RNN.append(I_ratio_embb_RNN[i][1] if I_ratio_embb_RNN[i][1] < 100 else 100)

r = []
r1 = []
r0 = []
r2 = []
r3 = []

for lower,upper,y in zip(lower_ratio_embb_random,upper_ratio_embb_random,range(len(lower_ratio_embb_random))):
    plt.plot((y,y),(lower,upper),'v:',color='tomato',label='Random',markersize=13)
    r0.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_embb_edge,upper_ratio_embb_edge,range(len(lower_ratio_embb_edge))):
    plt.plot((y,y),(lower,upper),'d--',color='royalblue',label='All_edge',markersize=13)
    r1.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_embb_regional,upper_ratio_embb_regional,range(len(lower_ratio_embb_regional))):
    plt.plot((y,y),(lower,upper),'^-.',color='y',label='Static',markersize=13)
    r2.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_embb_optimal,upper_ratio_embb_optimal,range(len(lower_ratio_embb_optimal))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='Optimal',markersize=13)
    r.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_embb_RNN,upper_ratio_embb_RNN,range(len(lower_ratio_embb_RNN))):
    plt.plot((y,y),(lower,upper),'^--',color='purple',label='RNN',markersize=13)
    r3.append(lower+(upper-lower) / 2)

plt.plot(r0,':',color='tomato')
plt.plot(r1,'--',color='royalblue')
plt.plot(r2,'-.',color='y')
plt.plot(r,'-',color='seagreen')
plt.plot(r3,'--',color='purple')

plt.xticks(range(len(lower_ratio_embb_optimal)),Number_slices,fontsize=25)
plt.ylim([0,100])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('eMBB users admittance ratio (%)',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

lower_ratio_uRLLC_optimal = []
upper_ratio_uRLLC_optimal = []
lower_ratio_uRLLC_edge = []
upper_ratio_uRLLC_edge = []
lower_ratio_uRLLC_random = []
upper_ratio_uRLLC_random = []
lower_ratio_uRLLC_regional = []
upper_ratio_uRLLC_regional = []
lower_ratio_uRLLC_RNN = []
upper_ratio_uRLLC_RNN = []

for i in range(len(Number_slices)):
    lower_ratio_uRLLC_optimal.append(I_ratio_uRLLC_optimal[i][0] if I_ratio_uRLLC_optimal[i][0] < 100 else 100)
    upper_ratio_uRLLC_optimal.append(I_ratio_uRLLC_optimal[i][1] if I_ratio_uRLLC_optimal[i][1] < 100 else 100)
    lower_ratio_uRLLC_edge.append(I_ratio_uRLLC_edge[i][0] if I_ratio_uRLLC_edge[i][0] < 100 else 100)
    upper_ratio_uRLLC_edge.append(I_ratio_uRLLC_edge[i][1] if I_ratio_uRLLC_edge[i][1] < 100 else 100)
    lower_ratio_uRLLC_random.append(I_ratio_uRLLC_random[i][0] if I_ratio_uRLLC_random[i][0] < 100 else 100)
    upper_ratio_uRLLC_random.append(I_ratio_uRLLC_random[i][1] if I_ratio_uRLLC_random[i][1] < 100 else 100)
    lower_ratio_uRLLC_regional.append(I_ratio_uRLLC_regional[i][0] if I_ratio_uRLLC_regional[i][0] < 100 else 100)
    upper_ratio_uRLLC_regional.append(I_ratio_uRLLC_regional[i][1] if I_ratio_uRLLC_regional[i][1] < 100 else 100)
    lower_ratio_uRLLC_RNN.append(I_ratio_uRLLC_RNN[i][0] if I_ratio_uRLLC_RNN[i][1] < 100 else 100)
    upper_ratio_uRLLC_RNN.append(I_ratio_uRLLC_RNN[i][1] if I_ratio_uRLLC_RNN[i][1] < 100 else 100)

r = []
r1 = []
r0 = []
r2 = []
r3 = []

for lower,upper,y in zip(lower_ratio_uRLLC_random,upper_ratio_uRLLC_random,range(len(lower_ratio_uRLLC_random))):
    plt.plot((y,y),(lower,upper),'v:',color='tomato',label='Random',markersize=13)
    r0.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_uRLLC_edge,upper_ratio_uRLLC_edge,range(len(lower_ratio_uRLLC_edge))):
    plt.plot((y,y),(lower,upper),'d--',color='royalblue',label='All_edge',markersize=13)
    r1.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_uRLLC_regional,upper_ratio_uRLLC_regional,range(len(lower_ratio_uRLLC_regional))):
    plt.plot((y,y),(lower,upper),'^-.',color='y',label='Static',markersize=13)
    r2.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_uRLLC_optimal,upper_ratio_uRLLC_optimal,range(len(lower_ratio_uRLLC_optimal))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='Optimal',markersize=13)
    r.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_uRLLC_RNN,upper_ratio_uRLLC_RNN,range(len(lower_ratio_uRLLC_RNN))):
    plt.plot((y,y),(lower,upper),'^--',color='purple',label='RNN',markersize=13)
    r3.append(lower+(upper-lower) / 2)
plt.plot(r0,':',color='tomato')
plt.plot(r1,'--',color='royalblue')
plt.plot(r2,'-.',color='y')
plt.plot(r,'-',color='seagreen')
plt.plot(r3,'--',color='purple')

plt.xticks(range(len(lower_ratio_uRLLC_optimal)),Number_slices,fontsize=25)
plt.ylim([0,100])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('uRLLC users admittance ratio (%)',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

lower_ratio_mMTC_optimal = []
upper_ratio_mMTC_optimal = []
lower_ratio_mMTC_edge = []
upper_ratio_mMTC_edge = []
lower_ratio_mMTC_random = []
upper_ratio_mMTC_random = []

lower_ratio_mMTC_regional = []
upper_ratio_mMTC_regional = []
lower_ratio_mMTC_RNN = []
upper_ratio_mMTC_RNN = []
for i in range(len(Number_slices)):
    lower_ratio_mMTC_optimal.append(I_ratio_mMTC_optimal[i][0] if I_ratio_mMTC_optimal[i][0] < 100 else 100)
    upper_ratio_mMTC_optimal.append(I_ratio_mMTC_optimal[i][1] if I_ratio_mMTC_optimal[i][1] < 100 else 100)
    lower_ratio_mMTC_edge.append(I_ratio_mMTC_edge[i][0] if I_ratio_mMTC_edge[i][0] < 100 else 100)
    upper_ratio_mMTC_edge.append(I_ratio_mMTC_edge[i][1] if I_ratio_mMTC_edge[i][1] < 100 else 100)
    lower_ratio_mMTC_random.append(I_ratio_mMTC_random[i][0] if I_ratio_mMTC_random[i][0] < 100 else 100)
    upper_ratio_mMTC_random.append(I_ratio_mMTC_random[i][1] if I_ratio_mMTC_random[i][1] < 100 else 100)
    lower_ratio_mMTC_regional.append(I_ratio_mMTC_regional[i][0] if I_ratio_mMTC_regional[i][0] < 100 else 100)
    upper_ratio_mMTC_regional.append(I_ratio_mMTC_regional[i][1] if I_ratio_mMTC_regional[i][1] < 100 else 100)
    lower_ratio_mMTC_RNN.append(I_ratio_mMTC_RNN[i][0] if I_ratio_mMTC_RNN[i][1] < 100 else 100)
    upper_ratio_mMTC_RNN.append(I_ratio_mMTC_RNN[i][1] if I_ratio_mMTC_RNN[i][1] < 100 else 100)

r = []
r1 = []
r0 = []
r2 = []
r3 = []

for lower,upper,y in zip(lower_ratio_mMTC_random,upper_ratio_mMTC_random,range(len(lower_ratio_mMTC_random))):
    plt.plot((y,y),(lower,upper),'v:',color='tomato',label='Random',markersize=13)
    r0.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_mMTC_edge,upper_ratio_mMTC_edge,range(len(lower_ratio_mMTC_edge))):
    plt.plot((y,y),(lower,upper),'d-.',color='royalblue',label='All_edge',markersize=13)
    r1.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_mMTC_regional,upper_ratio_mMTC_regional,range(len(lower_ratio_mMTC_regional))):
    plt.plot((y,y),(lower,upper),'^--',color='y',label='Static',markersize=13)
    r2.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_mMTC_optimal,upper_ratio_mMTC_optimal,range(len(lower_ratio_mMTC_optimal))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='Optimal',markersize=13)
    r.append(lower+(upper-lower) / 2)
for lower,upper,y in zip(lower_ratio_mMTC_RNN,upper_ratio_mMTC_RNN,range(len(lower_ratio_mMTC_RNN))):
    plt.plot((y,y),(lower,upper),'^--',color='purple',label='RNN',markersize=13)
    r3.append(lower+(upper-lower) / 2)

plt.plot(r0,':',color='tomato')
plt.plot(r1,'--',color='royalblue')
plt.plot(r2,'-.',color='y')
plt.plot(r,'-',color='seagreen')
plt.plot(r3,'--',color='purple')

plt.xticks(range(len(lower_ratio_mMTC_optimal)),Number_slices,fontsize=25)
plt.ylim([0,100])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('mMTC users admittance Ratio (%)',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

lower_cost_random = []
upper_cost_random = []
lower_cost_edge = []
upper_cost_edge = []
lower_cost_optimal = []
upper_cost_optimal = []
lower_cost_regional = []
upper_cost_regional = []
lower_cost_RNN = []
upper_cost_RNN = []

for i in range(len(Number_slices)):
    lower_cost_random.append(I_cost_random[i][0])
    upper_cost_random.append(I_cost_random[i][1])
    lower_cost_edge.append(I_cost_edge[i][0])
    upper_cost_edge.append(I_cost_edge[i][1])
    lower_cost_optimal.append(I_cost_optimal[i][0])
    upper_cost_optimal.append(I_cost_optimal[i][1])
    lower_cost_regional.append(I_cost_regional[i][0])

    upper_cost_regional.append(I_cost_regional[i][1])

    lower_cost_RNN.append(I_cost_RNN[i][0])
    upper_cost_RNN.append(I_cost_RNN[i][1])

r = []
r1 = []
r0 = []
r2 = []
r3 = []
for lower,upper,y in zip(lower_cost_random,upper_cost_random,range(len(lower_cost_random))):
    plt.plot((y,y),(lower,upper),'v:',color='tomato',label='Random',markersize=13)
    r0.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_cost_edge,upper_cost_edge,range(len(lower_cost_edge))):
    plt.plot((y,y),(lower,upper),'d--',color='royalblue',label='All_edge',markersize=13)
    r1.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_cost_regional,upper_cost_regional,range(len(lower_cost_regional))):
    plt.plot((y,y),(lower,upper),'^-.',color='y',label='Static',markersize=13)
    r2.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_cost_optimal,upper_cost_optimal,range(len(lower_cost_optimal))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='Optimal',markersize=13)
    r.append(lower+(upper-lower) / 2)
for lower,upper,y in zip(lower_cost_RNN,upper_cost_RNN,range(len(lower_cost_RNN))):
    plt.plot((y,y),(lower,upper),'^--',color='purple',label='RNN',markersize=13)
    r3.append(lower+(upper-lower) / 2)

plt.plot(r0,':',color='tomato')
plt.plot(r1,'--',color='royalblue')
plt.plot(r2,'-.',color='y')
plt.plot(r,'-',color='seagreen')
plt.plot(r3,'--',color='purple')

plt.xticks(range(len(lower_cost_optimal)),Number_slices,fontsize=25)
plt.ylim([0,200])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('Cost (monetary unit)',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

lower_fairness_random = []
upper_fairness_random = []
lower_fairness_edge = []
upper_fairness_edge = []
lower_fairness_optimal = []
upper_fairness_optimal = []
lower_fairness_regional = []
upper_fairness_regional = []

lower_fairness_RNN = []
upper_fairness_RNN = []

for i in range(len(Number_slices)):
    lower_fairness_random.append(I_fairness_random[i][0] if I_fairness_random[i][0] < 1 else 1)
    upper_fairness_random.append(I_fairness_random[i][1] if I_fairness_random[i][1] < 1 else 1)
    lower_fairness_edge.append(I_fairness_edge[i][0] if I_fairness_edge[i][0] < 1 else 1)
    upper_fairness_edge.append(I_fairness_edge[i][1] if I_fairness_edge[i][1] < 1 else 1)
    lower_fairness_optimal.append(I_fairness_optimal[i][0] if I_fairness_optimal[i][0] < 1 else 1)
    upper_fairness_optimal.append(I_fairness_optimal[i][1] if I_fairness_optimal[i][1] < 1 else 1)
    lower_fairness_regional.append(I_fairness_regional[i][0] if I_fairness_regional[i][0] < 1 else 1)
    upper_fairness_regional.append(I_fairness_regional[i][1] if I_fairness_regional[i][1] < 1 else 1)

    lower_fairness_RNN.append(I_fairness_RNN[i][0] if I_fairness_RNN[i][0] < 1 else 1)
    upper_fairness_RNN.append(I_fairness_RNN[i][1] if I_fairness_RNN[i][1] < 1 else 1)

r = []
r1 = []
r0 = []
r2 = []
r3 = []

for lower,upper,y in zip(lower_fairness_random,upper_fairness_random,range(len(lower_fairness_random))):
    plt.plot((y,y),(lower,upper),'v:',color='tomato',label='Random',markersize=13)
    r0.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_fairness_edge,upper_fairness_edge,range(len(lower_fairness_edge))):
    plt.plot((y,y),(lower,upper),'d--',color='royalblue',label='All_edge',markersize=13)
    r1.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_fairness_regional,upper_fairness_regional,range(len(lower_fairness_regional))):
    plt.plot((y,y),(lower,upper),'^-',color='y',label='Static',markersize=13)
    r2.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_fairness_optimal,upper_fairness_optimal,range(len(lower_fairness_optimal))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='Optimal',markersize=13)
    r.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_fairness_RNN,upper_fairness_RNN,range(len(lower_fairness_RNN))):
    plt.plot((y,y),(lower,upper),'^--',color='purple',label='RNN',markersize=13)
    r3.append(lower+(upper-lower) / 2)

plt.plot(r0,':',color='tomato')
plt.plot(r1,'--',color='royalblue')
plt.plot(r2,'-.',color='y')
plt.plot(r,'-',color='seagreen')
plt.plot(r3,'-',color='purple')

plt.xticks(range(len(lower_fairness_optimal)),Number_slices,fontsize=25)
plt.ylim([0.6,1])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('Fairness Index',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

lower_throughput_random = []
upper_throughput_random = []
lower_throughput_edge = []
upper_throughput_edge = []
lower_throughput_optimal = []
upper_throughput_optimal = []
lower_throughput_regional = []
upper_throughput_regional = []

lower_throughput_RNN = []
upper_throughput_RNN = []

for i in range(len(Number_slices)):
    lower_throughput_random.append(I_throughput_random[i][0])
    upper_throughput_random.append(I_throughput_random[i][1])
    lower_throughput_edge.append(I_throughput_edge[i][0])
    upper_throughput_edge.append(I_throughput_edge[i][1])
    lower_throughput_optimal.append(I_throughput_optimal[i][0])
    upper_throughput_optimal.append(I_throughput_optimal[i][1])
    lower_throughput_regional.append(I_throughput_regional[i][0])
    upper_throughput_regional.append(I_throughput_regional[i][1])
    lower_throughput_RNN.append(I_throughput_RNN[i][0])
    upper_throughput_RNN.append(I_throughput_RNN[i][1])

r = []
r1 = []
r0 = []
r2 = []
r3 = []

for lower,upper,y in zip(lower_throughput_random,upper_throughput_random,range(len(lower_throughput_random))):
    plt.plot((y,y),(lower,upper),'v:',color='tomato',label='Random',markersize=13)
    r0.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_throughput_edge,upper_throughput_edge,range(len(lower_throughput_edge))):
    plt.plot((y,y),(lower,upper),'d--',color='royalblue',label='All_edge',markersize=13)
    r1.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_throughput_regional,upper_throughput_regional,range(len(lower_throughput_regional))):
    plt.plot((y,y),(lower,upper),'^-.',color='y',label='Static',markersize=13)
    r2.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_throughput_optimal,upper_throughput_optimal,range(len(lower_throughput_optimal))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='Optimal',markersize=13)
    r.append(lower+(upper-lower) / 2)
for lower,upper,y in zip(lower_throughput_RNN,upper_throughput_RNN,range(len(lower_throughput_RNN))):
    plt.plot((y,y),(lower,upper),'^--',color='purple',label='RNN',markersize=13)
    r3.append(lower+(upper-lower) / 2)
plt.plot(r0,':',color='tomato')
plt.plot(r1,'--',color='royalblue')
plt.plot(r2,'-.',color='y')
plt.plot(r,'-',color='seagreen')
plt.plot(r3,'--',color='purple')

plt.xticks(range(len(lower_throughput_optimal)),Number_slices,fontsize=25)
plt.ylim([0,500])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('Throughput (Mbps)',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

# labels = N
# X = np.arange(len(labels))
# fig = plt.figure()
# ax = fig.add_axes([0, 0, 2, 1])
# ax.set_ylabel('% Servers_location used by admitted slices ')
# ax.set_xlabel('total_#_slices')
#
# ax.set_title('regional Scenario', y=-0.2)
# # ax.set_yticks(np.arange(0, 50, 10))
# ax.bar(X - 0.2, ratio_CU_reg_eMbb_regional, color='g', width=0.2, label='reg_server', edgecolor="black")
# ax.bar(X + 0.00, ratio_CU_reg_uRLLC_regional, color='g', width=0.2, edgecolor="black")
# ax.bar(X + 0.2, ratio_CU_reg_MTC_regional, color='g', width=0.2, edgecolor="black")
#
# # for k, label in enumerate(labels):
# #     ax.text(X[k]-0.3, max(ratio_CU_edge_eMbb[k],ratio_CU_reg_eMbb[k])+2, 'eMBB', rotation=90)
# #     ax.text(X[k]-0.1, max(ratio_CU_edge_uRLLC[k],ratio_CU_reg_uRLLC[k])+2, 'uRLLC', rotation=90)
# #     ax.text(X[k]+0.1, max(ratio_CU_edge_MTC[k],ratio_CU_reg_MTC[k])+2, 'mMTC', rotation=90)
#
# for k, label in enumerate(labels):
#     ax.text(X[k] - 0.25, 102, 'eMBB', rotation=90)
#     ax.text(X[k] - 0.05, 102, 'uRLLC', rotation=90)
#     ax.text(X[k] + 0.15, 102, 'mMTC', rotation=90)
#
# ax.bar(X - 0.2, ratio_CU_edge_eMbb_regional, bottom=ratio_CU_reg_eMbb_regional, color='r', width=0.2, label='edge_server',
#        edgecolor="black")
# ax.bar(X + 0.00, ratio_CU_edge_uRLLC_regional, bottom=ratio_CU_reg_uRLLC_regional, color='r', width=0.2,
#        edgecolor="black")
# ax.bar(X + 0.2, ratio_CU_edge_MTC_regional, bottom=ratio_CU_reg_MTC_regional, color='r', width=0.2, edgecolor="black")
#
# ax.set_axisbelow(True)
#
# ax.set_xticks(X)
# ax.set_xticklabels(labels)
# ax.legend()
# plt.grid()
# plt.show()
#


# ss = np.arange(1,5,1)
# indexes=[]
# servers=['edge#1','edge#2','edge#3','reg#1']
# for i in range(len(Number_slices)):
#     indexes.append(3*i+3)
# GOPS_utilization_edge = np.insert(GOPS_utilization_edge, obj=indexes, values=0)
# print(GOPS_utilization_edge)
#
# GOPS_utilization_optimal.shape= len(Number_slices),4
# GOPS_utilization_random.shape= len(Number_slices),4
# GOPS_utilization_regional.shape= len(Number_slices),4
# GOPS_utilization_edge.shape= len(Number_slices),4
# #GOPS_utilization_edge=np.append(GOPS_utilization_edge,0)
# for s in range (n_servers):
#     #for i in range(len(Number_slices)):
#     if s ==0:
#         plt.plot(Number_slices,GOPS_utilization_optimal[:,s],'ro-',label='edge#1')
#     elif s==1:
#         plt.plot(Number_slices, GOPS_utilization_optimal[:, s], 'bo-', label='edge#2')
#     elif s == 2:
#         plt.plot(Number_slices, GOPS_utilization_optimal[:, s], 'yo-', label='edge#3')
#     elif s ==3:
#         plt.plot(Number_slices, GOPS_utilization_optimal[:, s], 'go-', label='regional')
# plt.ylabel("GOPS Load")
# plt.xlabel("#_users")
# plt.title('Optimal Scenario')
# plt.legend()
# plt.grid()
# plt.title("")GOPS_utilization_optimal.shape= len(Number_slices),4

# plt.show()


# for s in range (n_servers):
#     #for i in range(len(Number_slices)):
#     if s ==0:
#         plt.plot(Number_slices,GOPS_utilization_random[:,s],'o-',label='random_edge#1')
#     elif s==1:
#         plt.plot(Number_slices, GOPS_utilization_random[:, s], 'bo-', label='random_edge#2')
#     elif s == 2:
#         plt.plot(Number_slices, GOPS_utilization_random[:, s], 'yo-', label='random_edge#3')
#     elif s ==3:
#         plt.plot(Number_slices, GOPS_utilization_random[:, s], 'go-', label='random_reg')

# # #
# # # for s in range (n_servers):
# # #     #for i in range(len(Number_slices)):
# # #     if s ==0:
# # #         plt.plot(Number_slices,GOPS_utilization_edge[:,s],'r^-',label='edge#1')
# # #     elif s==1:
# # #         plt.plot(Number_slices, GOPS_utilization_edge[:, s], 'b^-', label='edge#2')
# # #     elif s == 2:
# # #         plt.plot(Number_slices, GOPS_utilization_edge[:, s], 'y^-', label='edge#3')
# # #     elif s ==3:
# # #         plt.plot(Number_slices, GOPS_utilization_edge[:, s], 'g^-', label='regional')
# # # plt.ylabel("GOPS Load")
# # # plt.xlabel("#_users")
# # # plt.title('All_edge scenario')
# # # plt.legend()
# # # plt.grid()
# # # #plt.title("")GOPS_utilization_optimal.shape= len(Number_slices),4
# # #
# # # plt.show()
# # # for s in range (n_servers):
# # #     #for i in range(len(Number_slices)):
# # #     if s ==0:
# # #         plt.plot(Number_slices,GOPS_utilization_regional[:,s],'rx-',label='edge#1')
# # #     elif s==1:
# # #         plt.plot(Number_slices, GOPS_utilization_regional[:, s], 'bx-', label='edge#2')
# # #     elif s == 2:
# # #         plt.plot(Number_slices, GOPS_utilization_regional[:, s], 'yx-', label='edge#3')
# # #     elif s ==3:
# # #         plt.plot(Number_slices, GOPS_utilization_regional[:, s], 'gx-', label='regional')
# # #
# # #
# # #
# # # #plt.plot(ss,GOPS_utilization_edge,'bo-',label='All_edge')
# # # # plt.plot(ss,GOPS_utilization_optimal,'go-',label='Optimal')
# # # # plt.plot(ss,GOPS_utilization_regional,'yo-',label='Static')
# # # # ax.set_xticks(ss)
# # # # ax.set_xticklabels(servers)
# # # #plt.yticks(np.arange(0, 200, 10))
# # # plt.ylabel("GOPS Load")
# # # plt.xlabel("#_users")
# # # plt.title('Static scenario')
# # # plt.legend()
# # # plt.grid()
# # # #plt.title("")GOPS_utilization_optimal.shape= len(Number_slices),4
# # #
# # # plt.show()
# #
# # plt.plot(Number_slices,RB_utilization_1,'o-',label='RU#1')
# # plt.plot(Number_slices,RB_utilization_2,'bo-',label='RU#2')
# # plt.plot(Number_slices,RB_utilization_3,'yo-',label='RU#3')
# # plt.plot(Number_slices,RB_utilization_4,'go-',label='RU#4')
# # plt.ylabel("Radio resource load")
# # plt.xlabel("#_users")
# # plt.legend()
# # plt.grid()
# # #plt.title("")GOPS_utilization_optimal.shape= len(Number_slices),4
#
# plt.show()

lower_ratio_1 = []
upper_ratio_1 = []
lower_ratio_4 = []
upper_ratio_4 = []
lower_ratio_2 = []
upper_ratio_2 = []
lower_ratio_3 = []
upper_ratio_3 = []

for i in range(len(Number_slices)):
    lower_ratio_1.append(I_RB_1[i][0] if I_RB_1[i][0] < 100 else 100)
    upper_ratio_1.append(I_RB_1[i][1] if I_RB_1[i][1] < 100 else 100)
    lower_ratio_2.append(I_RB_2[i][0] if I_RB_2[i][0] < 100 else 100)
    upper_ratio_2.append(I_RB_2[i][1] if I_RB_2[i][1] < 100 else 100)
    lower_ratio_3.append(I_RB_3[i][0] if I_RB_3[i][0] < 100 else 100)
    upper_ratio_3.append(I_RB_3[i][1] if I_RB_3[i][1] < 100 else 100)
    lower_ratio_4.append(I_RB_4[i][0] if I_RB_4[i][0] < 100 else 100)
    upper_ratio_4.append(I_RB_4[i][1] if I_RB_4[i][1] < 100 else 100)

r = []
r1 = []
r0 = []
r2 = []

for lower,upper,y in zip(lower_ratio_1,upper_ratio_1,range(len(lower_ratio_1))):
    plt.plot((y,y),(lower,upper),'v:',color='tomato',label='RU#1',markersize=13)
    r0.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_2,upper_ratio_2,range(len(lower_ratio_2))):
    plt.plot((y,y),(lower,upper),'d--',color='royalblue',label='RU#2',markersize=13)
    r1.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_3,upper_ratio_3,range(len(lower_ratio_3))):
    plt.plot((y,y),(lower,upper),'^-.',color='y',label='RU#3',markersize=13)
    r2.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_4,upper_ratio_4,range(len(lower_ratio_4))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='RU#4',markersize=13)
    r.append(lower+(upper-lower) / 2)

plt.plot(r0,':',color='tomato')
plt.plot(r1,'--',color='royalblue')
plt.plot(r2,'-.',color='y')
plt.plot(r,'-',color='seagreen')

plt.xticks(range(len(lower_ratio_3)),Number_slices,fontsize=25)
plt.ylim([0,100])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('RB Allocation (%)',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

lower_ratio_1 = []
upper_ratio_1 = []
lower_ratio_4 = []
upper_ratio_4 = []
lower_ratio_2 = []
upper_ratio_2 = []
lower_ratio_3 = []
upper_ratio_3 = []

lower_ratio_all_edge = []
upper_ratio_all_edge = []

for i in range(len(Number_slices)):
    lower_ratio_1.append(I_GOPS_1[i][0])
    upper_ratio_1.append(I_GOPS_1[i][1] if I_GOPS_1[i][1] < 100 else 100)
    lower_ratio_2.append(I_GOPS_2[i][0])
    upper_ratio_2.append(I_GOPS_2[i][1] if I_GOPS_2[i][1] < 100 else 100)
    lower_ratio_3.append(I_GOPS_3[i][0])
    upper_ratio_3.append(I_GOPS_3[i][1] if I_GOPS_3[i][1] < 100 else 100)
    lower_ratio_4.append(I_GOPS_4[i][0])
    upper_ratio_4.append(I_GOPS_4[i][1] if I_GOPS_4[i][1] < 100 else 100)

    lower_ratio_all_edge.append(I_GOPS_all_edge[i][0])
    upper_ratio_all_edge.append(I_GOPS_all_edge[i][1] if I_GOPS_all_edge[i][1] < 100 else 100)

r = []
r1 = []
r0 = []
r2 = []

for lower,upper,y in zip(lower_ratio_1,upper_ratio_1,range(len(lower_ratio_1))):
    plt.plot((y,y),(lower,upper),'v:',color='tomato',label='edge#1',markersize=13)
    r0.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_2,upper_ratio_2,range(len(lower_ratio_2))):
    plt.plot((y,y),(lower,upper),'d--',color='royalblue',label='edge#2',markersize=13)
    r1.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_3,upper_ratio_3,range(len(lower_ratio_3))):
    plt.plot((y,y),(lower,upper),'^-.',color='y',label='edge#3',markersize=13)
    r2.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_4,upper_ratio_4,range(len(lower_ratio_4))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='regional',markersize=13)
    r.append(lower+(upper-lower) / 2)

plt.plot(r0,':',color='tomato')
plt.plot(r1,'--',color='royalblue')
plt.plot(r2,'-.',color='y')
plt.plot(r,'-',color='seagreen')

plt.xticks(range(len(lower_ratio_3)),Number_slices,fontsize=25)
plt.ylim([0,100])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('GOPS Allocation (%)',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

r = []
for lower,upper,y in zip(lower_ratio_all_edge,upper_ratio_all_edge,range(len(lower_ratio_all_edge))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='Optimal',markersize=13)
    r.append(lower+(upper-lower) / 2)
plt.plot(r,'-',color='seagreen')

plt.xticks(range(len(lower_ratio_all_edge)),Number_slices,fontsize=25)
plt.ylim([0,100])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('Total Edge GOPS Allocation (%)',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

# plt.plot(Number_slices,RB_1_TOTALL,'o-',label='RU#1')
# plt.plot(Number_slices,RB_2_TOTALL,'bo-',label='RU#2')
# plt.plot(Number_slices,RB_3_TOTALL,'yo-',label='RU#3')
# plt.plot(Number_slices,RB_4_TOTALL,'go-',label='RU#4')
# plt.ylabel("RB demand")
# plt.xlabel("Number of users")
# plt.legend()
# plt.grid()
# #plt.title("")GOPS_utilization_optimal.shape= len(Number_slices),4
#
# plt.show()

lower_ratio_1 = []
upper_ratio_1 = []
lower_ratio_4 = []
upper_ratio_4 = []
lower_ratio_2 = []
upper_ratio_2 = []
lower_ratio_3 = []
upper_ratio_3 = []

for i in range(len(Number_slices)):
    lower_ratio_1.append(I_RB_1_demand[i][0] if I_RB_1_demand[i][0] > 0 else 0)
    upper_ratio_1.append(I_RB_1_demand[i][1])
    lower_ratio_2.append(I_RB_2_demand[i][0] if I_RB_2_demand[i][0] > 0 else 0)
    upper_ratio_2.append(I_RB_2_demand[i][1])
    lower_ratio_3.append(I_RB_3_demand[i][0] if I_RB_3_demand[i][0] > 0 else 0)
    upper_ratio_3.append(I_RB_3_demand[i][1])
    lower_ratio_4.append(I_RB_4_demand[i][0] if I_RB_4_demand[i][0] > 0 else 0)
    upper_ratio_4.append(I_RB_4_demand[i][1])

r = []
r1 = []
r0 = []
r2 = []

for lower,upper,y in zip(lower_ratio_1,upper_ratio_1,range(len(lower_ratio_1))):
    plt.plot((y,y),(lower,upper),'v:',color='tomato',label='RU#1',markersize=13)
    r0.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_2,upper_ratio_2,range(len(lower_ratio_2))):
    plt.plot((y,y),(lower,upper),'d--',color='royalblue',label='RU#2',markersize=13)
    r1.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_3,upper_ratio_3,range(len(lower_ratio_3))):
    plt.plot((y,y),(lower,upper),'^-.',color='y',label='RU#3',markersize=13)
    r2.append(lower+(upper-lower) / 2)

for lower,upper,y in zip(lower_ratio_4,upper_ratio_4,range(len(lower_ratio_4))):
    plt.plot((y,y),(lower,upper),'o-',color='seagreen',label='RU#4',markersize=13)
    r.append(lower+(upper-lower) / 2)

plt.plot(r0,':',color='tomato')
plt.plot(r1,'--',color='royalblue')
plt.plot(r2,'-.',color='y')
plt.plot(r,'-',color='seagreen')

plt.xticks(range(len(lower_ratio_3)),Number_slices,fontsize=25)
# plt.ylim([0, 100])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('RB demand',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

lower_time_reduction = []
upper_time_reduction = []

for i in range(len(Number_slices)):
    lower_time_reduction.append(I_time_reduction[i][0])
    upper_time_reduction.append(I_time_reduction[i][1] if I_time_reduction[i][1] < 100 else 100)
r = []

for lower,upper,y in zip(lower_time_reduction,upper_time_reduction,range(len(lower_time_reduction))):
    plt.plot((y,y),(lower,upper),'o-',color='green',label='RNN-vs-ILP',markersize=13)
    r.append(lower+(upper-lower) / 2)

plt.plot(r,'-',color='g')

plt.xticks(range(len(lower_time_reduction)),Number_slices,fontsize=25)
plt.ylim([0,100])
plt.yticks(fontsize=25)
plt.xlabel('Number of users',fontsize=27)
plt.ylabel('Reduction in Execution time (in %)',fontsize=27)
handles,labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels,handles))
plt.legend(by_label.values(),by_label.keys(),fontsize=22)
plt.grid()
plt.show()

