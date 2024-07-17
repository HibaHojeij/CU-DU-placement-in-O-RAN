from docplex.mp.model import Model
import numpy as np
import random
import math
import platform
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import os

big_number = 100000
Nb_exp = 5000
dataset_collection={}
dataset = pd.DataFrame()
fairness_OPTIMAL = []


RB_utilization_1_all_runs = []  # rb utilization at each BS
RB_utilization_2_all_runs = []
RB_utilization_3_all_runs = []
RB_utilization_4_all_runs = []

GOPS_utilization_1_all_runs = []  # gops utilization at each server
GOPS_utilization_2_all_runs = []
GOPS_utilization_3_all_runs = []
GOPS_utilization_4_all_runs = []

COST_OPTIMAL = []
THROUGHPUT_OPTIMAL = []
ADMITTANCE_OPTIMAL = []


RATIO_eMBB_optimal = []

RATIO_uRLLC_optimal = []


RATIO_mMTC_optimal = []


ratio_CU_edge_eMbb_optimal_all_runs = []
ratio_CU_edge_uRLLC_optimal_all_runs = []
ratio_CU_edge_mMTC_optimal_all_runs = []

ratio_CU_reg_eMbb_optimal_all_runs = []
ratio_CU_reg_uRLLC_optimal_all_runs = []
ratio_CU_reg_mMTC_optimal_all_runs = []


GOPS_utilization_optimal_all_runs = []

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


    link_utilization_optimal = []

    RB_utilization_1 = []
    RB_utilization_2 = []
    RB_utilization_3 = []
    RB_utilization_4 = []

    GOPS_utilization_1 = []
    GOPS_utilization_2 = []
    GOPS_utilization_3 = []
    GOPS_utilization_4 = []


    GOPS_utilization_optimal = []

    ratio_admitted_optimal = []
    ratio_admitted_eMBB_optimal = []
    ratio_admitted_uRLLC_optimal = []
    ratio_admitted_mMTC_optimal = []


    ratio_CU_reg_eMbb_optimal = []
    ratio_CU_reg_uRLLC_optimal = []
    ratio_CU_reg_MTC_optimal = []
    ratio_CU_edge_eMbb_optimal = []
    ratio_CU_edge_uRLLC_optimal = []
    ratio_CU_edge_MTC_optimal = []

    cost_optimal = []

    total_throughput_optimal = []

    fairness_optimal = []

    n_servers = 4

    n_servers_edge = 3

    N = np.arange(20, 120, 20)
    Number_slices = [n for n in N]
    # Number_slices=[20]
    ###servers locations are fixed for all the experiments even when increasing the nb of UEs
    x_center = 0.500
    y_center = 0.500
    servers_edge = [i for i in range(n_servers_edge)]
    for i in range(n_servers_edge):
        r = random.randint(5, 10)
        ang = random.uniform(0, 1) * 2 * math.pi
        loc_x_edge.append(r * math.cos(ang) + x_center)
        loc_y_edge.append(r * math.sin(ang) + y_center)

    for i in range(n_servers_edge, n_servers):
        r = random.randint(40, 80)
        ang = random.uniform(0, 1) * 2 * math.pi
        loc_x_reg.append(r * math.cos(ang) + x_center)
        loc_y_reg.append(r * math.sin(ang) + y_center)

    for NN in Number_slices:

        # print ('n_slices=',NN )

        m = Model(name='RA-ORAN_optimal')
        m.parameters.timelimit = 60

        n_servers = 4

        Nb_BS = 4  # number of base_stations(supposed, it can be between a random interval
        RB_utilization = []

        # 25,25,50 INDUSTRIAL area
        # 50,30,20

        if (NN / 10) % 2 == 0:
            n_eMBB = int(NN * 0.25)
        else:
            n_eMBB = int(NN * 0.25) + 1

        n_uRLLC = int(NN * 0.25)
        n_MTC = int(NN * 0.5)

        n_slices = n_eMBB + n_MTC + n_uRLLC

        servers = [i for i in range(n_servers)]
        slices = [i for i in range(n_slices)]

        i = 0
        S = np.empty([n_servers * n_servers, 2], dtype=int)
        for s in range(n_servers):
            for s1 in range(n_servers):
                S[i] = (s, s1)
                i = i + 1
        loc_x_BS = []
        loc_y_BS = []
        #         loc_x_reg=[]
        #         loc_y_reg=[]
        #         loc_x_edge=[]
        #         loc_y_edge=[]
        rnd = np.random
        # rnd.seed(0)
        loc_x = rnd.rand(n_slices) * 1
        loc_y = rnd.rand(n_slices) * 1
        loc_x_BS = [0.250, 0.250, 0.750, 0.750]
        loc_y_BS = [0.250, 0.750, 0.250, 0.750]

        RB_total = 100 * Nb_BS  # 20MHZ bandwith= 100 RBs each BS
        N_RB_rem_embb = [50, 50, 50, 50]
        N_RB_rem_urllc = [25, 25, 25, 25]
        N_RB_rem_mmtc = [25, 25, 25, 25]

        #         N_RB_rem_embb= [250,250,250,250]
        #         N_RB_rem_urllc= [125,125,125,125]
        #         N_RB_rem_mmtc= [125,125,125,125]

        RB_eMBB = []
        RB_uRLLC = []
        RB_MTC = []
        for i in range(n_slices):
            if i < n_eMBB:
                if loc_x[i] < 0.500 and loc_y[i] < 0.500:  # this user is ascociated to the first BS
                    r = random.randint(10, 20)
                    if N_RB_rem_embb[0] - r >= 0:
                        RB_eMBB.append(r)
                        N_RB_rem_embb[0] -= r
                    elif N_RB_rem_embb[0] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_eMBB.append(N_RB_rem_embb[0])
                        N_RB_rem_embb[0] = 0
                    else:
                        RB_eMBB.append(0)
                elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                    r = random.randint(10, 20)
                    if N_RB_rem_embb[1] - r >= 0:
                        RB_eMBB.append(r)
                        N_RB_rem_embb[1] -= r
                    elif N_RB_rem_embb[1] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_eMBB.append(N_RB_rem_embb[1])
                        N_RB_rem_embb[1] = 0
                    else:
                        RB_eMBB.append(0)
                elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                    r = random.randint(10, 20)
                    if N_RB_rem_embb[2] - r >= 0:
                        RB_eMBB.append(r)
                        N_RB_rem_embb[2] -= r
                    elif N_RB_rem_embb[2] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_eMBB.append(N_RB_rem_embb[2])
                        N_RB_rem_embb[2] = 0
                    else:
                        RB_eMBB.append(0)
                elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                    r = random.randint(10, 20)
                    if N_RB_rem_embb[3] - r >= 0:
                        RB_eMBB.append(r)
                        N_RB_rem_embb[3] -= r
                    elif N_RB_rem_embb[3] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_eMBB.append(N_RB_rem_embb[3])
                        N_RB_rem_embb[3] = 0
                    else:
                        RB_eMBB.append(0)
            elif i < n_eMBB + n_uRLLC:
                if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1, 5)
                    if N_RB_rem_urllc[0] - r >= 0:
                        RB_uRLLC.append(r)
                        N_RB_rem_urllc[0] -= r
                    elif N_RB_rem_urllc[0] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_uRLLC.append(N_RB_rem_urllc[0])
                        N_RB_rem_urllc[0] = 0
                    else:
                        RB_uRLLC.append(0)
                elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1, 5)
                    if N_RB_rem_urllc[1] - r >= 0:
                        RB_uRLLC.append(r)
                        N_RB_rem_urllc[1] -= r
                    elif N_RB_rem_urllc[1] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_uRLLC.append(N_RB_rem_urllc[1])
                        N_RB_rem_urllc[1] = 0
                    else:
                        RB_uRLLC.append(0)
                elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1, 5)
                    if N_RB_rem_urllc[2] - r >= 0:
                        RB_uRLLC.append(r)
                        N_RB_rem_urllc[2] -= r
                    elif N_RB_rem_urllc[2] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_uRLLC.append(N_RB_rem_urllc[2])
                        N_RB_rem_urllc[2] = 0
                    else:
                        RB_uRLLC.append(0)
                elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1, 5)
                    if N_RB_rem_urllc[3] - r >= 0:
                        RB_uRLLC.append(r)
                        N_RB_rem_urllc[3] -= r
                    elif N_RB_rem_urllc[3] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_uRLLC.append(N_RB_rem_urllc[3])
                        N_RB_rem_urllc[3] = 0
                    else:
                        RB_uRLLC.append(0)
            elif i < n_eMBB + n_uRLLC + n_MTC:
                if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1, 5)
                    if N_RB_rem_mmtc[0] - r >= 0:
                        RB_MTC.append(r)
                        N_RB_rem_mmtc[0] -= r
                    elif N_RB_rem_mmtc[0] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_MTC.append(N_RB_rem_mmtc[0])
                        N_RB_rem_mmtc[0] = 0
                    else:
                        RB_MTC.append(0)
                elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                    r = random.randint(1, 5)
                    if N_RB_rem_mmtc[1] - r >= 0:
                        RB_MTC.append(r)
                        N_RB_rem_mmtc[1] -= r
                    elif N_RB_rem_mmtc[1] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_MTC.append(N_RB_rem_mmtc[1])
                        N_RB_rem_mmtc[1] = 0
                    else:
                        RB_MTC.append(0)
                elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1, 5)
                    if N_RB_rem_mmtc[2] - r >= 0:
                        RB_MTC.append(r)
                        N_RB_rem_mmtc[2] -= r
                    elif N_RB_rem_mmtc[2] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_MTC.append(N_RB_rem_mmtc[2])
                        N_RB_rem_mmtc[2] = 0
                    else:
                        RB_MTC.append(0)
                elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                    r = random.randint(1, 5)
                    if N_RB_rem_mmtc[3] - r >= 0:
                        RB_MTC.append(r)
                        N_RB_rem_mmtc[3] -= r
                    elif N_RB_rem_mmtc[3] > 0:  # giving the remaining RBs to the lastslice(to not waste RBs)
                        RB_MTC.append(N_RB_rem_mmtc[3])
                        N_RB_rem_mmtc[3] = 0

                    else:
                        RB_MTC.append(0)
        RB_slices = np.concatenate((RB_eMBB, RB_uRLLC, RB_MTC))
        # print("N_slices=",NN)
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
        RB_utilization_1.append(RB_1)  # out of 100 RBs of each BS
        RB_utilization_2.append(RB_2)
        RB_utilization_3.append(RB_3)
        RB_utilization_4.append(RB_4)
        ########now runnning 3 different scenarios but with same topology of the network

        n_servers = 4
        n_servers_edge = 3
        i = 0
        S = np.empty([n_servers * n_servers, 2], dtype=int)
        for s in range(n_servers):
            for s1 in range(n_servers):
                S[i] = (s, s1)
                i = i + 1

        # desicion variables
        theta_CU_optimal = m.binary_var_matrix(range(n_slices), range(n_servers), name='theta_CU')
        theta_DU_optimal = m.binary_var_matrix(range(n_slices), range(n_servers), name='theta_DU')

        c1 = m.add_constraints(
            m.sum(theta_CU_optimal[i, s] for s in range(n_servers)) <= 1 for i in range(n_slices))
        c2 = m.add_constraints(
            m.sum(theta_DU_optimal[i, s] for s in range(n_servers)) <= 1 for i in range(n_slices))
        c3 = m.add_constraints(
            m.sum(theta_DU_optimal[i, s] for s in range(n_servers_edge, n_servers)) == 0 for i in
            range(n_slices))  # not necessarly a solution can be found for DU
        # c3=m.add_constraints(m.sum(theta_DU[i,s] for s in range(0,n_servers_edge)) == 1 for i in range(n_slices))
        c55 = m.add_constraints(m.sum(theta_CU_optimal[i, s] for s in range(n_servers)) == m.sum(
            theta_DU_optimal[i, s] for s in range(n_servers)) for i in range(n_slices))


        for i in range(n_slices):
            if RB_slices[i] == 0:
                m.add_constraints(theta_DU_optimal[i, s] == 0 for s in range(n_servers))
                m.add_constraints(theta_CU_optimal[i, s] == 0 for s in range(n_servers))

        #############link_capacity_constraint
        N_sym = 14  # number of symbols per sub-frame
        N_SC = 12  # number of subcarrier per RB
        A = 4  # number of Antennas
        BTW = 32  # number of I Q bits
        l = [i for i in range(n_slices)]  # midhaul link capacity needed by each slice btw CU and DU
        for i in range(n_slices):
            if i < n_eMBB:
                l[i] = N_SC * N_sym * RB_eMBB[i] * A * BTW / 1000000  # Gbps1
            elif i < (n_eMBB + n_uRLLC):
                l[i] = N_SC * N_sym * RB_uRLLC[i - n_eMBB] * A * BTW / 1000000  # Gbps
            else:
                l[i] = N_SC * N_sym * RB_MTC[i - n_eMBB - n_uRLLC] * A * BTW / 1000000  # Gbps
        ############Model capacity C between servers
        Capacity = np.ones((n_servers, n_servers))  # link capacity available between 2 servers
        for s in range(n_servers):
            for s1 in range(n_servers):
                if s1 != s:
                    if s < n_servers_edge:
                        if s1 < n_servers_edge:
                            Capacity[s][s1] = random.randint(1, 10)  # Gbps edge-edge (max range = 10Gbps)
                        else:
                            Capacity[s][s1] = random.randint(10, 20)  # edge-regional (max range = 20 Gbps)
                    else:
                        if s1 < n_servers_edge:
                            Capacity[s][s1] = random.randint(10, 20)  # reg-edge
                else:  # same server=> link capacity very high number suppose 1000(exchange of information in the same server)
                    Capacity[s][s1] = 1000

        c4 = m.add_constraints(
            m.sum((theta_DU_optimal[i, s] + theta_CU_optimal[i, s1] - 1) * l[i] for i in range(n_slices)) <=
            Capacity[s][s1] for s, s1 in S)

        #################c5_server_capacity constraint
        R = np.ones(n_servers)
        R_DU = np.ones(n_slices)
        R_CU = np.ones(n_slices)
        alpha_DU = 0.4  # Scaling factor for the DU functionalities over all funct
        alpha_CU = 0.1  #
        Code_rate = [438, 466, 517, 567, 616, 666, 719, 772, 822, 873, 910, 948]
        spectral_eff = [2.5664, 2.7305, 3.0293, 3.3223, 3.6094, 3.9023, 4.2129, 4.5234, 4.8164, 5.1152, 5.3320,
                        5.5547]
        C = []
        S_E = []
        MMM = []#just to save mcs value
        for i in range(n_slices):

            MCS = random.randint(17, 28)
            MMM.append(MCS)
            C.append(Code_rate[MCS - 17] / 1024)
            S_E.append(spectral_eff[MCS - 17])
        M = 6  # modulation bits log2(64)
        L = 2  # number of MIMO layers
        A = 4  # number of Antennas
        ###total computetional power demand by all experiment can be 1800GOPS

        for i in range(n_servers):
            if i < n_servers_edge:
                R[i] = random.randint(100, 200)  # GOPS available at edge cloud servers
            else:
                R[i] = random.randint(1000, 2000)  # GOPS available at regional cloud servers

        for i in range(n_slices):
            if i < n_eMBB:
                R_DU[i] = alpha_DU * (3 * A + A ** 2 + M * C[i] * L / 3) * RB_eMBB[
                    i] / 10  # GOPS needed by DU functionalities for slice eMBB
            elif i < (n_eMBB + n_uRLLC):
                R_DU[i] = alpha_DU * (3 * A + A ** 2 + M * C[i] * L / 3) * RB_uRLLC[i - n_eMBB] / 10
            else:
                R_DU[i] = alpha_DU * (3 * A + A ** 2 + M * C[i] * L / 3) * RB_MTC[i - n_eMBB - n_uRLLC] / 10

        for i in range(n_slices):
            if i < n_eMBB:
                R_CU[i] = alpha_CU * (3 * A + A ** 2 + M * C[i] * L / 3) * RB_eMBB[
                    i] / 10  # GOPS needed by CU functionalities for slice eMBB
            elif i < (n_eMBB + n_uRLLC):
                R_CU[i] = alpha_CU * (3 * A + A ** 2 + M * C[i] * L / 3) * RB_uRLLC[i - n_eMBB] / 10
            else:
                R_CU[i] = alpha_CU * (3 * A + A ** 2 + M * C[i] * L / 3) * RB_MTC[i - n_eMBB - n_uRLLC] / 10


        c5 = m.add_constraints((m.sum(
            (theta_DU_optimal[i, s] * R_DU[i] + theta_CU_optimal[i, s] * R_CU[i]) for i in range(n_slices)) <=
                                R[s] for s in range(n_servers)))


        # c6 Latency constraint
        delta = np.zeros((n_servers, n_servers))
        delta_max = np.ones(n_slices)
        for i in range(n_slices):
            if i < n_eMBB:
                delta_max[i] = 500  # random.randint(4,10)*1000 #micro seconds
            elif i < (n_eMBB + n_uRLLC):
                delta_max[i] = random.randint(100, 300)  # 100 #random.randint(1,2)*1000
            else:
                delta_max[i] = 1000  # random.randint(2,4)*1000

        for s in range(n_servers):
            for s1 in range(n_servers):
                if s < n_servers_edge:
                    if s1 < n_servers_edge:
                        delta[s][s1] = 5 * math.sqrt((loc_x_edge[s] - loc_x_edge[s1]) ** 2 + (
                                loc_y_edge[s] - loc_y_edge[
                            s1]) ** 2)  # random.randint(50,100)  # us edge-edge (5us/km => 10-20 km betwwen edge-edge server)
                    else:
                        delta[s][s1] = 5 * math.sqrt((loc_x_edge[s] - loc_x_reg[s1 - n_servers_edge]) ** 2 + (
                                loc_y_edge[s] - loc_y_reg[
                            s1 - n_servers_edge]) ** 2)  # random.randint(200,400) #edge-regional(5us/km => 40-80 km betwwen edge-edge server)
                else:
                    if s1 < n_servers_edge:
                        delta[s][s1] = 5 * math.sqrt((loc_x_edge[s1] - loc_x_reg[s - n_servers_edge]) ** 2 + (
                                loc_y_edge[s1] - loc_y_reg[
                            s - n_servers_edge]) ** 2)  # random.randint(200,400) #reg-edge
                #             else:
                #                 delta[s][s1]= random.randint(1000,2000) #reg-reg
                if s == s1:
                    delta[s][s1] = 0


        c6 = m.add_constraints(
            delta[s, s1] * (theta_CU_optimal[i, s] + theta_DU_optimal[i, s1] - 1) <= delta_max[i] for i in
            range(n_slices) for s, s1 in S)
            # c6=m.add_constraints(delta[s,s1]*(theta_DU_optimal[i,s])<=delta_max[i] for i in range(n_slices) for s,s1 in S)

        nearest_edge = np.ones(Nb_BS)
        relative_dist_BS_server =[]
        for b in range(Nb_BS):
            for s in range(n_servers):
                if s< n_servers_edge:
                    relative_dist_BS_server.append ( math.sqrt((loc_x_BS[b] - loc_x_edge[s]) ** 2 + (loc_y_BS[b] - loc_y_edge[s]) ** 2))
                else:
                    relative_dist_BS_server.append ( math.sqrt((loc_x_BS[b] - loc_x_reg[s-n_servers_edge]) ** 2 + (loc_y_BS[b] - loc_y_reg[s-n_servers_edge]) ** 2))

        relative_dist_server_server= []
        loc_x_servers = loc_x_edge+loc_x_reg
        loc_y_servers = loc_y_edge+loc_y_reg

        relative_dist_server_server.append(math.sqrt((loc_x_servers[0] - loc_x_servers[1]) ** 2 + (loc_y_servers[0] - loc_y_servers[1]) ** 2))
        relative_dist_server_server.append(math.sqrt((loc_x_servers[0] - loc_x_servers[2]) ** 2 + (loc_y_servers[0] - loc_y_servers[2]) ** 2))
        relative_dist_server_server.append(math.sqrt((loc_x_servers[0] - loc_x_servers[3]) ** 2 + (loc_y_servers[0] - loc_y_servers[3]) ** 2))
        relative_dist_server_server.append(math.sqrt((loc_x_servers[1] - loc_x_servers[2]) ** 2 + (loc_y_servers[1] - loc_y_servers[2]) ** 2))
        relative_dist_server_server.append(math.sqrt((loc_x_servers[1] - loc_x_servers[3]) ** 2 + (loc_y_servers[1] - loc_y_servers[3]) ** 2))
        relative_dist_server_server.append(math.sqrt((loc_x_servers[2] - loc_x_servers[3]) ** 2 + (loc_y_servers[2] - loc_y_servers[3]) ** 2))

        BS_association=[]
        for i in range(n_slices):
            if loc_x[i] < 0.500 and loc_y[i] < 0.500:  # this user is ascociated to the first BS
                BS_association.append(0)
            elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                BS_association.append(1)
            elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                BS_association.append(2)
            elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                BS_association.append(3)


        for b in range(Nb_BS):
            minimum = -1
            nearest = 0
            for s in range(n_servers_edge):
                distance = math.sqrt((loc_x_BS[b] - loc_x_edge[s]) ** 2 + (loc_y_BS[b] - loc_y_edge[s]) ** 2)
                if distance < minimum:
                    minimum = distance
                    nearest = s
            nearest_edge[b] = s  # save for each bs the nearest edge server

        C_F = np.ones((n_slices, n_servers))  # link capacity available between 2 servers
        for i in range(n_slices):
            for s in range(n_servers):
                if s < n_servers_edge:
                    if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                        C_F[i][s] = 1 / math.sqrt(
                            (loc_x_BS[0] - loc_x_edge[s]) ** 2 + (loc_y_BS[0] - loc_y_edge[s]) ** 2)  # edge
                    elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                        C_F[i][s] = 1 / math.sqrt(
                            (loc_x_BS[1] - loc_x_edge[s]) ** 2 + (loc_y_BS[1] - loc_y_edge[s]) ** 2)  # edge
                    elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                        C_F[i][s] = 1 / math.sqrt(
                            (loc_x_BS[2] - loc_x_edge[s]) ** 2 + (loc_y_BS[2] - loc_y_edge[s]) ** 2)  # edge
                    elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                        C_F[i][s] = 1 / math.sqrt(
                            (loc_x_BS[3] - loc_x_edge[s]) ** 2 + (loc_y_BS[3] - loc_y_edge[s]) ** 2)  # edge
                else:
                    C_F[i][s] = 1  # regional

        priority_matrix = np.ones(n_slices)
        for i in range(n_slices):
            if i < n_eMBB:
                priority_matrix[i] = 10  # to maximize the throughput we give higher priority for eMBB users
            elif i < n_eMBB + n_uRLLC:
                priority_matrix[i] = 10  # 3
            else:
                priority_matrix[i] = 1

        ###OBJECTIVEEEEE


        # user_satisfaction = m.sum(C_F[i, s] * theta_CU_optimal[i, s] + C_F[i, s] * theta_DU_optimal[i, s] for i in range(n_slices) for s in range(n_servers))
        user_satisfaction = m.sum(
            C_F[i, s] * theta_CU_optimal[i, s] * priority_matrix[i] + C_F[i, s] * theta_DU_optimal[i, s] *
            priority_matrix[i] for i in range(n_slices) for s in range(n_servers))
        # user_satisfaction= m.sum(C_F[i,s]*theta_CU_optimal[i,s]  for i in range(n_slices) for s in range(n_servers))
        # user_satisfaction= m.sum(theta_CU_optimal[i,s] for i in range(n_slices) for s in range(n_servers))
        m.maximize(user_satisfaction)
        solution = m.solve()
        objectivee= m.objective_value


        Cost_matrix = np.ones(n_servers)  # link capacity available between 2 servers

        for s in range(n_servers):
            if s < n_servers_edge:
                Cost_matrix[s] = 1.59  # edge
            else:
                Cost_matrix[s] = 0.5  # regional

        solution_theta_DU = np.ones((n_slices, n_servers))
        for i in range(n_slices):
            for j in range(n_servers):
                solution_theta_DU[i][j] = (m.solution.get_value("theta_DU_{}_{}".format(i, j)))

        solution_theta_CU = np.ones((n_slices, n_servers))
        for i in range(n_slices):
            for j in range(n_servers):
                solution_theta_CU[i][j] = (m.solution.get_value("theta_CU_{}_{}".format(i, j)))


        ##throughput calculation
        throughput = 0
        for i in range(n_slices):
            for s in range(n_servers):
                # throughput += N_SC*N_sym*RB_slices[i]*S_E[i]*L*1600*solution_theta_CU[i,s]/1000000##in Mbps
                throughput += L * M * C[i] * N_SC * N_sym * RB_slices[i] * solution_theta_CU[i, s] * (
                        1 - 0.14) / 1000  ##in Mbps

        n_admitted_eMBB = sum(solution_theta_CU[i, s] for i in range(n_eMBB) for s in range(n_servers))
        n_admitted_uRLLC = sum(
            solution_theta_CU[i, s] for i in range(n_eMBB, n_eMBB + n_uRLLC) for s in range(n_servers))
        n_admitted_mMTC = sum(
            solution_theta_CU[i, s] for i in range(n_eMBB + n_uRLLC, n_slices) for s in range(n_servers))

        ######fairness index of admitted slices
        s = n_admitted_eMBB / n_eMBB + n_admitted_mMTC / n_MTC + n_admitted_uRLLC / n_uRLLC
        x = (n_admitted_eMBB / n_eMBB) ** 2 + (n_admitted_mMTC / n_MTC) ** 2 + (n_admitted_uRLLC / n_uRLLC) ** 2

        fairness = (s * s) / (3 * x)

        linkk_util = np.zeros((n_servers, n_servers))
        GOPS_util = np.zeros(n_servers)
        # just targeting to collect GOPS load for n=140 users

        for s, s1 in S:
            linkk_util[s][s1] = sum(
                l[i] * solution_theta_CU[i, s] * solution_theta_DU[i, s1] for i in range(n_slices)) / Capacity[s][
                                    s1] * 100
        for s in range(n_servers):
            GOPS_util[s] = ((sum(R_CU[i] * solution_theta_CU[i, s] for i in range(n_slices)) + sum(
                R_DU[i] * solution_theta_DU[i, s] for i in range(n_slices))) / R[s]) * 100

        # objectivee=np.append(objectivee,m.objective_value)
        Nb_servers_edge_CU = np.ones(3)
        Nb_servers_reg_CU = np.ones(3)
        Nb_servers_edge_CU = [sum(solution_theta_CU[i, s] for i in range(n_eMBB) for s in range(n_servers_edge)),
                              sum(solution_theta_CU[i, s] for i in range(n_eMBB, n_eMBB + n_uRLLC) for s in
                                  range(n_servers_edge)), sum(
                solution_theta_CU[i, s] for i in range(n_eMBB + n_uRLLC, n_slices) for s in range(n_servers_edge))]
        Nb_servers_reg_CU = [
            sum(solution_theta_CU[i, s] for i in range(n_eMBB) for s in range(n_servers_edge, n_servers)), sum(
                solution_theta_CU[i, s] for i in range(n_eMBB, n_eMBB + n_uRLLC) for s in
                range(n_servers_edge, n_servers)), sum(
                solution_theta_CU[i, s] for i in range(n_eMBB + n_uRLLC, n_slices) for s in
                range(n_servers_edge, n_servers))]

        # CC = sum(
        #     Cost_matrix[s] * R_CU[i] * solution_theta_CU[i, s] for i in range(n_slices) for s in range(n_servers))
        # we take into consideration the cost of deployment of just CU because the DU is always at the edge.
        CC = sum(Cost_matrix[s] * R_CU[i] * solution_theta_CU[i, s] for i in range(n_slices) for s in
                 range(n_servers)) + sum(
            Cost_matrix[s] * R_DU[i] * solution_theta_DU[i, s] for i in range(n_slices) for s in range(n_servers))

        link_utilization_optimal.append(linkk_util)
        GOPS_utilization_optimal = np.append(GOPS_utilization_optimal, GOPS_util)
        total_throughput_optimal.append(throughput)

        ratio_CU_edge_eMbb_optimal.append(
            Nb_servers_edge_CU[0] / n_admitted_eMBB * 100 if n_admitted_eMBB > 0 else 0)
        ratio_CU_edge_uRLLC_optimal.append(
            Nb_servers_edge_CU[1] / n_admitted_uRLLC * 100 if n_admitted_uRLLC > 0 else 0)
        ratio_CU_edge_MTC_optimal.append(
            Nb_servers_edge_CU[2] / n_admitted_mMTC * 100 if n_admitted_mMTC > 0 else 0)

        ratio_CU_reg_eMbb_optimal.append(
            Nb_servers_reg_CU[0] / n_admitted_eMBB * 100 if n_admitted_eMBB > 0 else 0)
        ratio_CU_reg_uRLLC_optimal.append(
            Nb_servers_reg_CU[1] / n_admitted_uRLLC * 100 if n_admitted_uRLLC > 0 else 0)
        ratio_CU_reg_MTC_optimal.append(
            Nb_servers_reg_CU[2] / n_admitted_mMTC * 100 if n_admitted_mMTC > 0 else 0)

        ratio_admitted_optimal.append(
            sum(solution_theta_CU[i, s] for i in range(n_slices) for s in range(n_servers)) / n_slices * 100)
        ratio_admitted_eMBB_optimal.append(
            sum(solution_theta_CU[i, s] for i in range(n_eMBB) for s in range(n_servers)) / n_eMBB * 100)
        ratio_admitted_uRLLC_optimal.append(sum(
            solution_theta_CU[i, s] for i in range(n_eMBB, n_eMBB + n_uRLLC) for s in
            range(n_servers)) / n_uRLLC * 100)
        ratio_admitted_mMTC_optimal.append(sum(
            solution_theta_CU[i, s] for i in range(n_eMBB + n_uRLLC, n_slices) for s in
            range(n_servers)) / n_MTC * 100)

        cost_optimal.append(CC)

        fairness_optimal.append(fairness)

        GOPS_utilization_1.append(GOPS_util[0])
        GOPS_utilization_2.append(GOPS_util[1])
        GOPS_utilization_3.append(GOPS_util[2])
        GOPS_utilization_4.append(GOPS_util[3])

#the dataframe row are users and columns are the features

        GOPS_av =[]
        LOC=[]
        LOC_new=[]
        Capacity_av=[]
        delta_av=[]
        # GOPS_av = np.tile(R, (n_slices, 1))
        # CCC = Capacity.flatten()  # from 2d array to 1d
        # ddd= delta.flatten()
        loc_x_EDGE=[]
        loc_y_EDGE=[]
        loc_x_REG= []
        loc_y_REG= []
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
            ind_CU=ind_DU=0
            #print(solution_theta_CU)
            temp = np.zeros(12) #12 elements vector for the location

            total_number_users.append(NN)


            if (1 in solution_theta_CU[i]) and (1 in solution_theta_DU[i]) :
                ind_CU=np.where(solution_theta_CU[i] ==1 )[0][0]
                ind_DU=np.where(solution_theta_DU[i] ==1 )[0][0]
                #temp[ind_DU+ind_CU*3]=1
                #LOC.append(temp)
                LOC.append([ind_CU+1,ind_DU+1])

            else:
               # LOC.append(np.zeros(12))
                LOC.append(np.zeros(2))

        for ll in LOC:
            if ll[0]!=0 and ll[1]!=0:
                LOC_new.append( (ll[1]-1 + (ll[0]-1)*3 + 1))
            else:
                LOC_new.append(0)

        slice_type = np.concatenate(([1] * n_eMBB, [2] * n_uRLLC, [3] * n_MTC))

        C_F = np.ones((n_slices, n_servers))  # link capacity available between 2 servers

        for i in range(n_slices):
            for s in range(n_servers):
                if s < n_servers_edge:
                    if loc_x[i] < 0.500 and loc_y[i] < 0.500:
                        C_F[i][s] = 1 / math.sqrt(
                            (loc_x_BS[0] - loc_x_edge[s]) ** 2 + (loc_y_BS[0] - loc_y_edge[s]) ** 2)  # edge
                    elif loc_x[i] > 0.500 and loc_y[i] < 0.500:
                        C_F[i][s] = 1 / math.sqrt(
                            (loc_x_BS[1] - loc_x_edge[s]) ** 2 + (loc_y_BS[1] - loc_y_edge[s]) ** 2)  # edge
                    elif loc_x[i] < 0.500 and loc_y[i] > 0.500:
                        C_F[i][s] = 1 / math.sqrt(
                            (loc_x_BS[2] - loc_x_edge[s]) ** 2 + (loc_y_BS[2] - loc_y_edge[s]) ** 2)  # edge
                    elif loc_x[i] > 0.500 and loc_y[i] > 0.500:
                        C_F[i][s] = 1 / math.sqrt(
                            (loc_x_BS[3] - loc_x_edge[s]) ** 2 + (loc_y_BS[3] - loc_y_edge[s]) ** 2)  # edge
                else:
                    C_F[i][s] = 1  # regional

        priority_matrix = np.ones(n_slices)
        for i in range(n_slices):
            if i < n_eMBB:
                priority_matrix[i] = 10  # to maximize the throughput we give higher priority for eMBB users
            elif i < n_eMBB + n_uRLLC:
                priority_matrix[i] = 10  # 3
            else:
                priority_matrix[i] = 1

        df1 = pd.DataFrame({'user_loc_x': loc_x,
                        'user_loc_y': loc_y,
                        # 'RU_loc':[],
                        'loc_edge_x_1': [loc_x_EDGE[0][0]]*n_slices,
                        'loc_edge_x_2': [loc_x_EDGE[0][1]]*n_slices,
                        'loc_edge_x_3': [loc_x_EDGE[0][2]]*n_slices,
                        'loc_edge_y_1': [loc_y_EDGE[0][0]]*n_slices,
                        'loc_edge_y_2': [loc_y_EDGE[0][1]]*n_slices,
                        'loc_edge_y_3': [loc_y_EDGE[0][2]]*n_slices,
                        'loc_reg_x': [loc_x_REG[0][0]]*n_slices,
                        'loc_reg_y': [loc_y_REG[0][0]]*n_slices,

                        'USer_association_BS':BS_association,

                        'rel_dist_Server_BS_0': relative_dist_BS_server[0],
                        'rel_dist_Server_BS_1': relative_dist_BS_server[1],
                        'rel_dist_Server_BS_2': relative_dist_BS_server[2],
                        'rel_dist_Server_BS_3': relative_dist_BS_server[3],
                        'rel_dist_Server_BS_4': relative_dist_BS_server[4],
                        'rel_dist_Server_BS_5': relative_dist_BS_server[5],
                        'rel_dist_Server_BS_6': relative_dist_BS_server[6],
                        'rel_dist_Server_BS_7': relative_dist_BS_server[7],
                        'rel_dist_Server_BS_8': relative_dist_BS_server[8],
                        'rel_dist_Server_BS_9': relative_dist_BS_server[9],
                        'rel_dist_Server_BS_10': relative_dist_BS_server[10],
                        'rel_dist_Server_BS_11': relative_dist_BS_server[11],
                        'rel_dist_Server_BS_12': relative_dist_BS_server[12],
                        'rel_dist_Server_BS_13': relative_dist_BS_server[13],
                        'rel_dist_Server_BS_14': relative_dist_BS_server[14],
                        'rel_dist_Server_BS_15': relative_dist_BS_server[15],

                        'rel_dist_ser_ser_0': relative_dist_server_server[0],
                        'rel_dist_ser_ser_1': relative_dist_server_server[1],
                        'rel_dist_ser_ser_2': relative_dist_server_server[2],
                        'rel_dist_ser_ser_3': relative_dist_server_server[3],
                        'rel_dist_ser_ser_4': relative_dist_server_server[4],
                        'rel_dist_ser_ser_5': relative_dist_server_server[5],

                        'RB': RB_slices,
                        'MCS': MMM,
                        # 'slice': [],
                        'GOPS_available_1': [GOPS_av[0][0]]*n_slices,
                        'GOPS_available_2': [GOPS_av[0][1]]*n_slices,
                        'GOPS_available_3': [GOPS_av[0][2]]*n_slices,
                        'GOPS_available_4': [GOPS_av[0][3]]*n_slices,
                        'GOPS_required_CU': R_CU,

                        'GOPS_required_DU': R_DU,

                        # 'link_cap_av': Capacity_av,
                        # 'link_required': l,
                        # 'link_latency': delta_av, #can be learned from the locations of server and user, i suppose no need for it
                        'link_latency_0_0': [delta[0][0]]*n_slices,
                        'link_latency_0_1': [delta[0][1]]*n_slices,
                        'link_latency_0_2': [delta[0][2]]*n_slices,
                        'link_latency_0_3': [delta[0][3]]*n_slices,
                        # 'link_latency_1_0': [delta[1][0]]*n_slices,
                        'link_latency_1_1': [delta[1][1]]*n_slices,
                        'link_latency_1_2': [delta[1][2]]*n_slices,
                        'link_latency_1_3': [delta[1][3]]*n_slices,
                        'link_latency_2_2': [delta[2][2]]*n_slices,
                        'link_latency_2_3': [delta[2][3]]*n_slices,
                        'link_latency_3_3': [delta[3][3]]*n_slices,
                        'TOTAL_GOPS_required_CU': [np.sum(R_CU)] * n_slices,
                        'TOTAL_GOPS_required_DU': [np.sum(R_DU)] * n_slices,
                        'max_latency': delta_max,
                        'total_num_users': total_number_users,
                        'slice_type': slice_type,
                        'priority': priority_matrix,
                        'C_F_s_0': C_F[:,0],
                        'C_F_s_1': C_F[:,1],
                        'C_F_s_2': C_F[:,2],
                        'C_F_s_3': C_F[:,3],
                        # 'objective_value':objectivee,
                        'LOCATION': LOC_new})

        if NN<100:
            # Desired number of rows
            desired_rows = 100

            # Calculate the difference between the current number of rows and the desired number
            difference = desired_rows - len(df1)

            # Create a new dataframe with zeros
            zeros_df = pd.DataFrame(0, index=range(difference), columns=df1.columns)
            # zeros_df=zeros_df.loc['LOCATION']

            # Concatenate the original dataframe with the zeros dataframe to reach the desired number of rows
            df_with_zeros = pd.concat([df1, zeros_df], axis=0)

            df1 = df_with_zeros

        dataset = dataset.append(df1,ignore_index=True)
        # dataset_collection[Nb_exp]=df1

df = pd.DataFrame(dataset)
os.makedirs('Desktop', exist_ok=True)
df.to_csv('MASSIVE-DATASET_5000runs.csv')
df.to_csv(index=True)

