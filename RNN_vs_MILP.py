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

big_number = 100000
Nb_exp = 100
dataset_collection = {}
dataset = pd.DataFrame()

fairness_OPTIMAL = []

COST_OPTIMAL = []
THROUGHPUT_OPTIMAL = []
ADMITTANCE_OPTIMAL = []

RATIO_eMBB_optimal = []

RATIO_uRLLC_optimal = []

RATIO_mMTC_optimal = []

TIME_MILP=[]
time_MILP=[]

for tt in range(Nb_exp):

    loc_x_reg = []
    loc_y_reg = []
    loc_x_edge = []
    loc_y_edge = []

    ratio_admitted_optimal = []

    ratio_admitted_eMBB_optimal = []
    ratio_admitted_uRLLC_optimal = []
    ratio_admitted_mMTC_optimal = []

    cost_optimal = []
    total_throughput_optimal = []
    fairness_optimal = []

    time_MILP=[]




    n_servers = 4
    n_servers_edge = 3

    N = np.arange(20, 120, 20)

    Number_slices = [n for n in N]
    # Number_slices=[60]
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

        rnd = np.random
        loc_x = rnd.rand(n_slices) * 1
        loc_y = rnd.rand(n_slices) * 1
        loc_x_BS = [0.250, 0.250, 0.750, 0.750]
        loc_y_BS = [0.250, 0.750, 0.250, 0.750]

        RB_total = 100 * Nb_BS  # 20MHZ bandwith= 100 RBs each BS
        N_RB_rem_embb = [50, 50, 50, 50]
        N_RB_rem_urllc = [25, 25, 25, 25]
        N_RB_rem_mmtc = [25, 25, 25, 25]

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

        t = time.time()
        solution = m.solve()
        time_MILP.append((time.time() - t)*100)
        # print(time_MILP)


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

        CC = sum(
            Cost_matrix[s] * R_CU[i] * solution_theta_CU[i, s] for i in range(n_slices) for s in range(n_servers))
        # we take into consideration the cost of deployment of just CU because the DU is always at the edge.
        # CC = sum(Cost_matrix[s] * R_CU[i] * solution_theta_CU[i, s] for i in range(n_slices) for s in
        #          range(n_servers)) + sum(
        #     Cost_matrix[s] * R_DU[i] * solution_theta_DU[i, s] for i in range(n_slices) for s in range(n_servers))


        ratio_admitted_optimal.append(
            sum(solution_theta_CU[i, s] for i in range(n_slices) for s in range(n_servers)) / n_slices * 100)
        # print(ratio_admitted_optimal)
        ratio_admitted_eMBB_optimal.append(
            sum(solution_theta_CU[i, s] for i in range(n_eMBB) for s in range(n_servers)) / n_eMBB * 100)
        ratio_admitted_uRLLC_optimal.append(sum(
            solution_theta_CU[i, s] for i in range(n_eMBB, n_eMBB + n_uRLLC) for s in
            range(n_servers)) / n_uRLLC * 100)
        ratio_admitted_mMTC_optimal.append(sum(
            solution_theta_CU[i, s] for i in range(n_eMBB + n_uRLLC, n_slices) for s in
            range(n_servers)) / n_MTC * 100)

        total_throughput_optimal.append(throughput)
        cost_optimal.append(CC)
        fairness_optimal.append(fairness)

        #the dataframe row are users and columns are the features

        GOPS_av =[]
        LOC=[]
        LOC_new=[]
        Capacity_av=[]
        delta_av=[]
        # GOPS_av = np.tile(R, (n_slices, 1))
        CCC = Capacity.flatten()  # from 2d array to 1d
        ddd= delta.flatten()
        loc_x_EDGE=[]
        loc_y_EDGE=[]
        loc_x_REG= []
        loc_y_REG= []
        for i in range(n_slices):
            GOPS_av.append(R)
            Capacity_av.append(CCC)
            delta_av.append(ddd)
            loc_x_EDGE.append(loc_x_edge)
            loc_y_EDGE.append(loc_y_edge)
            loc_x_REG.append(loc_x_reg)
            loc_y_REG.append(loc_y_reg)
            ind_CU=ind_DU=0
            #print(solution_theta_CU)
            temp = np.zeros(12) #12 elements vector for the location
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
                            'max_latency': delta_max,
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


    RATIO_eMBB_optimal.append(ratio_admitted_eMBB_optimal)
    RATIO_uRLLC_optimal.append(ratio_admitted_uRLLC_optimal)
    RATIO_mMTC_optimal.append(ratio_admitted_mMTC_optimal)
    COST_OPTIMAL.append(cost_optimal)
    fairness_OPTIMAL.append(fairness_optimal)
    THROUGHPUT_OPTIMAL.append(total_throughput_optimal)
    ADMITTANCE_OPTIMAL.append(ratio_admitted_optimal)
    TIME_MILP.append(time_MILP)


df = pd.DataFrame(dataset)
# df_one_hot = pd.get_dummies(df, columns=['RB'])
os.makedirs('Desktop', exist_ok=True)
df.to_csv('test_data_milpVSrnn.csv')
df.to_csv(index=True)


####RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
from torch.utils.data import Dataset
import pandas as pd
import torch
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from tqdm import tqdm  # For a nice progress bar!
import tensorboardX


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

input_size = 19
sequence_length = 100
hidden_size = 32
num_layers = 1
output_size= 13
num_classes = 13 ### is it the number of possible labels(we have 13 possibility of locations)????
learning_rate = 0.005
batch_size = 1 #i choose it to be a divisor of numer of samples otehr wise
num_epochs = 1 #100#50


df = pd.read_csv("test_data_milpVSrnn.csv")
####Do preprocessing on the dataset as done before and mapp it dataset to be loaded to the RNN model
class CustomDataset(Dataset):
    def __init__(self):
        # load data and shuffle, befor splitting  #########################SHUFFLE DATA!!!!!!!!!
        self.df = pd.read_csv("test_data_milpVSrnn.csv")
        self.df_labels = df[['LOCATION']]
        self.df=df.drop(columns=['LOCATION'] )#drop label
        self.df.drop(self.df.columns[0], axis=1, inplace=True) ##drop the indexes of the dataframe
        LIST_data=[]
        LIST_target =[]
        for BATCH in range(int(df.shape[0]/sequence_length)//int(batch_size)):
            nested_list = []
            for index in range(batch_size):
                list=[]
                for j in range(sequence_length):
                    list.append([self.df.iloc[j+sequence_length*index + batch_size*sequence_length*BATCH]])
                nested_list.append(list) #collect each batch of data into nested_list
            nested_list=torch.tensor(nested_list).squeeze(dim=2).float()  ##dim= 2 will remove the singleton entry of the tensor added at the 3rd location and will preserve the batch size if it was =1
            # mean = nested_list.mean()
            # std = nested_list.std()
            # normalized = (nested_list - mean) / std
            LIST_data.append(nested_list) ##append the whole batch into LIST
            nested_target = []
            for index in range(batch_size):
                list = []
                for j in range(sequence_length):
                    list.append([self.df_labels.iloc[j + sequence_length * index]])
                nested_target.append(list)
            nested_target = torch.tensor(nested_target).squeeze(dim=2).long()
            nested_target = nested_target.squeeze(dim=2)
            LIST_target.append(nested_target)

        self.test = LIST_data
        self.test_labels =LIST_target

    def __len__(self):
        return len(self.test)
    def __getitem__(self, idx):
        return self.test[idx], self.test_labels[idx]
        return self

TEST_DATA = CustomDataset()

class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.embedding = nn.Embedding(input_size, embedding_dim, num_layers)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # fully connected
        # self.fc1 = nn.Linear(hidden_size * 2, 64)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(32, 16)
        # self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        # x = self.embedding(x)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        # print("----------",h0.shape,"////",c0.shape,";;;;;;",x.shape)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out = self.dropout(out)
        out = self.fc(out)
        # out = self.fc1(out)
        # # out = self.relu(out)
        # out = self.fc2(out)
        # # out = self.relu(out)
        # out = self.fc3(out)
        # out = self.relu(out)
        # out = self.fc4(out)
        # out = self.sigmoid(out)

        return out

state_dict = torch.load('model.pt')

rnn_model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)
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

TIME_RNN=[]

ratio_CU_edge_eMbb_RNN_all_runs = []
ratio_CU_edge_uRLLC_RNN_all_runs = []
ratio_CU_edge_mMTC_RNN_all_runs = []

ratio_CU_reg_eMbb_RNN_all_runs = []
ratio_CU_reg_uRLLC_RNN_all_runs = []
ratio_CU_reg_mMTC_RNN_all_runs = []

time_RNN=[]

ratio_admitted_RNN = []

ratio_admitted_eMBB_RNN = []
ratio_admitted_uRLLC_RNN = []
ratio_admitted_mMTC_RNN = []

cost_RNN = []
total_throughput_RNN = []
fairness_RNN = []
Code_rate = [438, 466, 517, 567, 616, 666, 719, 772, 822, 873, 910, 948]
spectral_eff = [2.5664, 2.7305, 3.0293, 3.3223, 3.6094, 3.9023, 4.2129, 4.5234, 4.8164, 5.1152, 5.3320,5.5547]
n_servers = 4
n_servers_edge = 3
counter = 0 ##to specify the number of users

N = np.arange(20, 120, 20)

Number_slices = [n for n in N]
    #
with torch.no_grad():
    for batch_idx, data in enumerate(tqdm(TEST_DATA.test)):  ##for i, batch in enumerate(dataloader):
        ##data in this case is one sample in the following order{20,40,60,80,100}

        counter+=20
        NN=counter
        if (NN/10)%2==0:
            n_eMBB = int(NN * 0.25)
        else :
            n_eMBB = int(NN * 0.25) +1

        n_uRLLC = int(NN * 0.25)
        n_MTC = int(NN * 0.5)
        n_slices = n_eMBB + n_MTC + n_uRLLC

        targets = TEST_DATA.test_labels[batch_idx]

        features = data.squeeze(dim=0)
        features = features.numpy()
        features=features[:NN]
        C=[]
        S_E=[]
        # print(n_slices)
        for i in range(n_slices):

            # print(features[i])
            MCS = int(features[i,11])##get MCS from dataset ,I THINK @ INDEX 10 OF THE INPUT TENSOR,
            # MCS = 20
            # print(MCS)
            C.append(Code_rate[MCS - 17] / 1024)
            S_E.append(spectral_eff[MCS - 17])
        M = 6  # modulation bits log2(64)
        L = 2  # number of MIMO layers
        A = 4  # number of Antennas

        t=time.time()
        scores = rnn_model(data)
        time_RNN.append((time.time()-t)*100)
        # print(time_RNN)
        _, max_index = torch.max(scores, dim=-1,keepdim=False)  ## returns the max_value,max_index tuple element wise over the all sequences of the batch [32,100]
        predictions = max_index
        predictions=predictions.squeeze(0) ##remove extra dimension from output

        ##get theta_cu and du from predictions along with n_slices.......
        predictions = predictions.numpy() ##convert output to numpy

        predictions = predictions[:NN] #remove padding from output

        solution_theta_DU = np.zeros((n_slices, n_servers))
        solution_theta_CU = np.zeros((n_slices, n_servers))
        for i in range(n_slices):
            if predictions[i] != 0:
                if (predictions[i] % 3) != 0:
                    ind_DU = int((predictions[i] % 3) - 1)
                else:
                    ind_DU = 2

                # print(']]]]]]', ind_DU)
                ind_CU= int((predictions[i]-1-ind_DU)/3)
                # print('+++++++++++', ind_CU)

                solution_theta_CU[i,ind_CU]=1 #append vector of length n_server with 1 on the choosen CU
                solution_theta_DU[i,ind_DU]=1
            ##else Prediction reamin zero=> user not allocated

        ####MUST check if the predictions fits the delay constraints and gops capacity.. As done in the random scenario
        R= features[0,12:16] #GOPS availbale at each server : vector of size n_servers
        rem_server_cap = R  # remaining server capacity
        R_DU=features[:,17]
        R_CU=features[:,16]
        for s in range(n_servers):
            for i in range(n_slices):
                rem_server_cap[s] -= solution_theta_DU[i, s] * R_DU[i] + solution_theta_CU[i, s] * R_CU[i]
                if rem_server_cap[s] <= 0:
                    # print('no enough RBs')
                    # if server capacity is not satisfied
                    solution_theta_DU[i, s] = solution_theta_CU[i, s] = 0
        delta_max= features[:,18]
        loc_x_edge= features[0,2:5]
        loc_y_edge=features[0,5:8]
        loc_x_reg= np.array([features[0,8]])
        loc_y_reg= np.array([features[0,9]])
        for s in range(n_servers):
            for s1 in range(n_servers):
                if s < n_servers_edge:
                    if s1 < n_servers_edge:
                        delta[s][s1] = 5 * math.sqrt((loc_x_edge[s] - loc_x_edge[s1]) ** 2 + (
                                loc_y_edge[s] - loc_y_edge[s1]) ** 2)  # random.randint(50,100)  # us edge-edge (5us/km => 10-20 km betwwen edge-edge server)
                    else:
                        delta[s][s1] = 5 * math.sqrt((loc_x_edge[s] - loc_x_reg[s1 - n_servers_edge]) ** 2 + (
                                loc_y_edge[s] - loc_y_reg[s1 - n_servers_edge]) ** 2)  # random.randint(200,400) #edge-regional(5us/km => 40-80 km betwwen edge-edge server)
                else:
                    if s1 < n_servers_edge:
                        delta[s][s1] = 5 * math.sqrt((loc_x_edge[s1] - loc_x_reg[s - n_servers_edge]) ** 2 + (
                                loc_y_edge[s1] - loc_y_reg[ s - n_servers_edge]) ** 2)  # random.randint(200,400) #reg-edge
                if s == s1:
                    delta[s][s1] = 0

        for i in range(n_slices):
            for s in range(n_servers):
                for s1 in range(n_servers):
                    if delta[s, s1] * (solution_theta_CU[i, s] * solution_theta_DU[i, s1]) > delta_max[i]:
                        # print("delta violated")
                        solution_theta_DU[i, s] = solution_theta_CU[i, s] = 0

        # print('CU',solution_theta_CU)
        # print('DU',solution_theta_DU)

        ##throughput calculation

        RB_slices= features[:,10]
        throughput = 0
        for i in range(n_slices):
            for s in range(n_servers):
                # throughput += N_SC*N_sym*RB_slices[i]*S_E[i]*L*1600*solution_theta_CU[i,s]/1000000##in Mbps
                throughput += L * M * C[i] * N_SC * N_sym * RB_slices[i] * solution_theta_CU[i, s] * (
                        1 - 0.14) / 1000  ##in Mbps
        print(NN)
        n_admitted_eMBB = sum(solution_theta_CU[i, s] for i in range(n_eMBB) for s in range(n_servers))
        n_admitted_uRLLC = sum(
            solution_theta_CU[i, s] for i in range(n_eMBB, n_eMBB + n_uRLLC) for s in range(n_servers))
        n_admitted_mMTC = sum(
            solution_theta_CU[i, s] for i in range(n_eMBB + n_uRLLC, n_slices) for s in range(n_servers))

        ######fairness index of admitted slices
        s = n_admitted_eMBB / n_eMBB + n_admitted_mMTC / n_MTC + n_admitted_uRLLC / n_uRLLC
        x = (n_admitted_eMBB / n_eMBB) ** 2 + (n_admitted_mMTC / n_MTC) ** 2 + (n_admitted_uRLLC / n_uRLLC) ** 2

        fairness = (s * s) / (3 * x)

        # objectivee=np.append(objectivee,m.objective_value)
        Cost_matrix = np.ones(n_servers)  # link capacity available between 2 servers

        for s in range(n_servers):
            if s < n_servers_edge:
                Cost_matrix[s] = 1.59  # edge
            else:
                Cost_matrix[s] = 0.5  # regional

        CC = sum(Cost_matrix[s] * R_CU[i] * solution_theta_CU[i, s] for i in range(n_slices) for s in range(n_servers))
        # print('n_slices', n_slices)
        # print(batch_idx)

        ratio_admitted_RNN.append(
            sum(solution_theta_CU[i, s] for i in range(n_slices) for s in range(n_servers)) / n_slices * 100)
        # print(ratio_admitted_RNN)


        ratio_admitted_eMBB_RNN.append(
            sum(solution_theta_CU[i, s] for i in range(n_eMBB) for s in range(n_servers)) / n_eMBB * 100)
        ratio_admitted_uRLLC_RNN.append(sum(
            solution_theta_CU[i, s] for i in range(n_eMBB, n_eMBB + n_uRLLC) for s in
            range(n_servers)) / n_uRLLC * 100)
        ratio_admitted_mMTC_RNN.append(sum(
            solution_theta_CU[i, s] for i in range(n_eMBB + n_uRLLC, n_slices) for s in
            range(n_servers)) / n_MTC * 100)

        total_throughput_RNN.append(throughput)
        cost_RNN.append(CC)
        fairness_RNN.append(fairness)

        # print(batch_idx)
        if (batch_idx+1) % len(Number_slices) == 0:  ######every 5 instances means after each experiment append to all_runs
            # print('collect all runss')
            #append to all_runs
            RATIO_eMBB_RNN.append(ratio_admitted_eMBB_RNN)
            RATIO_uRLLC_RNN.append(ratio_admitted_uRLLC_RNN)
            RATIO_mMTC_RNN.append(ratio_admitted_mMTC_RNN)
            COST_RNN.append(cost_RNN)
            print(cost_RNN)
            print(ratio_admitted_RNN)
            FAIRNESS_RNN.append(fairness_RNN)
            THROUGHPUT_RNN.append(total_throughput_RNN)
            ADMITTANCE_RNN.append(ratio_admitted_RNN)

            TIME_RNN.append(time_RNN)

            #re-initialize for next Exp
            ratio_admitted_RNN = []
            ratio_admitted_eMBB_RNN = []
            ratio_admitted_uRLLC_RNN = []
            ratio_admitted_mMTC_RNN = []
            cost_RNN = []
            total_throughput_RNN = []
            fairness_RNN = []
            counter=0

            time_RNN=[]

##############################################PLOTTING

I_ratio_optimal = []
I_cost_optimal = []
I_fairness_optimal = []
I_throughput_optimal = []
I_ratio_embb_optimal = []
I_ratio_uRLLC_optimal = []
I_ratio_mMTC_optimal = []

I_time_MILP=[]
I_time_RNN=[]

I_ratio_RNN = []
I_cost_RNN = []
I_fairness_RNN = []
I_throughput_RNN = []
I_ratio_embb_RNN = []
I_ratio_uRLLC_RNN = []
I_ratio_mMTC_RNN = []

I_time_reduction=[]
for N in range(len(Number_slices)):
    ratio_values_embb_optimal = []
    ratio_values_uRLLC_optimal = []
    ratio_values_mMTC_optimal = []
    Admittance_values_optimal = []
    throughput_values_optimal = []
    cost_values_optimal = []
    fairness_values_optimal = []

    time_values_MILP=[]
    time_values_RNN=[]

    ratio_values_embb_RNN = []
    ratio_values_uRLLC_RNN = []
    ratio_values_mMTC_RNN = []
    Admittance_values_RNN = []
    throughput_values_RNN = []
    cost_values_RNN = []
    fairness_values_RNN = []

    time_values_reduction=[]

    for i in range(Nb_exp):

        Admittance_values_optimal.append(ADMITTANCE_OPTIMAL[i][N])
        throughput_values_optimal.append(THROUGHPUT_OPTIMAL[i][N])
        cost_values_optimal.append(COST_OPTIMAL[i][N])
        fairness_values_optimal.append(fairness_OPTIMAL[i][N])
        ratio_values_embb_optimal.append(RATIO_eMBB_optimal[i][N])
        ratio_values_uRLLC_optimal.append(RATIO_uRLLC_optimal[i][N])
        ratio_values_mMTC_optimal.append(RATIO_mMTC_optimal[i][N])

        time_values_MILP.append(TIME_MILP[i][N])
        time_values_RNN.append(TIME_RNN[i][N])

        reduction = np.subtract(TIME_MILP,TIME_RNN)/TIME_MILP*100
        time_values_reduction.append(reduction[i][N])
        # print(np.subtract(TIME_RNN,TIME_MILP)/TIME_MILP*100)

        Admittance_values_RNN.append(ADMITTANCE_RNN[i][N])
        throughput_values_RNN.append(THROUGHPUT_RNN[i][N])
        cost_values_RNN.append(COST_RNN[i][N])
        fairness_values_RNN.append(FAIRNESS_RNN[i][N])
        ratio_values_embb_RNN.append(RATIO_eMBB_RNN[i][N])
        ratio_values_uRLLC_RNN.append(RATIO_uRLLC_RNN[i][N])
        ratio_values_mMTC_RNN.append(RATIO_mMTC_RNN[i][N])

    I_ratio_optimal.append(
        st.t.interval(confidence=0.95, df=len(Admittance_values_optimal) - 1, loc=np.mean(Admittance_values_optimal),
                      scale=st.sem(Admittance_values_optimal)))
    I_throughput_optimal.append(
        st.t.interval(confidence=0.95, df=len(throughput_values_optimal) - 1, loc=np.mean(throughput_values_optimal),
                      scale=st.sem(throughput_values_optimal)))
    I_cost_optimal.append(
        st.t.interval(confidence=0.95, df=len(cost_values_optimal) - 1, loc=np.mean(cost_values_optimal),
                      scale=st.sem(cost_values_optimal)))
    I_fairness_optimal.append(
        st.t.interval(confidence=0.95, df=len(fairness_values_optimal) - 1, loc=np.mean(fairness_values_optimal),
                      scale=st.sem(fairness_values_optimal)))

    I_time_MILP.append(
        st.t.interval(confidence=0.95, df=len(time_values_MILP) - 1, loc=np.mean(time_values_MILP),
                      scale=st.sem(time_values_MILP)))
    I_time_RNN.append(
        st.t.interval(confidence=0.95, df=len(time_values_RNN) - 1, loc=np.mean(time_values_RNN),
                      scale=st.sem(time_values_RNN)))
    I_time_reduction.append(
        st.t.interval(confidence=0.95, df=len(time_values_reduction) - 1, loc=np.mean(time_values_reduction),
                      scale=st.sem(time_values_reduction)))



    for i in range(len(I_ratio_optimal)):
        if math.isnan(I_ratio_optimal[i][0]):
            I_ratio_optimal[i] = Admittance_values_optimal[0], Admittance_values_optimal[0]

    I_ratio_RNN.append(
        st.t.interval(confidence=0.95, df=len(Admittance_values_RNN) - 1, loc=np.mean(Admittance_values_RNN),
                      scale=st.sem(Admittance_values_RNN)))
    I_throughput_RNN.append(
        st.t.interval(confidence=0.95, df=len(throughput_values_RNN) - 1, loc=np.mean(throughput_values_RNN),
                      scale=st.sem(throughput_values_RNN)))
    I_cost_RNN.append(
        st.t.interval(confidence=0.95, df=len(cost_values_RNN) - 1, loc=np.mean(cost_values_RNN),
                      scale=st.sem(cost_values_RNN)))
    I_fairness_RNN.append(
        st.t.interval(confidence=0.95, df=len(fairness_values_RNN) - 1, loc=np.mean(fairness_values_RNN),
                      scale=st.sem(fairness_values_RNN)))


    for i in range(len(I_ratio_RNN)):
        if math.isnan(I_ratio_RNN[i][0]):
            I_ratio_RNN[i] = Admittance_values_RNN[0], Admittance_values_RNN[0]

lower_ratio_RNN = []
upper_ratio_RNN = []

lower_ratio_optimal = []
upper_ratio_optimal = []

for i in range(len(Number_slices)):

    lower_ratio_optimal.append(I_ratio_optimal[i][0])
    upper_ratio_optimal.append(I_ratio_optimal[i][1] if I_ratio_optimal[i][1] < 100 else 100)

    lower_ratio_RNN.append(I_ratio_RNN[i][0])
    upper_ratio_RNN.append(I_ratio_RNN[i][1] if I_ratio_RNN[i][1] < 100 else 100)

r = []
r1 = []

for lower, upper, y in zip(lower_ratio_optimal, upper_ratio_optimal, range(len(lower_ratio_optimal))):
    plt.plot((y, y), (lower, upper), 'o-', color='green', label='Optimal')
    r.append(lower + (upper - lower) / 2)

for lower, upper, y in zip(lower_ratio_RNN, upper_ratio_RNN, range(len(lower_ratio_RNN))):
    plt.plot((y, y), (lower, upper), '^--', color='purple', label='RNN')
    r1.append(lower + (upper - lower) / 2)

plt.plot(r, '-', color='g')
plt.plot(r1, '--', color='purple')

plt.xticks(range(len(lower_ratio_optimal)), Number_slices, fontsize=11)
plt.ylim([0, 120])
plt.yticks(fontsize=11)
plt.xlabel('Nb_Users')
plt.ylabel('Admittance Ratio (%)', fontsize=13)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=10)
plt.grid()
plt.show()

lower_cost_optimal = []
upper_cost_optimal = []

lower_cost_RNN = []
upper_cost_RNN = []

for i in range(len(Number_slices)):

    lower_cost_optimal.append(I_cost_optimal[i][0])
    upper_cost_optimal.append(I_cost_optimal[i][1])

    lower_cost_RNN.append(I_cost_RNN[i][0])
    upper_cost_RNN.append(I_cost_RNN[i][1])
r = []
r1 = []


for lower, upper, y in zip(lower_cost_optimal, upper_cost_optimal, range(len(lower_cost_optimal))):
    plt.plot((y, y), (lower, upper), 'o-', color='green', label='Optimal')
    r.append(lower + (upper - lower) / 2)

for lower, upper, y in zip(lower_cost_RNN, upper_cost_RNN, range(len(lower_cost_RNN))):
    plt.plot((y, y), (lower, upper), '^--', color='purple', label='RNN')
    r1.append(lower + (upper - lower) / 2)

plt.plot(r, '-', color='g')
plt.plot(r1, '--', color='purple')

plt.xticks(range(len(lower_cost_optimal)), Number_slices, fontsize=11)
plt.ylim([0, 100])
plt.yticks(fontsize=11)
plt.xlabel('Nb_Users')
plt.ylabel('Cost ($)', fontsize=13)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=10)
plt.grid()
plt.show()

lower_fairness_optimal = []
upper_fairness_optimal = []

lower_fairness_RNN = []
upper_fairness_RNN = []

for i in range(len(Number_slices)):

    lower_fairness_optimal.append(I_fairness_optimal[i][0])
    upper_fairness_optimal.append(I_fairness_optimal[i][1])

    lower_fairness_RNN.append(I_fairness_RNN[i][0])
    upper_fairness_RNN.append(I_fairness_RNN[i][1])

r = []
r1 = []

for lower, upper, y in zip(lower_fairness_optimal, upper_fairness_optimal, range(len(lower_fairness_optimal))):
    plt.plot((y, y), (lower, upper), 'o-', color='green', label='Optimal')
    r.append(lower + (upper - lower) / 2)

for lower, upper, y in zip(lower_fairness_RNN, upper_fairness_RNN, range(len(lower_fairness_RNN))):
    plt.plot((y, y), (lower, upper), '^--', color='purple', label='RNN')
    r1.append(lower + (upper - lower) / 2)


plt.plot(r, '-', color='g')
plt.plot(r1, '--', color='purple')

plt.xticks(range(len(lower_fairness_optimal)), Number_slices, fontsize=11)
plt.ylim([0, 1])
plt.yticks(fontsize=11)
plt.xlabel('Nb_Users')
plt.ylabel('fairness', fontsize=13)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=10)
plt.grid()
plt.show()

lower_throughput_optimal = []
upper_throughput_optimal = []

lower_throughput_RNN = []
upper_throughput_RNN = []


for i in range(len(Number_slices)):

    lower_throughput_optimal.append(I_throughput_optimal[i][0])
    upper_throughput_optimal.append(I_throughput_optimal[i][1])

    lower_throughput_RNN.append(I_throughput_RNN[i][0])
    upper_throughput_RNN.append(I_throughput_RNN[i][1])


r = []
r1 = []

for lower, upper, y in zip(lower_throughput_optimal, upper_throughput_optimal, range(len(lower_throughput_optimal))):
    plt.plot((y, y), (lower, upper), 'o-', color='green', label='Optimal')
    r.append(lower + (upper - lower) / 2)

for lower, upper, y in zip(lower_throughput_RNN, upper_throughput_RNN, range(len(lower_throughput_RNN))):
    plt.plot((y, y), (lower, upper), '^--', color='purple', label='RNN')
    r1.append(lower + (upper - lower) / 2)

plt.plot(r, '-', color='g')
plt.plot(r1, '--', color='purple')

plt.xticks(range(len(lower_throughput_optimal)), Number_slices, fontsize=11)
plt.ylim([0, 500])
plt.yticks(fontsize=11)
plt.xlabel('Nb_Users')
plt.ylabel('Throughput (Mbps)', fontsize=13)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=10)
plt.grid()
plt.show()


lower_time_RNN = []
upper_time_RNN = []

lower_time_MILP = []
upper_time_MILP = []

for i in range(len(Number_slices)):

    lower_time_MILP.append(I_time_MILP[i][0])
    upper_time_MILP.append(I_time_MILP[i][1])

    lower_time_RNN.append(I_time_RNN[i][0])
    upper_time_RNN.append(I_time_RNN[i][1])

r = []
r1 = []

for lower, upper, y in zip(lower_time_MILP, upper_time_MILP, range(len(lower_time_MILP))):
    plt.plot((y, y), (lower, upper), 'o-', color='green',  label='MILP')
    r.append(lower + (upper - lower) / 2)

for lower, upper, y in zip(lower_time_RNN, upper_time_RNN, range(len(lower_time_RNN))):
    plt.plot((y, y), (lower, upper), '^--', color='purple',  label='RNN')
    r1.append(lower + (upper - lower) / 2)

plt.plot(r, '-', color='g')
plt.plot(r1, '--', color='purple')

plt.xticks(range(len(lower_time_MILP)), Number_slices, fontsize=11)
plt.ylim([0, 5])
plt.yticks(fontsize=11)
plt.xlabel('Nb_Users')
plt.ylabel('Execution time (ms)', fontsize=13)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=10)
plt.grid()
plt.show()


lower_time_reduction = []
upper_time_reduction = []

for i in range(len(Number_slices)):

    lower_time_reduction.append(I_time_reduction[i][0])
    upper_time_reduction.append(I_time_reduction[i][1])

r = []

for lower, upper, y in zip(lower_time_reduction, upper_time_reduction, range(len(lower_time_reduction))):
    plt.plot((y, y), (lower, upper), 'o-', color='green',  label='RNN-MILP')
    r.append(lower + (upper - lower) / 2)

plt.plot(r, '-', color='g')

plt.xticks(range(len(lower_time_reduction)), Number_slices, fontsize=11)
plt.ylim([0, 100])
plt.yticks(fontsize=11)
plt.xlabel('Nb_Users')
plt.ylabel('Reduction in Execution time (in %)', fontsize=13)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), fontsize=10)
plt.grid()
plt.show()