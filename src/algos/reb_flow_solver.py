import os
import subprocess
from collections import defaultdict
from src.misc.utils import mat2str
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpStatus, value
import pulp

def solveRebFlow(env,res_path,desiredAcc,CPLEXPATH):
    #CPLEXPATH='None'
    if CPLEXPATH=='None':
        return solveRebFlow_pulp(env, desiredAcc)
    else: 
        t = env.time
        accRLTuple = [(n,int(round(desiredAcc[n]))) for n in desiredAcc]
        accTuple = [(n,int(env.acc[n][t+1])) for n in env.acc]
        edgeAttr = [(i,j,env.G.edges[i,j]['time']) for i,j in env.G.edges]
    
        modPath = os.getcwd().replace('\\','/')+'/src/cplex_mod/'
        OPTPath = os.getcwd().replace('\\','/')+'/' + 'saved_files/cplex_logs/rebalancing/' + res_path + '/'
        if not os.path.exists(OPTPath):
            os.makedirs(OPTPath)
        datafile = OPTPath + f'data_{t}.dat'
        resfile = OPTPath + f'res_{t}.dat'
        with open(datafile,'w') as file:
            file.write('path="'+resfile+'";\r\n')
            file.write('edgeAttr='+mat2str(edgeAttr)+';\r\n')
            file.write('accInitTuple='+mat2str(accTuple)+';\r\n')
            file.write('accRLTuple='+mat2str(accRLTuple)+';\r\n')
        modfile = modPath+'minRebDistRebOnly.mod'
        my_env = os.environ.copy()
        my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        out_file =  OPTPath + f'out_{t}.dat'
        with open(out_file,'w') as output_f:
            subprocess.check_call([CPLEXPATH+"oplrun", modfile, datafile], stdout=output_f, env=my_env)
        output_f.close()

        # 3. collect results from file
        flow = defaultdict(float)
        with open(resfile,'r', encoding="utf8") as file:
            for row in file:
                item = row.strip().strip(';').split('=')
                if item[0] == 'flow':
                    values = item[1].strip(')]').strip('[(').split(')(')
                    for v in values:
                        if len(v) == 0:
                            continue
                        i,j,f = v.split(',')
                        flow[int(i),int(j)] = float(f)

        action = [flow[i,j] for i,j in env.edges]

        return action

def solveRebFlow_pulp(env, desiredAcc):

    t = env.time
    
    # Prepare the data: rounding desiredAcc and getting current vehicle counts
    accRLTuple = [(n, int(round(desiredAcc[n]))) for n in desiredAcc]
    accTuple = [(n, int(env.acc[n][t+1])) for n in env.acc]
    
    # Extract the edges and the times
    edgeAttr = [(i, j, env.G.edges[i, j]['time']) for i, j in env.G.edges]
    edges = [(i, j) for i, j in env.G.edges]

    # Map vehicle availability and desired vehicles for each region
    acc_init = {n: int(env.acc[n][t+1]) for n in env.acc}
    desired_vehicles = {n: int(round(desiredAcc[n])) for n in desiredAcc}

    region = [n for n in acc_init]
    # Time on each edge (used in the objective)
    time = {(i, j): env.G.edges[i, j]['time'] for i, j in edges}

    # Define the PuLP problem
    model = LpProblem("RebalancingFlowMinimization", LpMinimize)
    
    # Decision variables: rebalancing flow on each edge
    rebFlow = {(i, j): LpVariable(f"rebFlow_{i}_{j}", lowBound=0, cat='Integer') for (i, j) in edges}

    # Objective: minimize total time (cost) of rebalancing flows
    model += lpSum(rebFlow[(i, j)] * time[(i, j)] for (i, j) in edges), "TotalRebalanceCost"
    
    # Constraints for each region (node)
    for k in region:
        # 1. Flow conservation constraint (ensure net inflow/outflow achieves desired vehicle distribution)
        model += (
            lpSum(rebFlow[(j, i)]-rebFlow[(i, j)] for (i, j) in edges if j != i and i==k)
        ) >= desired_vehicles[k] - acc_init[k], f"FlowConservation_{k}"

        # 2. Rebalancing flows from region i should not exceed the available vehicles in region i
        model += (
            lpSum(rebFlow[(i, j)] for (i, j) in edges if i != j and i==k) <= acc_init[k], 
            f"RebalanceSupply_{k}"
        )
    
    # Solve the problem
    status = model.solve(pulp.PULP_CBC_CMD(msg=False, options=["primalTol=1e-9", "dualTol=1e-9", "mipGap=1e-9"]))
    #print objective value 
    # Check if the solution is optimal
    if LpStatus[status] == "Optimal":
        # Collect the rebalancing flows
        flow = defaultdict(float)
        for (i, j) in edges:
            flow[(i, j)] = rebFlow[(i, j)].varValue
        #print(len(rebFlow.keys()))
       
        #flow_result = {(i, j): value(rebFlow[(i, j)]) for (i, j) in edges}
        
        action = [flow[i,j] for i,j in env.edges]
        return action
    else:
        print(f"Optimization failed with status: {LpStatus[status]}")
        return None

