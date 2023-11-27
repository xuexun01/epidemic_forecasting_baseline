import pandas as pd
import numpy as np


df = pd.read_csv("/home/xuexun/Desktop/short.csv")

state_table = {}
for row in df.itertuples():
    state_table[row.shortcut] = row.state

state_order = list(state_table.keys())


adj = np.zeros((len(state_order), len(state_order)))
with open("/home/xuexun/Desktop/neighbor.txt") as file:
    lines = file.readlines()
    for line in lines:
        line = line[:-1]
        states = line.split(" ")
        origin_state = states[0]
        dest_states = states[1:]
        if origin_state in state_order:
            adj[state_order.index(origin_state), state_order.index(origin_state)] = 1
            for dest in dest_states:
                if dest in state_order:
                    adj[state_order.index(origin_state), state_order.index(dest)] = 1
    file.close()

np.savetxt("./data/state_adj.txt", adj, fmt='%d', delimiter=',')