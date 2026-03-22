from matplotlib import pyplot as plt

V = {"L1": 0.0, "L2": 0.0}
update_minimum = 0.0001

k = 0
gamma = 0.9

while True:
    V_next = V.copy()
    V["L1"] = 0.5 * (-1 + gamma * V["L1"]) + 0.5 * (1 + gamma * V["L2"])
    V["L2"] = 0.5 * (0 + gamma * V["L1"]) + 0.5 * (-1 + gamma * V["L2"])

    k += 1
    differ = max(abs(V_next["L1"] - V["L1"]), abs(V_next["L2"] - V["L2"]))
    if differ < update_minimum:
        break


print("k:", k)
print("V:", V)