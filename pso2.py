import random
from typing import List, TypedDict, Callable, Optional, NoReturn
from tabulate import tabulate


class PSOIteration(TypedDict):
    iteration: int
    pbest: Optional[List[List[float]]]
    gbest: Optional[List[float]]
    v: List[List[float]]
    x: List[List[float]]


class PSO:
    # Inisialisasi dengan constructor
    def __init__(self, x0: List[List[float]], v0: List[float], c1: float, c2: float, r1: float, r2: float, w: float, func: Callable[..., float] ) -> None:
        self.iteration = 0
        self.x = x0.copy()
        self.v = [v0.copy() for _ in range(len(x0))]
        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2
        self.w = w
        self.func = func

        self.pbest = None
        self.gbest = None

    def update_pbest(self) -> NoReturn:
        if self.pbest is None:
            self.pbest = self.x.copy()
            return

        for i, x_i in enumerate(self.x):
            func_x = self.func(*x_i)
            func_pbest = self.func(*self.pbest[i])
            if func_x < func_pbest:
                self.pbest[i] = x_i.copy()

    def update_gbest(self) -> NoReturn:
        for i, pbest_i in enumerate(self.pbest):
            if self.gbest is None:
                self.gbest = pbest_i.copy()

            func_pbest = self.func(*pbest_i)
            func_gbest = self.func(*self.gbest)
            if func_pbest < func_gbest:
                self.gbest = pbest_i.copy()

    def update_v(self) -> NoReturn:
        # Melakukan update pada setiap v dengan rumus v*w + c1*r1*(Pbest - P) + c2*r2*(Gbest - P)
        for i in range(len(self.v)):
            for j in range(len(self.v[i])):
                self.v[i][j] = self.v[i][j]*self.w + \
                    self.c1*self.r1*(self.pbest[i][j] - self.x[i][j]) + \
                    self.c2*self.r2*(self.gbest[j] - self.x[i][j])

    def update_x(self) -> NoReturn:
        # Melakukan update pada setiap (x, y) dengan (x+vx, y+vy)
        for i in range(len(self.x)):
            for j in range(len(self.x[i])):
                self.x[i][j] += self.v[i][j]

    # Method untuk melanjutkan iterasi
    def next_iter(self) -> NoReturn:
        self.iteration += 1
        self.update_pbest()
        self.update_gbest()
        self.update_v()
        self.update_x()

    # Method untuk menjalankan PSO berdasarkan jumlah iterasi yang ditentukan
    # Method ini menghasilkan list histori hasil nilai setiap iterasi
    def run(self, iter_count: int) -> List[PSOIteration]:
        iterations = []
        it0 = dict()
        it0["iteration"] = self.iteration
        it0["pbest"] = self.pbest
        it0["gbest"] = self.gbest
        it0["v"] = [v_i.copy() for v_i in self.v]
        it0["x"] = [x_i.copy() for x_i in self.x]
        iterations.append(it0)
        for _ in range(iter_count):
            self.next_iter()
            new_it = dict()
            new_it["iteration"] = self.iteration
            new_it["pbest"] = [pbest_i.copy() for pbest_i in self.pbest]
            new_it["gbest"] = self.gbest.copy()
            new_it["v"] = [v_i.copy() for v_i in self.v]
            new_it["x"] = [x_i.copy() for x_i in self.x]
            iterations.append(new_it)
        return iterations


# Fungsi tujuan
def f(x, y):
    return (1.25 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (1.5 - x + x*y**3)**2


# Main entry
if __name__ == "__main__":
    # 10 pasang acak (x, y) dalam range [-4.5, 4.5]
    x0 = [[random.uniform(-4.5, 4.5), random.uniform(-4.5, 4.5)]
          for _ in range(10)]
    # Instansiasi PSO dengan r1 dan r2 acak dalam range [1, 0]
    new_pso = PSO(x0=x0, v0=[0, 0], c1=1, c2=0.5, r1=random.uniform(
        0, 1), r2=random.uniform(0, 1), w=1, func=f)

    # Jumlah iterasi yang akan dilakukan
    n = 10
    
    # Hasil iterasi
    iterations = new_pso.run(n)

    # List untuk formatting hasil iterasi agar dapat diprint menjadi table
    iterTable = []
    for i, it in enumerate(iterations):
        iterItem = [str(it["iteration"])]
        iterF = "-"
        iterPbest = "-"
        iterGbest = "-"
        if it["iteration"] > 0:
            iterF = ",\n".join([f"{f_i:.3f}" for f_i in [f(*x_i)
                               for x_i in iterations[i-1]["x"]]])
            iterPbest = ",\n".join(
                str([f"{pb_ij:.3f}" for pb_ij in pb_i]) for pb_i in it["pbest"])
            iterGbest = ", ".join(
                f"{gb_i:.3f}" for gb_i in it["gbest"])
        iterV = ",\n".join([str([f"{v_ij:.3f}" for v_ij in v_i])
                           for v_i in it["v"]])
        iterX = ",\n".join([str([f"{x_ij:.3f}" for x_ij in x_i])
                           for x_i in it["x"]])
        iterItem.append(iterF)
        iterItem.append(iterPbest)
        iterItem.append(iterGbest)
        iterItem.append(iterV)
        iterItem.append(iterX)
        iterTable.append(iterItem)

    # Print table hasil iterasi
    print(tabulate(iterTable, headers=[
          "Iteration", "f(x, y)", "Pbest", "Gbest", "v", "New (x, y)"], tablefmt="simple_grid"))
