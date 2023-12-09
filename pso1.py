import numpy as np
from tabulate import tabulate

def f(x):
    return (2*x**2 + x - 4)**2

class PSO():
    def __init__(self, c, w):
        self.x = np.random.uniform(-4, 4, 10)  # nilai awal x berupa 10 bilangan acak dengan batas -4 sampai 4
        self.v = np.zeros_like(self.x) # nilai awal v = 0
        self.c = c
        self.w = w

        self.oldX = self.x.copy()
        self.pBest = self.x.copy()
        self.gBest = self.x[np.argmin([f(x_i) for x_i in self.x])]

    def findPBest(self):
        for i in range(len(self.x)):
            if f(self.x[i]) < f(self.pBest[i]):
                self.pBest[i] = self.x[i]
            else:
                self.pBest[i] = self.oldX[i]

    def findGBest(self):
        fValues = [f(x_i) for x_i in self.x]
        self.gBest = self.x[np.argmin(fValues)]

    def updateV(self):
        for i in range(len(self.x)):
            R1, R2 = np.random.rand(), np.random.rand()  # nilai R1 dan R2  bilangan acak (0, 1)
            self.v[i] = (self.w * self.v[i]) + (self.c[0] * R1 * (self.pBest[i] - self.x[i])) + (self.c[1] * R2 * (self.gBest - self.x[i]))

    def updateX(self):
        for i in range(len(self.x)):
            self.x[i] += self.v[i]

    def iter(self, n):
        headers = ["Iteration", "x", "f(x)", "Pbest", "Gbest", "v"]
        table = []
        
        for i in range(n):
            f_values = [f(x_i) for x_i in self.x]
            row = [
                i+1,
                "\n".join(map("{0:.3f}".format, self.x)),
                "\n".join(map("{0:.3f}".format, f_values)),
                "\n".join(map("{0:.3f}".format, self.pBest)),
                "{0:.3f}".format(self.gBest),
                "\n".join(map("{0:.3f}".format, self.v)),
            ]

            self.updateV()
            self.updateX()
            self.findPBest()
            self.findGBest()
            self.oldX = self.x.copy()

            table.append(row)

        print(tabulate(table, headers=headers, tablefmt="simple_grid"))


if __name__ == "__main__":
    # Nilai c dan v sesuai dengan soal no 1 a)
    c = [0.5, 1]
    w = 1.0

    pso = PSO( c, w)
    pso.iter(3)
