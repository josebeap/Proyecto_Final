from mpi4py import MPI
import mpi4py as mp

mp.Rc.initialize = True

def main():
    # Ejecutar el comando mpiexec
    mpiexec.run(["mpiexec", "-n", "4", "python", "-c", "from mpi4py import MPI; comm = MPI.COMM_WORLD; print('Hola desde el proceso', comm.rank)"])

if __name__ == "__main__":
    main()