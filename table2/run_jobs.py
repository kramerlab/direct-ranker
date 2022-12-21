from multiprocessing import Pool, cpu_count
import os


def run_script(f):
    os.system("source run_scripts/{}".format(f))

with Pool(cpu_count()) as p:
    p.map(run_script, [f for f in os.listdir("run_scripts")])
