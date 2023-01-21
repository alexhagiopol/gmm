import numpy as np
import os
import random
import subprocess

def compile():
    current_path = os.path.dirname(__file__)
    build_folder_name = "precompiled_functions_build"
    build_path = os.path.join(current_path, build_folder_name)
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    os.chdir(build_path)
    subprocess.check_call(["cmake", "-DCMAKE_BUILD_TYPE=Release", ".."])
    subprocess.check_call(["make"])
    os.chdir(current_path)

if __name__ == "__main__":
    compile()
