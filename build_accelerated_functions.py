import numpy as np
import os
import random
import subprocess

def compile():
    current_path = os.path.dirname(__file__)
    build_folder_name = "accelerated_functions_build"
    build_path = os.path.join(current_path, build_folder_name)
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    os.chdir(build_path)
    subprocess.check_call(["cmake", ".."])
    subprocess.check_call(["make"])
    os.chdir(current_path)

def test():
    import accelerated_functions_build.accelerated_functions as af
    input1 = random.randint(1, 1000)
    input2 = random.randint(1, 1000)
    result = af.add(input1, input2)
    assert(result == input1 + input2)
    input3 = np.ndarray(shape=(2,2), dtype=np.float64)
    input3.setflags(write=True)
    input4 = np.float64(random.randint(1, 1000))
    af.fill(input3, input4)
    assert(np.sum(input3) == 4 * input4)


if __name__ == "__main__":
    compile()
    test()
