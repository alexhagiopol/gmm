import os
import random
import subprocess

if __name__ == "__main__":
    current_path = os.path.dirname(__file__)
    build_folder_name = "accelerated_functions_build"
    build_path = os.path.join(current_path, build_folder_name)
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    os.chdir(build_path)
    subprocess.check_call(["cmake", ".."])
    subprocess.check_call(["make"])
    os.chdir(current_path)
    import accelerated_functions_build.accelerated_functions as af
    input1 = random.randint(0, 1000)
    input2 = random.randint(0, 1000)
    print("Python code: sending inputs", input1, input2)
    result = af.add(input1, input2)
    print("Python code: received result", result)
