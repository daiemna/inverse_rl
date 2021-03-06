from setuptools import setup, find_packages
import traceback

requirements = ["numpy", "scipy", "path.py", "python-dateutil", 
                "joblib", "mako", "ipywidgets", "numba", "flask", 
                "pygame", "h5py", "matplotlib", "opencv-python", "scikit-learn", "https://github.com/pytorch/pytorch/archive/v0.1.9.tar.gz",
                "torchvision", "mpi4py", "pandas"]

pip_requirements = ["Pillow", "atari-py", "pyprind", "ipdb", "boto3", "PyOpenGL", "nose2", 
                    "pyzmq", "tqdm", "msgpack-python", "git+https://github.com/inksci/mujoco-py-v0.5.7.git", 
                    "cached_property", "line_profiler", "cloudpickle", "Cython", "redis", "keras==1.2.1", 
                    "git+https://github.com/Theano/Theano.git@adfe319ce6b781083d8dc3200fb4481b00853791#egg=Theano", 
                    "git+https://github.com/neocxi/Lasagne.git@484866cf8b38d878e92d521be445968531646bb8#egg=Lasagne", 
                    "git+https://github.com/plotly/plotly.py.git@2594076e29584ede2d09f2aa40a8a195b3f3fc66#egg=plotly", 
                    "awscli", "git+https://github.com/openai/gym.git@v0.7.4#egg=gym", "pyglet", 
                    "git+https://github.com/neocxi/prettytensor.git", "jupyter", "progressbar2", "chainer==1.18.0", "https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-linux_x86_64.whl; 'linux' in sys_platform", "https://storage.googleapis.com/tensorflow/mac/gpu/tensorflow_gpu-1.0.1-py3-none-any.whl; sys_platform == 'darwin'",
                    "numpy-stl==2.2.0", "nibabel==2.1.0", "pylru==1.0.9", "hyperopt", "polling", 
                    "git+https://github.com/ansrivas/pylogging.git", "git+https://github.com/rll/rllab.git", 
                    "git+https://github.com/DEAP/deap@master", "ruamel.yaml"]

try:
    from pip._internal import main
    main(['install'] + requirements + pip_requirements)
except Exception:
    # Going to use easy_install for
    traceback.print_exc()

print("--------------------------------------------")

setup(
    name='irl-lab',
    version='0.1.0',
    include_package_data=True,
    setup_requires=requirements,
    packages=find_packages(),
)
