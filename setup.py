from setuptools import setup, find_packages
import os
import subprocess


def restart_kernel():
    """
    Restarts the kernel if running in a Jupyter Notebook or IPython environment.
    """
    try:
        # Check if IPython is running
        from IPython import get_ipython,Application
        if get_ipython() is not None:
            app = Application.instance()
            app.kernel.do_shutdown(True) 
    except ImportError:
        pass  # IPython is not installed; skip restarting

# Helper function to install a sub-package
def install_subpackage(subpackage_path):
    """
    Install a sub-package from its setup.py file.
    """
    subprocess.check_call(["pip","install","-e",subpackage_path])

# Define the path to the scene_graph_benchmark package
scene_graph_path = os.path.join(
    os.path.dirname(__file__),"src","scene_graph_benchmark"
)

install_subpackage(scene_graph_path)

setup(
    name="bit_image_captioning",
    version="0.1.0",
    description="Arabic Image Captioning using Pre-training of Deep Bidirectional Transformers",
    author="Mahmood Anaam",
    author_email="eng.mahmood.anaam@gmail.com",
    url="https://github.com/Mahmood-Anaam/BiT-ImageCaptioning",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    install_requires=[
        "pytorch-transformers",
        "anytree",
        "yacs",
        "clint",
        "nltk",
        "joblib",
        

        ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

# Install the sub-package first


# restart_kernel()
