"""Box NMS setup script."""
from setuptools import setup, Extension
import pybind11


cpp_nms = Extension(
    "box_utils._c.box_nms",
    language='c++',
    sources=sorted([
        "src/c/nmsmodule.cpp"
    ]),
    include_dirs=sorted([
        "include/",
        pybind11.get_include()
    ]),
)


setup(
    ext_modules=[cpp_nms]
)
