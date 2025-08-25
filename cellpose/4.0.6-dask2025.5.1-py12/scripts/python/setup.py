import setuptools

setuptools.setup(
    name="cellpose_tools",
    version="4.0.6",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "cellpose==4.0.6",
        "dask==2025.5.1",
        "distributed==2025.5.1",
        "numpy>=1.26.0,<2",
        "zarr>=2.18,<3"
    ]
)
