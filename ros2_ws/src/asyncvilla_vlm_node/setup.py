from setuptools import setup

package_name = 'asyncvilla_vlm_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dion',
    maintainer_email='113128824+RAIN-CA@users.noreply.github.com',
    description='AsyncViLLA VLM node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_node = asyncvilla_vlm_node.vlm_node:main',
            'real_vlm_node = asyncvilla_vlm_node.real_vlm_node:main',
        ],
    },
)
