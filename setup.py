from setuptools import setup

setup(
    name="Orange3 Cyber",
    packages=["orangecyber"],
    package_data={"orangecyber": ["icons/*.svg", "cyber/stix/*.json"]},
    classifiers=["Example :: Invalid"],
    # Declare orangedemo package to contain widgets for the "Cyber" category
    entry_points={
        "orange.widgets": (
            "Cyber = orangecyber.widgets",
        ),
        "orange.canvas.help": (
            'intersphinx = orangecyber.widgets:intersphinxdef',
        ),
        'orange.data.io.search_paths': (
            'stix = orangecyber.cyber',
        ),
    },
)