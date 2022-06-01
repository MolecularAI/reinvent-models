import os
import json

project_root = os.path.dirname(__file__)
with open(os.path.join(project_root, '../../reinvent_models/configs/config.json'), 'r') as f:
    config = json.load(f)

MAIN_TEST_PATH = config["MAIN_TEST_PATH"]

OLD_PRIOR_PATH = config["OLD_PRIOR_PATH"]
PRIOR_PATH = config["PRIOR_PATH"]
OLD_LIBINVENT_PRIOR_PATH = config["OLD_LIBINVENT_PRIOR_PATH"]
LIBINVENT_PRIOR_PATH = config["LIBINVENT_PRIOR_PATH"]
LINK_INVENT_PRIOR_PATH = config["LINK_INVENT_PRIOR_PATH"]