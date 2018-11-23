from fitters.plane_fitter import PlaneFitter
from fitters.sphere_fitter import SphereFitter
from fitters.cylinder_fitter import CylinderFitter
from fitters.cone_fitter import ConeFitter

import numpy as np

NAME_TO_FITTER_DICT = {
    'plane': PlaneFitter,
    'sphere': SphereFitter,
    'cylinder': CylinderFitter,
    'cone': ConeFitter,
}
primitive_name_to_id_dict = {}
all_fitter_classes = []

def get_all_fitter_classes():
    return all_fitter_classes

def primitive_name_to_id(name):
    return primitive_name_to_id_dict[name]

def get_n_registered_primitives():
    return len(all_fitter_classes)
    
def create_fitters(primitive_name_list):
    return [NAME_TO_FITTER_DICT[name]() for name in primitive_name_list]

def register_primitives(primitive_name_list):
    # Must be called once before everything
    global all_fitter_classes, primitive_name_to_id_dict
    all_fitter_classes = []
    primitive_name_to_id_dict = {}

    for idx, name in enumerate(primitive_name_list):
        fitter = NAME_TO_FITTER_DICT[name]
        all_fitter_classes.append(fitter)
        primitive_name_to_id_dict[name] = idx

    print('Registered ' + ','.join(primitive_name_list))

def create_primitive_from_dict(d):
    if d['type'] not in NAME_TO_FITTER_DICT.keys():
        return None
    return NAME_TO_FITTER_DICT[d['type']].create_primitive_from_dict(d)
