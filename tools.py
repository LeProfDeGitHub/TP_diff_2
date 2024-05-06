import os
import shutil


def add_floders():
    if os.path.exists('figure'):
        shutil.rmtree('figure')
        
    os.makedirs('figure')
    os.makedirs('figure\\grad_desc_fix_step')
    os.makedirs('figure\\grad_desc_optimal_step')
    os.makedirs('figure\\grad_desc_fix_step\\partial_func')
    os.makedirs('figure\\grad_desc_optimal_step\\partial_func')

