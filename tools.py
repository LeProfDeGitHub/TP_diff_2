import os
import shutil


def add_floders():
    """
    Create a new folder 'figure' and subfolders 'grad_desc_fix_step' and 'grad_desc_optimal_step'
    if they already exist, they are deleted and recreated.
    """
    if os.path.exists('figure'):
        shutil.rmtree('figure')
        
    os.makedirs('figure')
    os.makedirs('figure\\grad_desc_fix_step')
    os.makedirs('figure\\grad_desc_optimal_step')
    os.makedirs('figure\\conjuguate_gradient')
    os.makedirs('figure\\grad_desc_fix_step\\partial_func')
    os.makedirs('figure\\grad_desc_optimal_step\\partial_func')

