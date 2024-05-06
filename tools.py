import os


def add_floders():
    if not os.path.exists('figure'):
        os.makedirs('figure')
    if not os.path.exists('figure\\grad_desc_fix_step'):
        os.makedirs('figure\\grad_desc_fix_step')
    if not os.path.exists('figure\\grad_desc_optimal_step'):
        os.makedirs('figure\\grad_desc_optimal_step')
    if not os.path.exists('figure\\newton_optimal_step'):
        os.makedirs('figure\\newton_optimal_step')
    if not os.path.exists('figure\\newton_fix_step'):
        os.makedirs('figure\\newton_fix_step')

