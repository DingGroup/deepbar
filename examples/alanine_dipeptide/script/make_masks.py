__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2020/05/16 00:57:25"

import numpy as np
import simtk.openmm.app as app
import math

def make_masks_0(dim, low, high):
    n = dim
    masks = []

    for i in range(3):
        mask_filter = np.ones(n)
        mask_filter[0::2] = 0
        masks.append({'name': 'Affine Coupling',
                      'filter': list(mask_filter),
                      'parameters': None,
                      'forward': True})

        mask_filter = np.ones(n)
        mask_filter[1::2] = 0
        masks.append({'name': 'Affine Coupling',
                      'filter': list(mask_filter),
                      'parameters': None,
                      'forward': True})
        
        mask_filter = np.ones(n)
        mask_filter[0::3] = 0
        masks.append({'name': 'Affine Coupling',
                      'filter': list(mask_filter),
                      'parameters': None,
                      'forward': True})

        mask_filter = np.ones(n)
        mask_filter[1::3] = 0
        masks.append({'name': 'Affine Coupling',
                      'filter': list(mask_filter),
                      'parameters': None,
                      'forward': True})

        mask_filter = np.ones(n)
        mask_filter[2::3] = 0
        masks.append({'name': 'Affine Coupling',
                      'filter': list(mask_filter),
                      'parameters': None,
                      'forward': True})
                
    mask_filter = np.ones(n)
    mask_filter[:] = 0.
    masks.append({'name': 'Scale',
                  'filter': list(mask_filter),
                  'parameters': {'low': low, 'high': high},
                  'forward': True})
    
    return masks

def make_masks_1(dim, low, high):
    n = dim
    masks = []

    mask_filter = np.ones(n)
    mask_filter[:] = 0.
    masks.append({'name': 'Scale',
                  'filter': list(mask_filter),
                  'parameters': {'low': low, 'high': high},
                  'forward': False})
    
    for stride in [8, 4, 2, 1]:
        for k in range(2):
            mask_filter = np.ones(n)
            flag = [ i%(2*stride) < stride  for i in range(n)]
            mask_filter[flag] = 0
            masks.append({'name': "Affine Coupling",
                          'filter': list(mask_filter),
                          'parameters': None,
                          'forward': True})
            
            mask_filter = np.ones(n)
            flag = [ i%(2*stride) >= stride  for i in range(n)]
            mask_filter[flag] = 0
            masks.append({'name': "Affine Coupling",
                          'filter': list(mask_filter),
                          'parameters': None,
                          'forward': True})                    
    
    return masks
