1. Run ``python ./script/run_simulation.py --solvent vacuum`` for the simulation in vacuum.

2. Run ``python ./script/run_simulation.py --solvent OBC2`` for the simulation in implicit solvent

3. Run ``python ./script/make_coor_transformer_and_compute_internal_coordinate.py --solvent vacuum`` to convert Cartesian coordinates into internal coordinates for simulations in vacuum.

4. Run ``python ./script/make_coor_transformer_and_compute_internal_coordinate.py --solvent OBC2`` to convert Cartesian coordinates into internal coordinates for simulations in implicit solvent.

5. Run ``python ./script/learn_mmflow_model.py --solvent vacuum --num_transforms 20 --hidden_size 16`` to learn mmflow models using internal coordinates from vaccum simulations

6. Run ``python ./script/learn_mmflow_model.py --solvent OBC2 --num_transforms 20 --hidden_size 16`` to learn mmflow models using internal coordinates from implicit solvent simulations

7. Run ``python ./script/sample_from_mmflow_model_and_compute_log_q.py --solvent vacuum --num_transforms 20 --hidden_size 16`` to sample from the mmflow model trained on vaccum simulations

8. Run ``python ./script/sample_from_mmflow_model_and_compute_log_q.py --solvent OBC2 --num_transforms 20 --hidden_size 16`` to sample from the mmflow model trained implicit solvent simlations.

9. Run ``python ./script/compute_log_p.py --solvent vacuum --num_transforms 20 --hidden_size 16`` to compute energy on samples from the mmflow model trained on vaccum simulations

10. Run ``python ./script/compute_log_p.py --solvent vacuum --num_transforms 20 --hidden_size 16`` to compute energy on samples from the mmflow model trained on implicit solvent simulations.

11. Run ``python ./script/compute_free_energy_with_fastmbar.py --solvent vacuum --num_transforms 20 --hidden_size 16`` to compute absolute free energy for vaccum state, which is the value of  ``fastmbar.F[-1]``. You should record this number.
    
12. Run ``python ./script/compute_free_energy_with_fastmbar.py --solvent OBC2 --num_transforms 20 --hidden_size 16`` to compute absolute free energy for implicit solvent state, which is the value of  ``fastmbar.F[-1]``. You should record this number too.

The value from step 12 minus the value from step 11 will be the result of solvation free energy calculated using DeepBAR.
We can compare this value with that calculated using traditional approach which is step 13

13. 12. Run ``compute_deltaF_with_alchem.py`` to compute solvation free energy using the traditional approach. The result is the value of  ``fastmbar.F[-1]``. This number should be close to the result from DeepBAR


    


    

   

   
   
   
   
