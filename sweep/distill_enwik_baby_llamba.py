from sweep.enwik_baby_llamba import *

out_dir = "out/dist"
with_dist = "True"  # "Fake"|"True"|"False"; str not bool watch out
wandb_log = True
dist_coef = 0.5
teacher_dir = "out/teacher"
layers_to_teach = [0]  # teaching only the 1st layer
