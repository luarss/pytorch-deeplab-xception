import os


b_files = open('./b_list.txt')
g_files = open('./g_list.txt')
bg_files = open('./bg_list.txt')


b_list = [b.replace('\n','') for b in b_files]
g_list = [g.replace('\n','') for g in g_files]
bg_list = [bg.replace('\n','') for bg in bg_files]

# list output (train/val.txt) in this direcotry
train_file = './Dataset/train.txt'
val_file   = './Dataset/val.txt'
test_file = './Dataset/test.txt'

val_frac = 0.2
train_frac = 1 - val_frac
assert val_frac < 1 and val_frac > 0
assert train_frac < 1 and train_frac > 0
num_b = 1000
num_g = 1000
num_bg = 1000
num_trainval = num_b + num_g + num_bg

# to preserve diversity
num_test = 300
num_b_test = int(num_b / num_trainval * num_test)
num_g_test = int(num_g / num_trainval * num_test)
num_bg_test = int(num_bg / num_trainval * num_test)

# Save training file list (only needed the name, not the extension)
with open(train_file, 'w') as train: 
    num_b_eff = int(num_b*train_frac)
    num_g_eff = int(num_g*train_frac)
    num_bg_eff = int(num_bg*train_frac)

    # copy bg files
    for idx in range(num_bg_eff):
        filename = bg_list[idx].split('.')[0] + '\n'
        train.write(filename)
    # copy b files
    for idx in range(num_b_eff):
        filename = b_list[idx].split('.')[0] + '\n'
        train.write(filename)

    # copy g files
    for idx in range(num_g_eff):
        filename = g_list[idx].split('.')[0] + '\n'
        train.write(filename)


with open(val_file, 'w') as val:
    for idx in range(num_bg_eff, num_bg):
        filename = bg_list[idx].split('.')[0] + '\n'
        val.write(filename)
    for idx in range(num_b_eff, num_b):
        filename = b_list[idx].split('.')[0] + '\n'
        val.write(filename)
    for idx in range(num_g_eff, num_g):
        filename = g_list[idx].split('.')[0] + '\n'
        val.write(filename)

with open(test_file, 'w') as test:
    for idx in range(num_bg, num_bg + num_bg_test):
        filename = bg_list[idx].split('.')[0] + '\n'
        test.write(filename)
    for idx in range(num_b, num_b + num_b_test):
        filename = b_list[idx].split('.')[0] + '\n'
        test.write(filename)
    for idx in range(num_g, num_g + num_g_test):
        filename = g_list[idx].split('.')[0] + '\n'
        test.write(filename)


