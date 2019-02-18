import os
import argparse
import random

VIEWS = 12
NCATS = 40

def prepare_data(file,shuffle=False):
    all_files = []
    with open(file, "r") as f:
        for line in f:
            line2 = line.split()[0]
            with open(line2) as f2:
                cat = f2.readline()
                f2.readline().strip()
                one_model = []
                for view in range(VIEWS):
                    to_append = "{} {}".format(f2.readline().strip(),cat)
                    one_model.append(to_append)
                rot_index = random.randint(0,VIEWS)
                one_model = one_model[rot_index:]+one_model[0:rot_index]
                all_files.append(one_model)
    if shuffle:
        random.shuffle(all_files)

    final_files = []
    for f in all_files:
        for ff in f:
            final_files.append(ff)
    print(len(final_files))
    
    with open(file.split('.')[0]+"rotnet.txt", 'w') as f:
        for line in final_files:  
            print(line)
            f.write(line)
        
        
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='')
    
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    
    prepare_data(args.file, args.shuffle)
                
    
    