import os
import pathlib
import glob
import pandas as pd
import shutil

def check_combined_dataset(loc):
    
    label_file = os.path.join(loc, 'labels.csv')
    
    if not os.path.exists(label_file):
        raise ValueError("labels.csv doesn't exist. Please ensure loc argument is correct.")

    df_labels = pd.read_csv(label_file)
    csv_filenames = set(df_labels['filename'])
    
    filesystem_files = set(glob.glob(f"{loc}/*.png"))
    
    #ensures exactly same image files in filesystem tree and in csv files
    assert(len(csv_filenames.symmetric_difference(filesystem_files))==0)
    
    #ensure labels are correct between class column and image filenames
    N_correct = (df_labels['filename'].apply(lambda x: x.split('/')[-1].split('_')[0])==df_labels['class']).sum()
    
    assert(N_correct==df_labels.shape[0])
    
def combine_datasets(loc):
    '''Hacky - replace os by pathlib
    Add checks
    '''
    
    target_path = os.path.join(loc, 'combined')
    if os.path.exists(target_path):
        shutil.rmtree(target_path, ignore_errors=True)
    
    logo_list = os.listdir(loc) #rh,anaconda...
    dataset_type_list = os.listdir(os.path.join(loc, logo_list[0]))#train,test
    
    for dataset in dataset_type_list:#create combined/train, combined/test
        pathlib.Path(os.path.join(loc, 'combined', dataset)).mkdir(parents=True, exist_ok=True)
    
    #copy images over
    df_dict = {}
    for logo in logo_list: #rh
        for dataset in dataset_type_list: #train
            #csv files
            if dataset not in df_dict:
                df_dict[dataset] = []
            df_dict[dataset].append(pd.read_csv(os.path.join(loc, logo, dataset, f'{logo}.csv')))
            
            for img_file in glob.glob(os.path.join(loc, logo, dataset, '*.png')):
                shutil.copy2(img_file, os.path.join(loc, 'combined', dataset))
    
    def create_path(path, idx_to_exclude):
        return '/'.join([val if idx!=idx_to_exclude else "combined" for idx, val in enumerate(path.split('/'))]) 
    
    #process csvs
    for dataset in df_dict:
        df = pd.concat(df_dict[dataset], axis=0)
        
        df['filename'] = df['filename'].apply(lambda x: create_path(x, 1))
        
        df.to_csv(os.path.join(loc, 'combined', dataset, 'labels.csv'), index=False)
        
        check_combined_dataset(os.path.join(loc, 'combined', dataset))
