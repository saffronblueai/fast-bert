from fast_bert.prediction import BertClassificationPredictor
from pathlib import Path
import json
import pandas as pd
import sys
import time

import s3fs
import subprocess

from fastscript import Param, call_parse

s3 = s3fs.S3FileSystem(anon=False)

def download_uri(s):
    filename = s.split('/')[-1]
    print(f"Downloading {filename} from S3 ...")
    local_file = f"/tmp/{filename}"
    s3.get(s[5:], local_file)
    print(f"Downloaded {filename}")
    return local_file
    
def upload_uri(local_file, s):
    filename = s.split('/')[-1]
    print(f"Uploading {filename} to S3 ...")
    s3.put(local_file, s[5:])
    print(f"Uploaded {filename}")

@call_parse
def main( model_uri: Param("S3 uri with NLP model", str),
          data_uri: Param("S3 uri with input csv file", str),
          result_uri: Param("S3 uri where to put output csv file with added \
                                inference columns", str),
          inference_columns: Param("text columns separated in the csv file on \
                        which inference will be run", str) ):
    try:
        local_model = download_uri(model_uri)
    except:
        print(f"Failed to download NLP model. Exiting...")
        sys.exit(2)
        
    try:
        local_csv = download_uri(data_uri)
    except:
        print(f"Failed to download input csv file. Exiting...")
        sys.exit(2)
        
    model_dir = Path("/tmp/model")
    model_dir.mkdir(exist_ok=True)
    
    out = subprocess.Popen(['tar', 'xzf', local_model, '-C', model_dir],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
                
    stdout, stderr = out.communicate()
    if not stderr:
        print("Model extacted sucessfully")
    else:
        print(stderr.decode('ascii'))
        print(f"Model extaction error. Exiting...")
        sys.exit(1)
        
    model_config = model_dir / 'model_config.json'
    with open(model_config) as f:
        config = json.load(f)
        
    print("Loading model")
        
    predictor = BertClassificationPredictor(
        model_path=str(model_dir / 'model_out'),
        label_path=str(model_dir), # location for labels.csv file
        model_type=config['model_type'],
        multi_label=config['multi_label'],
        do_lower_case=config['do_lower_case'],
    )
    try:
        print("Loading input csv")
        df = pd.read_csv(local_csv)
    except:
        print("Failed to load input csv file. Exiting...")
        sys.exit(1)
        
    inference_columns = inference_columns.split(',')
    for c in inference_columns:
        if c not in df.columns:
            print(f"{c} is not a column name in input csv file. Exiting...")
            sys.exit(2)
            
    for c in inference_columns:
        
        print(f"Starting inference for {c} column")
        
        start = time.time()
        
        text = df.loc[~df[c].isna(), c].tolist()
        
        out = predictor.predict_batch(text)
        result = pd.DataFrame(list(map(dict, out)))
        for r in result.columns:
            df.loc[~df[c].isna(), f"{c}_{r}"] = result[r].tolist()
            
        print(f"Inference time for {len(text)} rows was {time.time() - start}")
        
    df.to_csv(local_csv, index=False)
    
    upload_uri(local_csv, result_uri)
    
    print("We are done with inference!")
