from hypersurrogatemodel.utils import Logger
import pandas as pd
import json
import os
from pathlib import Path

logger = Logger("DataReconstruct")

class log_iterator():
    def __init__(self, lst):
        self.iterable = lst
        self.index = 0
        self.length = len(lst)
        self.current_item = None
    
    def __iter__(self):
        return self
    
    def next(self):
        if self.current_item is not None:
            logger.success(f"Step {self.index}/{self.length}: {self.current_item} success")
        if self.index == self.length:
            return
        if self.index < self.length:
            self.current_item = self.iterable[self.index]
            self.index += 1
            logger.info(f"Step {self.index}/{self.length}: {self.current_item}...")
        else:
            raise StopIteration
        
class DatasetStruct():
    def __init__(self,data_name:str, data:pd.DataFrame, partition = -1):  # input needs: source data, data name, data description, data goal
        self.data_name = data_name
        self.datas = data
        self.partition = partition
    def data_cleaning (self):
        self.answer = self.datas[["true_final_train_accuracy"]][:self.partition]
        self.text = self.datas[['uid', 'unified_text_description']][:self.partition]
        

        answer_dict = self.answer.to_dict(orient='records')
        data_dict = self.text.to_dict(orient='records')
        self.wrapped_data = []
        for data, answer in zip(data_dict, answer_dict):
            self.wrapped_data.append({"text": str(data), "answer": answer["true_final_train_accuracy"]})  
        logger.success("Data cleaned: reformat to json serializable")
        
    
    def save_to_json(self, file_path:str):
        with open(file_path, 'w+') as f:
            json.dump(self.wrapped_data, f, indent=2, ensure_ascii=False)
        logger.success(f"DataFrame saved as JSON to {file_path}")
    
def NAS_bench_201(part = -1):
    logger.step("loading data set")
    original_data_path = Path(os.getcwd()) / "data" / "processed" / "master_dataset.parquet"
    try:
        original_data = pd.read_parquet(original_data_path)
        logger.success(f"data set loaded from {original_data_path}")
        
        dataset_union = []
        for dataset_source in original_data['dataset_source'].unique():
            dataset_union.append(DatasetStruct(data_name=dataset_source, data=original_data[original_data['dataset_source']==dataset_source],partition=part))
        
        
    except Exception as e:
        logger.error(str(e))
    
    json_output_path = Path(os.getcwd()) / "data" / "processed" / "NAS_bench_201"
    pathjoint = []
    for ds in dataset_union:
        ds.data_cleaning()
        ds.save_to_json(json_output_path / f"{ds.data_name}_cleaned.json")
        pathjoint.append(json_output_path / f"{ds.data_name}_cleaned.json")
    return pathjoint

if __name__ == "__main__":
    NAS_bench_201()