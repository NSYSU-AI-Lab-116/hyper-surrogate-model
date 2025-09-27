from hypersurrogatemodel.utils import Logger
import pandas as pd
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from hypersurrogatemodel.config import config

logger = Logger("DataReconstruct")


class log_iterator:
    def __init__(self, lst):
        self.iterable = lst
        self.index = 0
        self.length = len(lst)
        self.current_item = None

    def __iter__(self):
        return self

    def next(self):
        if self.current_item is not None:
            logger.success(
                f"Step {self.index}/{self.length}: {self.current_item} success"
            )
        if self.index == self.length:
            return
        if self.index < self.length:
            self.current_item = self.iterable[self.index]
            self.index += 1
            logger.info(f"Step {self.index}/{self.length}: {self.current_item}...")
        else:
            raise StopIteration


class DatasetStruct:
    def __init__(
        self, data_name: str, data: pd.DataFrame, partition=-1
    ):  # input needs: source data, data name, data description, data goal
        self.data_name = data_name
        self.datas = data
        self.partition = partition

    def data_cleaning(self) -> None:
        self.answers = self.datas["true_final_train_accuracy"][
            : self.partition
        ].tolist()
        self.text = self.datas[["uid", "unified_text_description"]][
            : self.partition
        ].to_dict(orient="records")
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.text, self.answers, test_size=0.2, random_state=42
        )

        for i, _ in enumerate(iterable=self.train_x):
            self.train_x[i]["true_acc"] = self.train_y[i]
        logger.success(f"{self.data_name} Train data reformatted")

        for i, _ in enumerate(iterable=self.test_x):
            self.test_x[i]["true_acc"] = self.test_y[i]
        logger.success(f"{self.data_name} Test data reformatted")

    def save_to_json(self):
        os.makedirs(name=config.dataset.preprocess_train_path, exist_ok=True)  # type: ignore
        with open(
            os.path.join(
                config.dataset.preprocess_train_path, f"{self.data_name}_train.json"
            ),
            "w+",
        ) as ftr:  # type: ignore
            json.dump(self.train_x, ftr, indent=2, ensure_ascii=False)
            logger.success(
                f"{self.data_name}_train saved as JSON to {config.dataset.preprocess_train_path}"
            )

        os.makedirs(config.dataset.preprocess_test_path, exist_ok=True)  # type: ignore
        with open(
            os.path.join(
                config.dataset.preprocess_test_path, f"{self.data_name}_test.json"
            ),
            "w+",
        ) as fte:  # type: ignore
            json.dump(self.test_x, fte, indent=2, ensure_ascii=False)
            logger.success(
                f"{self.data_name}_test saved as JSON to {config.dataset.preprocess_test_path}"
            )


def NAS_bench_201(part=-1):
    logger.step("loading data set")
    original_data_path = (
        Path(os.getcwd()) / "data" / "processed" / "master_dataset.parquet"
    )
    try:
        original_data = pd.read_parquet(original_data_path)
        logger.success(f"data set loaded from {original_data_path}")

        dataset_union = []
        for dataset_source in original_data["dataset_source"].unique():
            dataset_union.append(
                DatasetStruct(
                    data_name=dataset_source,
                    data=original_data[
                        original_data["dataset_source"] == dataset_source
                    ],
                    partition=part,
                )
            )

        json_output_path = config.dataset.preprocess_train_path
        logger.info(message=f"debug msg: json output path: {json_output_path}")

        for ds in dataset_union:
            ds.data_cleaning()
            ds.save_to_json()  # type: ignore
    except Exception as e:
        logger.error(message=str(e))


if __name__ == "__main__":
    NAS_bench_201(config.dataset.dataset_partition)  # type: ignore
