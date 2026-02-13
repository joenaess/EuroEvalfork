import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset, DatasetDict
from euroeval.data_loading import load_raw_data
from euroeval.data_models import DatasetConfig, Task
from euroeval.enums import TaskGroup
from euroeval.languages import ENGLISH


class TestColumnMapping:
    @pytest.fixture
    def mock_dataset(self):
        data = {"my_text": ["sample text"], "my_label": [0], "extra": [1]}
        dataset = Dataset.from_dict(data)
        return DatasetDict({"train": dataset, "val": dataset, "test": dataset})

    @pytest.fixture
    def task(self):
        return Task(
            name="test_task",
            task_group=TaskGroup.SEQUENCE_CLASSIFICATION,
            template_dict={ENGLISH: MagicMock()},
            metrics=[],
            default_num_few_shot_examples=0,
            default_max_generated_tokens=10,
        )

    def test_column_mapping_renames_columns(self, mock_dataset, task):
        config = DatasetConfig(
            task=task,
            languages=[ENGLISH],
            name="test_dataset",
            source="test_source",
            column_mapping={"my_text": "text", "my_label": "label"},
        )

        with patch("euroeval.data_loading.load_dataset", return_value=mock_dataset):
            loaded_dataset = load_raw_data(config, cache_dir=".", api_key=None)

        assert "text" in loaded_dataset["train"].column_names
        assert "label" in loaded_dataset["train"].column_names
        assert "my_text" not in loaded_dataset["train"].column_names
        assert "my_label" not in loaded_dataset["train"].column_names
        assert "extra" in loaded_dataset["train"].column_names

    def test_no_column_mapping(self, mock_dataset, task):
        config = DatasetConfig(
            task=task, languages=[ENGLISH], name="test_dataset", source="test_source"
        )

        with patch("euroeval.data_loading.load_dataset", return_value=mock_dataset):
            loaded_dataset = load_raw_data(config, cache_dir=".", api_key=None)

        assert "my_text" in loaded_dataset["train"].column_names
        assert "text" not in loaded_dataset["train"].column_names
