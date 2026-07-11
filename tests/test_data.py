from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mlops_eval.data import generate_dataset, read_dataset, stratified_split, write_dataset


class DataTests(unittest.TestCase):
    def test_generation_is_deterministic(self):
        self.assertEqual(generate_dataset(20, 7), generate_dataset(20, 7))

    def test_round_trip_and_split(self):
        records = generate_dataset(120)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "customers.csv"
            write_dataset(records, path)
            loaded = read_dataset(path)
        self.assertEqual(records, loaded)
        train, validation, test = stratified_split(loaded)
        self.assertEqual(len(records), len(train) + len(validation) + len(test))
        train_ids = {record.customer_id for record in train}
        validation_ids = {record.customer_id for record in validation}
        test_ids = {record.customer_id for record in test}
        self.assertTrue(train_ids.isdisjoint(validation_ids))
        self.assertTrue(train_ids.isdisjoint(test_ids))
        self.assertTrue(validation_ids.isdisjoint(test_ids))
        self.assertEqual({record.customer_id for record in records}, train_ids | validation_ids | test_ids)


if __name__ == "__main__":
    unittest.main()
