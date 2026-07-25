# src/oraclebot/data/dataset.py
# Generische JSON-Lines-Persistenz fuer Trainingsbeispiele (verwendet von train_barrier_model.py).
import json
import logging

logger = logging.getLogger(__name__)


def save_dataset_jsonl(examples: list, path: str):
    """Speichert Trainingsbeispiele als JSON-Lines-Datei (ein Beispiel pro Zeile)."""
    with open(path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    logger.info(f"{len(examples)} Beispiele gespeichert: {path}")


def load_dataset_jsonl(path: str) -> list:
    """Laedt Trainingsbeispiele aus einer JSON-Lines-Datei."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples
