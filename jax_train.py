import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments

from arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from trainer import TevatronTrainer

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TevatronTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Ensure the arguments are correctly parsed and used
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training arguments: {training_args}")

    # Further processing and training setup
    # ...

if __name__ == "__main__":
    main()
