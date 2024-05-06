This project aims to design and create a model to analyze emotionally a subject based on its gait.

This model is created based on the [psymo] dataset. Due to hardware limits, it has been tested with a very reduced dataset.

The project uses poetry to manage Python dependencies and has been developed in Python 3.10.12.

# Train
First, you have to read the dataset and normalize it, for this purpose, you have to modify (if necessary) DATASET_PATH LABELS_PATH and DATA_PATH in [prepare_dataset.py] script. After checking or modifying the dataset path you have to run the following command
```bash
poetry run python3 prepare_dataset.py
```

After this dataset preprocessing, you can run the following command to train the model (you should check DATASET_PATH and LABELS_PATH in [graph_model.py] too)
```bash
poetry run python3 graph_model.py
```

[psymo]: https://github.com/cosmaadrian/psymo
[prepare_dataset.py]: https://github.com/izarte/gait_emotion_recognition/blob/main/prepare_dataset.py
[graph_model.py]: https://github.com/izarte/gait_emotion_recognition/blob/main/graph_model.py
