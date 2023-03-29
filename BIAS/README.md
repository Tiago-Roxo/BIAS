# BIAS: A Body-based Interpretable Active Speaker Approach

1. Download the AVA-ActiveSpeaker [\[github\]](https://github.com/cvdfoundation/ava-dataset) [\[website\]](https://research.google.com/ava/download.html#ava_active_speaker_download) and [WASD](https://tiago-roxo.github.io/WASD/) datasets;
2. Create the environment with `conda` using the script `create_env.sh`;
3. To train the model, execute the command:
```bash
python3 main.py --dataPath $DATASET
```
where `$DATASET` is the folder containing the dataset (AVA-ActiveSpeaker or WASD). Alternatively, you can run the script `train.sh` which contains an example of training the model in WASD. After each epoch, the model weights will be saved and output its predictions to a CSV (`val_res.csv`), in the `exps/wasd` folder (by default);

4. To evaluate the model, execute the command:
```bash
python3 main.py --dataPath $DATASET --evaluation
```
where `$DATASET` is the folder containing the dataset (AVA-ActiveSpeaker or WASD). Alternatively, you can run the script `infer.sh` which contains an example of evaluating the model in WASD. The model predictions are saved in a CSV (`val_res.csv`), in the `exps/wasd` folder (by default).

The BIAS model **trained on WASD** can be downloaded [here](https://drive.google.com/file/d/1emfDPgBAfQGNwMsnW4E6Tduxq2OYyKsB/view?usp=share_link).

The BIAS model **trained on AVA-ActiveSpeaker** can be downloaded [here](https://drive.google.com/file/d/1HqX6Fgfjz0hfgfmOjdqQ0c0LmPI1oE1Q/view?usp=share_link).



## Cite

```bibtex

```
