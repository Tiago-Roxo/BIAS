# BIAS Interpretability

1. Download the AVA-ActiveSpeaker [\[github\]](https://github.com/cvdfoundation/ava-dataset) [\[website\]](https://research.google.com/ava/download.html#ava_active_speaker_download) and [WASD](https://tiago-roxo.github.io/WASD/) datasets;
2. To obtain the perceived importance, we use the values of the SE vector for each feature (audio, body, and face). These can be obtained by executing the command:
```bash
python3 main.py \
    --evaluation \
    --dataPath $DATASET \
    --inputModel pretrained_BIAS_Visual.model \
    --outputFile pretrained_BIAS_Visual.csv
```
which will use the `pretrained_BIAS_Visual.model` to output the predictions into the `pretrained_BIAS_Visual.csv` and save the values of SE vector for each feature in `pretrained_BIAS_Visual.pkl` in the `vetor_dictionary` folder. The content of the created .pkl is the following:
```
{
    "entity_id": {
        "audio": list with 128 values,
        "face": list with 128 values,
        "body": list with 128 values,
        "body_backbone": list with 512 values,
        "face_backbone": list with 512 values,
    }
}
```
where `entity_id` refers to an `entity_id` of the CSV files from WASD or AVA-ActiveSpeaker.

3. To obtain the attention heatmaps, execute the command:
```bash
python3 visual_interpretability.py
```
which will use the `pretrained_BIAS_Visual.model` to output the heatmaps of body and face into the `visual_images` folder. The heatmaps are created based on the index of the `dataLoader.py`, which can be changed in line #152 of this file. The index choosen relates to a line of the `val_loader.csv` (from the WASD or AVA-ActiveSpeaker dataset folder). 

*E.g.*, if line #152 contains `line = [self.miniBatch[index+586]]`, it refers to line #587 of `val_loader.csv`, which corresponds to `entity_id` *G0CArrOND6I_21-49_0000_0060:1*. 

The Visual BIAS model can be downloaded [here](https://drive.google.com/file/d/1ToC1o9TDSVMLVrRSFlbQ8qxXbiIRNJkB/view?usp=share_link).
 

## Cite

```bibtex
@misc{roxo2024bias,
      title={BIAS: A Body-based Interpretable Active Speaker Approach}, 
      author={Tiago Roxo and Joana C. Costa and Pedro R. M. Inácio and Hugo Proença},
      year={2024},
      eprint={2412.05150},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05150}, 
}
```
