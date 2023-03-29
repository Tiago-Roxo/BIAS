import json
import os
import datasets


class COCOBuilderConfig(datasets.BuilderConfig):

    def __init__(self, name, splits, **kwargs):
        super().__init__(name, **kwargs)
        self.splits = splits


# Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{'{a} }r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  archivePrefix = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

# Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
COCO is a large-scale object detection, segmentation, and captioning dataset.
"""

# Add a link to an official homepage for the dataset here
_HOMEPAGE = "http://cocodataset.org/#home"

# Add the licence for the dataset here if you can find it
_LICENSE = ""

# Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)

# This script is supposed to work with local (downloaded) COCO dataset.
_URLs = {}


# Name of the dataset usually match the script name with CamelCase instead of snake_case
class COCODataset(datasets.GeneratorBasedBuilder):
    """An example dataset script to work with the local (downloaded) COCO dataset"""

    VERSION = datasets.Version("0.0.0")

    BUILDER_CONFIG_CLASS = COCOBuilderConfig
    BUILDER_CONFIGS = [
        COCOBuilderConfig(name='2017', splits=['train', 'valid', 'test']),
    ]
    DEFAULT_CONFIG_NAME = "2017"

    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset

        feature_dict = {
            "image_id": datasets.Value("int64"),
            "caption_id": datasets.Value("int64"),
            "caption": datasets.Value("string"),
            "height": datasets.Value("int64"),
            "width": datasets.Value("int64"),
            "file_name": datasets.Value("string"),
            "image_path": datasets.Value("string"),
        }

        features = datasets.Features(feature_dict)

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        data_dir = self.config.data_dir
        
        if not data_dir:
            raise ValueError(
                "This script is supposed to work with local (downloaded) COCO dataset. The argument `data_dir` in `load_dataset()` is required."
            )

        _DL_URLS = {
            "train": os.path.join(data_dir, "train2017.zip"),
            "val": os.path.join(data_dir, "val2017.zip"),
            "test": os.path.join(data_dir, "test2017.zip"),
            "annotations_trainval": os.path.join(data_dir, "annotations_trainval2017.zip"),
            "image_info_test": os.path.join(data_dir, "image_info_test2017.zip"),
        }
        archive_path = dl_manager.download_and_extract(_DL_URLS)
        
        splits = []
        for split in self.config.splits:
            if split == 'train':
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "json_path": os.path.join(archive_path["annotations_trainval"], "annotations", "captions_train2017.json"),
                        "image_dir": os.path.join(archive_path["train"], "train2017"),
                        "split": "train",
                    }
                )
            elif split in ['val', 'valid', 'validation', 'dev']:
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "json_path": os.path.join(archive_path["annotations_trainval"], "annotations", "captions_val2017.json"),
                        "image_dir": os.path.join(archive_path["val"], "val2017"),
                        "split": "valid",
                    },
                )
            elif split == 'test':
                dataset = datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "json_path": os.path.join(archive_path["image_info_test"], "annotations", "image_info_test2017.json"),
                        "image_dir": os.path.join(archive_path["test"], "test2017"),
                        "split": "test",
                    },
                )
            else:
                continue

            splits.append(dataset)

        return splits

    def _generate_examples(
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
        self, json_path, image_dir, split
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        _features = ["image_id", "caption_id", "caption", "height", "width", "file_name", "image_path", "id"]
        features = list(_features)

        if split in "valid":
            split = "val"

        with open(json_path, 'r', encoding='UTF-8') as fp:
            data = json.load(fp)

        # list of dict
        images = data["images"]
        entries = images

        # build a dict of image_id -> image info dict
        d = {image["id"]: image for image in images}

        # list of dict
        if split in ["train", "val"]:
            annotations = data["annotations"]

            # build a dict of image_id ->
            for annotation in annotations:
                _id = annotation["id"]
                image_info = d[annotation["image_id"]]
                annotation.update(image_info)
                annotation["id"] = _id

            entries = annotations

        for id_, entry in enumerate(entries):

            entry = {k: v for k, v in entry.items() if k in features}

            if split == "test":
                entry["image_id"] = entry["id"]
                entry["id"] = -1
                entry["caption"] = -1

            entry["caption_id"] = entry.pop("id")
            entry["image_path"] = os.path.join(image_dir, entry["file_name"])

            entry = {k: entry[k] for k in _features if k in entry}

            yield str((entry["image_id"], entry["caption_id"])), entry