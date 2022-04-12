# KID
Source code for "A Double-Graph Based Framework for Frame Semantic Parsing" @ NAACL 2022
## Overview
TL;DR: A framework for frame semantic parsing aims to utilize double-graph structure to inject knowledge into parsing and strengthen interacrions between arguments.

Paper Link: [A Double-Graph Based Framework for Frame Semantic Parsing](https://openreview.net/pdf?id=STUnTbwKMXm), TODO: replace it with camera-ready version.

We find Ontological Frame Knowledge can contribute to frame semantic parsing by both intra-frame and inter-frame reasoning. To make use of it, we build FKG (Frame Knowledge Graph) on definitions of FEs, frame relations and FE mappings. Besides, We regard frame semantic parsing as a process to add nodes in FSG (Frame Semantic Graph) incrementally, which can strengthen relations beween arguments and interactions of subtasks in frame semantic parsing. 

## Requirements

## How to Run Our Code?
### Datapreprocess
You can get some data files from scratch, we will also provide [links](https://drive.google.com/drive/folders/1MRqhlWwHsAVnwH-HUb8990nTasvkK-fR?usp=sharing) to download remaining data files in future.


Run `data_preprocess.py` to get data_instance_dic and word/lemma vectors from [GloVe](https://nlp.stanford.edu/projects/glove/).
```
python data_preprocess.py
```

Then run `dep_parsing.py` to build dependency trees for each sentence.
```
python dep_parsing.py
```

Run `build_fe_name_dict.py` to get fe_label_to_name.npy.
```
python build_fe_name_dict.py
```

