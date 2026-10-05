# Dataset Directory: AgriFreshNET Processed_Data

Place your pre-extracted **AgriFreshNET Processed_Data** dataset inside this directory.

## Expected Directory Hierarchy

Your processed dataset contains **24 distinct classes** (8 food items across 3 freshness levels: Fresh, Semi-Fresh, and Rotten).

The directory structure can follow either a standard split format:

```text
dataset/
├── train/
│   ├── fresh_banana/
│   ├── semifresh_banana/
│   ├── rotten_banana/
│   ├── fresh_orange/
│   ├── ... (all 24 classes)
├── val/
│   ├── ... (all 24 classes)
└── test/
    ├── ... (all 24 classes)
```

Or a single processed folder containing class subdirectories:
```text
dataset/
├── AgriFreshNET_Processed_Data/
│   ├── fresh_banana/
│   ├── semifresh_banana/
│   ├── rotten_banana/
│   └── ... (24 class directories)
```

## Supported 24 Classes:
- **Banana**: `fresh_banana`, `semifresh_banana`, `rotten_banana`
- **Orange**: `fresh_orange`, `semifresh_orange`, `rotten_orange`
- **Pineapple**: `fresh_pineapple`, `semifresh_pineapple`, `rotten_pineapple`
- **Tomato**: `fresh_tomato`, `semifresh_tomato`, `rotten_tomato`
- **Eggplant**: `fresh_eggplant`, `semifresh_eggplant`, `rotten_eggplant`
- **Cucumber**: `fresh_cucumber`, `semifresh_cucumber`, `rotten_cucumber`
- **Papaya**: `fresh_papaya`, `semifresh_papaya`, `rotten_papaya`
- **Bittermelon**: `fresh_bittermelon`, `semifresh_bittermelon`, `rotten_bittermelon`

*Note: The dataset loader in `src/dataset.py` automatically detects and maps class folder names regardless of casing or underscores.*
