import pandas as pd
import json
import numpy as np

def extract_image_info(path: str, seed: int = 13, keep_divisible_16 = True) -> pd.DataFrame:
    """
    Create a DataFrame with 4 columns: 
        image jpeg name to retrieve the original image
        height and width 
        A unique caption. The original daaset contains 5 captions per image, but we only keep one randomly.

    This functions works on the MS COCO dataset. 

    Args:
        path (str): path to captions_val2017.json of MS COCO dataset.
        seed (int, optional): random seed to select the unique caption.

    Returns:
        pd.DataFrame: DataFrame with 4 columns: image jpeg name, height, width, caption
    """
    # set random seed for reproducibility of caption selection
    np.random.seed(seed)

    # open json file and load data
    with open(path, 'r') as f:
        data = json.load(f)

    # extract captions
    caption_df = pd.DataFrame(data['annotations'])[['image_id', 'caption']].iloc[np.random.permutation(len(data['annotations']))].reset_index(drop=True)
    caption_df.drop_duplicates(subset=["image_id"], keep="first", inplace=True)

    # extract image dimension and jpeg name
    images_df = pd.DataFrame(data['images'])[['id', 'file_name', 'height', 'width']].rename(columns={'id': 'image_id'})
    df = pd.merge(caption_df, images_df, on='image_id', how='inner').drop(columns=['image_id'])

    if keep_divisible_16:
        df = df[df.apply(_non_divisible16, axis=1)]

    # free space
    del data, caption_df, images_df

    return df

def _non_divisible16(x):
    return (x['height'] % 16 == 0) and (x['width'] % 16 == 0)