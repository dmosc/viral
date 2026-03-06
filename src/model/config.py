class Config:
    d_model: int = 64
    num_tabular_features: int = 10
    text_model_id = 'distilbert-base-uncased'
    video_model_id = 'MCG-NJU/videomae-base'
    max_text_len = 256
    num_frames = 16
    checkpoint_path = "./checkpoints"
    num_workers = 4
    dataset_id = 'rodmosc/viral'
