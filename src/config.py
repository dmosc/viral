class Config:
    d_model: int = 64
    num_tabular_features: int = 5
    text_model_id = 'distilbert-base-uncased'
    video_model_id = 'MCG-NJU/videomae-base'
    max_text_len = 256
    num_frames = 16
    checkpoint_path = "data/checkpoints"
    num_workers = 4
    base_dataset_id = 'The-data-company/TikTok-10M'
    dataset_id = 'rodmosc/viral'
    data_path = 'data/videos'
    engagement_weights = {
        'shares': 4,
        'saves': 3,
        'comments': 2,
        'likes': 1
    }
    p_virality_threshold = 0.95
