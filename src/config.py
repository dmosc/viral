class Config:
    epochs = 10
    d_model: int = 64
    num_tabular_features: int = 17
    text_model_id = 'distilbert-base-uncased'
    video_model_id = 'MCG-NJU/videomae-base'
    object_detection_model_id = 'facebook/detr-resnet-50'
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
    # Viral examples are upweighted by this factor to counter class
    # imbalance during training.
    viral_loss_weight = int(p_virality_threshold / (1 - p_virality_threshold))
    video_resolution = (224, 224)
    required_dims = [
        'author_follower_count',
        'author_following_count',
        'author_total_heart_count',
        'author_video_count',
        'author_friend_count',
        'duration',
        'width',
        'height',
        'aspect_ratio',
        'vq_score',
        'user_verified',
        'is_private',
        'is_ad',
        'share_enabled',
        'stitch_enabled',
        'day_of_week',
        'hour_of_day',
        'engagement_score',
        'view_velocity_score',
        'video_bytes'
    ]
