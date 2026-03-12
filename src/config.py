class Config:
    seed = 42
    epochs = 15
    batch_size = 32
    gradient_accumulation_steps = 4
    max_grad_norm = 1.0
    d_model = 512
    train_size = 0.5
    test_size = 0.1
    dropout = 0.3
    num_tabular_features: int = 19
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
    viral_loss_weight = 12
    regression_loss_contribution = 0.3
    classification_loss_contribution = 1 - regression_loss_contribution
    video_resolution = (224, 224)
    # Rebalancing config (used by train_rebalanced.py)
    rebalance_strategy = 'oversample'  # 'none', 'oversample', 'smote_hybrid'
    target_viral_ratio = 0.3
    val_start_month = '2025-01'
    test_start_month = '2025-04'
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
