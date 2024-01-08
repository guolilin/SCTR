class hparams:

    train_or_test = 'test'
    output_dir = 'bak/bak_101/logs/adam_vnet/'
    output_dir_test = 'results/adam_vnet`/'
    aug = None
    latest_checkpoint_file = 'checkpoint_latest.pt'
    total_epochs = 10000000
    epochs_per_checkpoint = 1
    batch_size = 4
    ckpt = None
    init_lr = 0.005
    scheduer_step_size = 20
    scheduer_gamma = 0.95
    debug = False
    mode = '3d' # '2d or '3d'
    in_class = 1
    out_class = 1

    crop_or_pad_size = 64,64,64
    patch_size = 64,64,64

    # for test
    patch_overlap = 48,48,48

    fold_arch = '*.nii*'

    save_arch = '.nii.gz'

    source_train_dir = 'dataset/train_source'
    label_train_dir = 'dataset/train_label'
    loc_train_dir = 'dataset/train_loc'

    source_test_dir = 'dataset/all_source'
    label_test_dir = 'dataset/all_label'

    # source_test_dir = 'dataset/test_source'
    # label_test_dir = 'dataset/test_label'

    # source_test_dir = 'dataset/zcmu_test_source'
    # label_test_dir = 'dataset/zcmu_test_label'
