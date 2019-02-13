import miapy.config.configuration as miapy_cfg


class Configuration(miapy_cfg.ConfigurationBase):
    VERSION = 1
    TYPE = ''

    @classmethod
    def version(cls) -> int:
        return cls.VERSION

    @classmethod
    def type(cls) -> str:
        return cls.TYPE

    def __init__(self):
        self.hdf_file = 'dataset.h5'
        # Path to file with pre-defined subjects for train/validate/test:
        #   train_id1,train_id2,train_id3;val_id1,val_id2;test_id1,test_id2...
        self.subjects_train_val_test_file = None
        # Transformer:rotate:shift where
        #   rotate is boolean (True/False)
        #   shift the amount of pixels to shift
        # e.g. data.preprocess.RandomRotateTransform:1:10
        self.data_augment_transform = None
        self.batch_size = 5
        self.batch_size_eval = 20
        self.epochs = 15
        self.optimizer = 'Adam'
        self.learning_rate = 1e-05
        # SGD only:
        self.learning_rate_decay_steps = 100
        self.learning_rate_decay_rate = 0.8

        self.log_num_epoch = 20
        # visualize first layer of convolution every N epoch
        self.visualize_layer_num_epoch = 10
        # log r2 score on train and validation set every N epoch
        self.log_eval_num_epoch = 20
        # 'package.name' of a function returning the model (network)
        self.model = 'model.alexnet.conv_net_alexnet_modV5'
        # Replacements: %m for model, %t for timestamp
        self.checkpoint_dir = '../checkpoints/%m-%t'
        # Number of checkpoints to keep
        self.checkpoint_keep = 3
        # Intervall in seconds to save checkpoints
        self.checkpoint_save_interval = 3600

    def save(self, path: str):
        """Save a configuration file.

        Args:
            path (str): The path to the configuration file.
        """

        miapy_cfg.JSONConfigurationParser.save(path, self)

    @staticmethod
    def load(path: str):
        """Loads a configuration file.

        Args:
            path (str): The path to the configuration file.

        Returns:
            (config_cls): The configuration.
        """

        return miapy_cfg.JSONConfigurationParser.load(path, Configuration)


