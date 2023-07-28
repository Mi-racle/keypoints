import os
from pathlib import Path

import tensorflow as tf
import torch

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):

    def __init__(self, log_dir: Path):
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        self.log_dir = log_dir
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(str(log_dir))

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default(step=step):
            tf.summary.scalar(name=tag, data=value)

    def save_model(self, name, model):
        torch.save(model.state_dict(), self.log_dir / name)
