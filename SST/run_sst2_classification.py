import tensorflow as tf
import csv
import os
import collections
import fcntl
from joblib import Parallel, delayed

import sys
sys.path.append(r"F:\project\bert\bert")
import tokenization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

# flags.DEFINE_string(
#     "bert_config_file", None,
#     "The config json file corresponding to the pre-trained BERT model. "
#     "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

class InputExample(object):
  """A single training/test example for simple sequence classification."""
  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample."""
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label

class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.label_id = label_id
    self.segment_ids = segment_ids

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      print(lines[:5])
      return lines

class SST2Processor(DataProcessor):
  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")),
        "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return [0, 1]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      sentence, sentiment = line.strip().split("\t")
      text_a = tokenization.convert_to_unicode(sentence)
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(sentiment)
      examples.append(
          InputExample(guid=guid, text_a=text_a, label=label))
    return examples

def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  Parallel(n_jobs=6, verbose=4, backend='mulpiprocessing')(
        delayed(TFRecoder_writer)(output_file, ex_index, example, label_list,
                    max_seq_length, tokenizer) for ex_index, example in enumerate(examples))

def TFRecoder_writer(output_file, ex_index, example, label_list,
                    max_seq_length, tokenizer):
  writer = tf.python_io.TFRecordWriter(output_file)

  fcntl.flock(writer.fileno(), fcntl.LOCK_EX) #加锁
  feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)
  def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(feature.input_ids)
  features["input_mask"] = create_int_feature(feature.input_mask)
  features["segment_ids"] = create_int_feature(feature.segment_ids)
  features["label_ids"] = create_int_feature([feature.label_id])

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  writer.write(tf_example.SerializeToString())
  
  writer.close()

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    '''SST数据集没有tokens_b'''
    # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    pass
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  input_mask = [1] * len(input_ids)
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    processor = SST2Processor()

    tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    label_list = processor.get_labels()
    train_examples = processor.get_train_examples(FLAGS.data_dir)

    # if FLAGS.do_train:
    #     train_examples = processor.get_train_examples(FLAGS.data_dir)
    #     num_train_steps = int(
    #         len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    #     num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        # tf.logging.info("***** Running training *****")
        # tf.logging.info("  Num examples = %d", len(train_examples))
        # tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        # tf.logging.info("  Num steps = %d", num_train_steps)